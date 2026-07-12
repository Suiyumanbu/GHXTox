"""Validation-constrained probability stacking for GHXTox checkpoints."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, random_split

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import DEFAULT_DEVICE, move_batch_to_device, resolve_device, save_json


def _validation_subset(dataset: PeptideTensorDataset, val_fraction: float, seed: int) -> Subset:
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return val_dataset


def _load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[GHXToxModel, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _collect_probs(
    checkpoint_path: str | Path,
    processed_path: str | Path,
    device_name: str,
    batch_size: int,
    validation: bool,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    model, checkpoint = _load_model(checkpoint_path, device)
    config = checkpoint["config"]
    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_plm_dim, processed_path)

    if validation:
        train_cfg = config.get("train", {})
        val_fraction = float(train_cfg.get("val_fraction", 0.15))
        seed = int(config.get("seed", 42))
        eval_dataset = _validation_subset(dataset, val_fraction, seed)
    else:
        eval_dataset = dataset

    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_peptides)
    sample_ids: list[str] = []
    sequences: list[str] = []
    labels: list[float] = []
    probs: list[float] = []
    gates: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            prob = torch.sigmoid(output["logits"]).detach().cpu()
            gate = output["global_gate"].detach().cpu()
            sample_ids.extend(batch["sample_id"])
            sequences.extend(batch["sequence"])
            labels.extend(batch["labels"].detach().cpu().tolist())
            probs.extend(prob.tolist())
            gates.extend(gate.tolist())

    return {
        "sample_id": sample_ids,
        "sequence": sequences,
        "labels": torch.tensor(labels, dtype=torch.float32),
        "probs": torch.tensor(probs, dtype=torch.float32),
        "gates": torch.tensor(gates, dtype=torch.float32),
    }


def _assert_aligned(reference: dict[str, Any], candidate: dict[str, Any], member_name: str) -> None:
    if reference["sample_id"] != candidate["sample_id"]:
        raise ValueError(f"Sample IDs are not aligned for ensemble member {member_name}.")
    if not torch.equal(reference["labels"], candidate["labels"]):
        raise ValueError(f"Labels are not aligned for ensemble member {member_name}.")


def _weight_grid(num_members: int, step: float) -> list[torch.Tensor]:
    if num_members <= 0:
        raise ValueError("At least one ensemble member is required.")
    denom = int(round(1.0 / step))
    if abs(denom * step - 1.0) > 1e-6:
        raise ValueError("--weight-step must evenly divide 1.0, e.g. 0.1, 0.05, 0.025.")

    weights: list[torch.Tensor] = []
    for cuts in combinations(range(denom + num_members - 1), num_members - 1):
        last = -1
        parts = []
        for cut in (*cuts, denom + num_members - 1):
            parts.append(cut - last - 1)
            last = cut
        weights.append(torch.tensor(parts, dtype=torch.float32) / float(denom))
    return weights


def _probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probs)


def _search_weights_and_threshold(
    member_probs: torch.Tensor,
    labels: torch.Tensor,
    metric: str,
    weight_step: float,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    thresholds = torch.linspace(0.01, 0.99, 99).tolist()
    thresholds = sorted(set(round(float(x), 6) for x in thresholds))
    best_weights = torch.zeros(member_probs.shape[0], dtype=torch.float32)
    best_weights[0] = 1.0
    best_threshold = 0.5
    best_metrics = binary_metrics(_probs_to_logits(member_probs[0]), labels, threshold=best_threshold)
    best_score = float(best_metrics[metric])

    for weights in _weight_grid(member_probs.shape[0], weight_step):
        probs = (weights.view(-1, 1) * member_probs).sum(dim=0)
        candidates = thresholds + [round(float(x), 6) for x in probs.unique().tolist()]
        for threshold in sorted(set(x for x in candidates if 0.0 < x < 1.0)):
            metrics = binary_metrics(_probs_to_logits(probs), labels, threshold=threshold)
            score = float(metrics[metric])
            if score > best_score:
                best_score = score
                best_weights = weights
                best_threshold = float(threshold)
                best_metrics = metrics
    return best_weights, best_threshold, best_metrics


def _write_predictions(
    output_path: str | Path,
    reference: dict[str, Any],
    probs: torch.Tensor,
    gates: torch.Tensor,
    threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sequence",
                "label",
                "toxicity_probability",
                "prediction",
                "mean_global_3d_gate",
            ],
        )
        writer.writeheader()
        for sample_id, sequence, label, prob, gate in zip(
            reference["sample_id"],
            reference["sequence"],
            reference["labels"].tolist(),
            probs.tolist(),
            gates.tolist(),
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": int(label),
                    "toxicity_probability": f"{float(prob):.6f}",
                    "prediction": int(float(prob) >= threshold),
                    "mean_global_3d_gate": f"{float(gate):.6f}",
                }
            )


def _parse_member(raw: str) -> dict[str, str]:
    parts = raw.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--member must be formatted as name|checkpoint|train_processed|eval_processed"
        )
    return {
        "name": parts[0],
        "checkpoint": parts[1],
        "train_processed": parts[2],
        "eval_processed": parts[3],
    }


def run_stacking(args: argparse.Namespace) -> dict[str, Any]:
    members = [_parse_member(item) for item in args.member]
    if len(members) < 2:
        raise ValueError("Stacking requires at least two --member values.")

    val_outputs = []
    eval_outputs = []
    for member in members:
        val_outputs.append(
            _collect_probs(
                member["checkpoint"],
                member["train_processed"],
                args.device,
                args.batch_size,
                validation=True,
            )
        )
        eval_outputs.append(
            _collect_probs(
                member["checkpoint"],
                member["eval_processed"],
                args.device,
                args.batch_size,
                validation=False,
            )
        )

    val_ref = val_outputs[0]
    eval_ref = eval_outputs[0]
    for member, val_output, eval_output in zip(members[1:], val_outputs[1:], eval_outputs[1:]):
        _assert_aligned(val_ref, val_output, member["name"])
        _assert_aligned(eval_ref, eval_output, member["name"])

    val_probs = torch.stack([item["probs"] for item in val_outputs])
    eval_probs = torch.stack([item["probs"] for item in eval_outputs])
    eval_gates = torch.stack([item["gates"] for item in eval_outputs]).mean(dim=0)

    weights, threshold, val_metrics = _search_weights_and_threshold(
        val_probs,
        val_ref["labels"],
        metric=args.metric,
        weight_step=args.weight_step,
    )
    ensemble_eval_probs = (weights.view(-1, 1) * eval_probs).sum(dim=0)
    eval_metrics = binary_metrics(_probs_to_logits(ensemble_eval_probs), eval_ref["labels"], threshold=threshold)

    result = {
        "members": [member["name"] for member in members],
        "weights": {member["name"]: float(weight) for member, weight in zip(members, weights.tolist())},
        "threshold": float(threshold),
        "metric": args.metric,
        "weight_step": float(args.weight_step),
        "num_validation_samples": float(val_ref["labels"].numel()),
        "num_eval_samples": float(eval_ref["labels"].numel()),
        **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        **{f"eval_{key}": float(value) for key, value in eval_metrics.items()},
    }
    save_json(result, args.output)
    if args.predictions:
        _write_predictions(args.predictions, eval_ref, ensemble_eval_probs, eval_gates, threshold)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-constrained GHXTox probability stacking.")
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="Ensemble member formatted as name|checkpoint|train_processed|eval_processed.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"], default="mcc")
    parser.add_argument("--weight-step", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_stacking(args)
    print(f"weights: {result['weights']}")
    print(f"threshold: {result['threshold']:.6f}")
    print(f"val_{result['metric']}: {result['val_' + result['metric']]:.6f}")
    print(f"eval_{result['metric']}: {result['eval_' + result['metric']]:.6f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
