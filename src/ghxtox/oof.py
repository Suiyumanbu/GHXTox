"""Collect and evaluate fixed-fold out-of-fold predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.folds import load_fold_indices
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_device


def _enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable dropout sampling while keeping all other modules in evaluation mode."""

    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def _confidence_diagnostics(
    batch: dict,
    output: dict[str, torch.Tensor],
    probabilities: torch.Tensor,
    mc_probabilities: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    mask = batch["mask"].bool()
    lengths = mask.sum(dim=1).clamp_min(1)
    plddt = batch["plddt"].float().clamp(0.0, 1.0)
    valid_plddt = plddt.masked_fill(~mask, 0.0)
    mean_plddt = valid_plddt.sum(dim=1) / lengths
    min_plddt = plddt.masked_fill(~mask, 1.0).min(dim=1).values
    low_plddt_fraction = ((plddt < 0.55) & mask).sum(dim=1).float() / lengths
    clipped = probabilities.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log()) / math.log(2.0)
    global_gate = output.get("global_gate")
    if global_gate is None:
        global_gate = torch.zeros_like(probabilities)
    if mc_probabilities is None:
        mc_mean = probabilities
        mc_std = torch.zeros_like(probabilities)
    else:
        mc_mean = mc_probabilities.mean(dim=0)
        mc_std = mc_probabilities.std(dim=0, unbiased=False)
    return {
        "probability_margin": (2.0 * probabilities - 1.0).abs(),
        "probability_entropy": entropy,
        "mc_mean_probability": mc_mean,
        "mc_std_probability": mc_std,
        "mean_plddt": mean_plddt,
        "min_plddt": min_plddt,
        "low_plddt_fraction": low_plddt_fraction,
        "global_gate": global_gate.reshape(-1),
        "sequence_length": lengths.float(),
    }


def _best_threshold(logits: torch.Tensor, labels: torch.Tensor, metric: str) -> tuple[float, dict]:
    probabilities = torch.sigmoid(logits)
    candidates = torch.linspace(0.01, 0.99, 99).tolist()
    candidates.extend(float(value) for value in probabilities.unique().tolist())
    candidates = sorted(set(round(float(value), 6) for value in candidates if 0.0 < value < 1.0))
    threshold = 0.5
    metrics = binary_metrics(logits, labels, threshold=threshold)
    score = float(metrics[metric])
    for candidate in candidates:
        candidate_metrics = binary_metrics(logits, labels, threshold=candidate)
        candidate_score = float(candidate_metrics[metric])
        if candidate_score > score:
            threshold = candidate
            metrics = candidate_metrics
            score = candidate_score
    return threshold, metrics


def evaluate_oof(
    checkpoints: list[str | Path],
    processed: str | Path,
    fold_manifest: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    device_name: str = "cuda",
    batch_size: int = 64,
    metric: str = "mcc",
    mc_samples: int = 0,
) -> dict:
    device = resolve_device(device_name)
    dataset = PeptideTensorDataset(processed, require_labels=True)
    rows = []
    fold_metrics = []
    logits_by_index: dict[int, torch.Tensor] = {}
    labels_by_index: dict[int, torch.Tensor] = {}

    for fold, checkpoint_path in enumerate(checkpoints):
        _, validation_indices = load_fold_indices(fold_manifest, fold, len(dataset))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
        validate_plm_feature_dim(dataset.records, required_dim, processed)
        model = GHXToxModel(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        modality = str(config.get("model", {}).get("modality", "fusion")).lower()
        sequence_only = modality == "sequence_only"
        include_atom = modality in {"atom_only", "sequence_atom", "fusion_atom_residual"}
        subset = Subset(dataset, validation_indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_peptides,
                include_structure=not sequence_only,
                include_atom=include_atom,
            ),
        )
        fold_logits = []
        fold_labels = []
        position = 0
        with torch.no_grad():
            for batch in loader:
                batch_on_device = move_batch_to_device(batch, device)
                model.eval()
                output = model(batch_on_device)
                logits = output["logits"].detach().cpu()
                labels = batch_on_device["labels"].detach().cpu()
                probabilities = torch.sigmoid(logits)
                mc_probabilities = None
                if mc_samples > 0:
                    torch.manual_seed(17000 + fold * 1000 + position)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(17000 + fold * 1000 + position)
                    _enable_mc_dropout(model)
                    mc_probabilities = torch.stack(
                        [torch.sigmoid(model(batch_on_device)["logits"]).detach().cpu() for _ in range(mc_samples)]
                    )
                    model.eval()
                diagnostics = _confidence_diagnostics(
                    batch_on_device,
                    {
                        "global_gate": output["global_gate"].detach().cpu()
                        if torch.is_tensor(output.get("global_gate"))
                        else None
                    },
                    probabilities,
                    mc_probabilities,
                )
                indices = validation_indices[position : position + len(logits)]
                position += len(logits)
                for row_offset, (index, sample_id, sequence, label, logit, probability) in enumerate(zip(
                    indices,
                    batch["sample_id"],
                    batch["sequence"],
                    labels.tolist(),
                    logits.tolist(),
                    probabilities.tolist(),
                )):
                    if index in logits_by_index:
                        raise RuntimeError(f"Duplicate OOF prediction for source index {index}.")
                    logits_by_index[index] = torch.tensor(logit)
                    labels_by_index[index] = torch.tensor(label)
                    row = {
                            "source_index": index,
                            "fold": fold,
                            "sample_id": sample_id,
                            "sequence": sequence,
                            "label": int(label),
                            "logit": f"{float(logit):.9g}",
                            "toxicity_probability": f"{float(probability):.9g}",
                    }
                    for key, values in diagnostics.items():
                        row[key] = f"{float(values[row_offset]):.9g}"
                    rows.append(row)
                fold_logits.append(logits)
                fold_labels.append(labels)
        fold_metrics.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                **binary_metrics(torch.cat(fold_logits), torch.cat(fold_labels), threshold=0.5),
            }
        )

    if set(logits_by_index) != set(range(len(dataset))):
        raise RuntimeError(f"OOF coverage is {len(logits_by_index)}/{len(dataset)}.")
    ordered_logits = torch.stack([logits_by_index[index] for index in range(len(dataset))]).float()
    ordered_labels = torch.stack([labels_by_index[index] for index in range(len(dataset))]).float()
    threshold, optimized = _best_threshold(ordered_logits, ordered_labels, metric)
    summary = {
        "protocol": {
            "processed": str(processed),
            "fold_manifest": str(fold_manifest),
            "num_folds": len(checkpoints),
            "num_samples": len(dataset),
            "threshold_selection": f"all OOF predictions maximize {metric}",
            "mc_dropout_samples": mc_samples,
        },
        "fold_metrics_at_0.5": fold_metrics,
        "aggregate_at_0.5": binary_metrics(ordered_logits, ordered_labels, threshold=0.5),
        "oof_threshold": threshold,
        "aggregate_at_oof_threshold": optimized,
    }
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["source_index"])))
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed-fold OOF checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", default="mcc", choices=["accuracy", "balanced_accuracy", "f1", "mcc"])
    parser.add_argument("--mc-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_oof(
        args.checkpoints,
        args.processed,
        args.fold_manifest,
        args.output_csv,
        args.output_json,
        device_name=args.device,
        batch_size=args.batch_size,
        metric=args.metric,
        mc_samples=args.mc_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
