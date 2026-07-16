"""Evaluate a trained GHXTox checkpoint on a labeled dataset."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.metrics import binary_metrics
from ghxtox.nested_folds import load_nested_indices
from ghxtox.models import GHXToxModel
from ghxtox.preprocess import preprocess_fasta
from ghxtox.utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEVICE,
    DEFAULT_STRUCTURE_CACHE_DIR,
    DEFAULT_TEST_FASTA,
    DEFAULT_TEST_PROCESSED,
    DEFAULT_THRESHOLD,
    move_batch_to_device,
    resolve_inference_checkpoint,
    resolve_device,
    save_json,
)


def _prepare_input(args: argparse.Namespace) -> Path:
    if args.processed:
        processed = Path(args.processed)
        if processed.exists():
            return processed
    if not args.input_fasta:
        raise ValueError("Either --processed or --input-fasta is required.")
    output = Path(args.temp_processed)
    preprocess_fasta(
        input_path=args.input_fasta,
        output_path=output,
        structure_mode=args.structure_mode,
        structure_cache_dir=args.structure_cache_dir,
        max_length=args.max_length,
    )
    return output


def _write_predictions(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    threshold: float,
    model_checkpoint: str = "",
    fallback_used: bool = False,
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
                "global_3d_gate",
                "model_checkpoint",
                "fallback_used",
                "decision_threshold",
            ],
        )
        writer.writeheader()
        for row in rows:
            prob = float(row["toxicity_probability"])
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "label": int(row["label"]),
                    "toxicity_probability": f"{prob:.9g}",
                    "prediction": int(prob >= threshold),
                    "global_3d_gate": f"{float(row['global_3d_gate']):.9g}",
                    "model_checkpoint": model_checkpoint,
                    "fallback_used": int(fallback_used),
                    "decision_threshold": f"{threshold:.9g}",
                }
            )


def evaluate(args: argparse.Namespace) -> dict[str, float]:
    device = resolve_device(args.device)
    processed_path = _prepare_input(args)
    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    if args.nested_manifest:
        roles = load_nested_indices(args.nested_manifest, args.outer_fold, len(dataset))
        dataset = Subset(dataset, roles[args.nested_role])
    records = dataset.dataset.records if isinstance(dataset, Subset) else dataset.records
    checkpoint, selected_checkpoint, threshold, fallback_used = resolve_inference_checkpoint(
        args.checkpoint,
        records,
        device,
        requested_threshold=args.threshold,
    )
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(records, required_plm_dim, processed_path)
    modality = str(config.get("model", {}).get("modality", "fusion")).lower()
    include_structure = modality not in {"sequence_only", "atom_only", "sequence_atom"}
    include_atom = modality in {"atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"}
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_peptides,
            include_structure=include_structure,
            include_atom=include_atom,
        ),
    )

    logits_all = []
    labels_all = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            if labels is None:
                raise RuntimeError("Evaluation requires labels, but this batch has no labels.")
            output = model(batch)
            logits = output["logits"].detach().cpu()
            probs = torch.sigmoid(logits)
            gates = output["global_gate"].detach().cpu()
            labels_cpu = labels.detach().cpu()
            logits_all.append(logits)
            labels_all.append(labels_cpu)

            for sample_id, sequence, label, prob, gate in zip(
                batch["sample_id"],
                batch["sequence"],
                labels_cpu.tolist(),
                probs.tolist(),
                gates.tolist(),
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "sequence": sequence,
                        "label": label,
                        "toxicity_probability": prob,
                        "global_3d_gate": gate,
                    }
                )

    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    metrics = binary_metrics(logits_cat, labels_cat, threshold=threshold)
    metrics["threshold"] = float(threshold)
    metrics["mean_global_gate"] = float(
        sum(float(row["global_3d_gate"]) for row in rows) / max(len(rows), 1)
    )
    metrics["num_samples"] = float(len(rows))
    metrics["checkpoint_epoch"] = float(checkpoint.get("epoch", -1))
    metrics["model_checkpoint"] = selected_checkpoint
    metrics["fallback_used"] = bool(fallback_used)

    output_path = Path(args.output)
    save_json(metrics, output_path)
    if args.predictions:
        _write_predictions(
            rows,
            args.predictions,
            threshold,
            model_checkpoint=selected_checkpoint,
            fallback_used=fallback_used,
        )

    if fallback_used:
        print(
            "Chemical-site tensors were unavailable; "
            f"used fallback checkpoint {selected_checkpoint}."
        )
    print(f"Evaluation metrics saved to {output_path}")
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "auroc",
        "auprc",
        "mean_global_gate",
    ]:
        value = metrics[key]
        print(f"{key}: {value:.6f}")
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a GHXTox checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processed", default=DEFAULT_TEST_PROCESSED, help="Labeled preprocessed .pt file.")
    parser.add_argument("--input-fasta", default=DEFAULT_TEST_FASTA, help="Labeled raw FASTA file.")
    parser.add_argument("--output", default="runs/ghxtox/eval_metrics.json")
    parser.add_argument("--predictions", default=None, help="Optional labeled prediction CSV output.")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Decision threshold. Defaults to checkpoint metadata ({DEFAULT_THRESHOLD} for 3D-v2).",
    )
    parser.add_argument("--temp-processed", default="runs/ghxtox/eval_input.pt")
    parser.add_argument("--structure-mode", default="heuristic", choices=["heuristic", "cached"])
    parser.add_argument("--structure-cache-dir", default=DEFAULT_STRUCTURE_CACHE_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--nested-manifest", default=None)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--nested-role", choices=["train", "validation", "calibration", "test"], default="test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
