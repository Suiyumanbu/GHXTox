"""Evaluate frozen fold-specific GHXTox models on generated conformer bags.

The script is deliberately OOF-oriented: every training sample is evaluated by
the checkpoint for the fold in which that sample was held out. It reports fixed
equal-weight aggregation rules and never searches a test-set threshold.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ghxtox.data import collate_peptides, validate_plm_feature_dim
from ghxtox.geometry_features import structure_feature_matrix
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_device, save_json


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _local_cache_path(manifest: str | Path, value: str) -> str:
    """Resolve server-authored absolute cache paths after an archive is moved locally."""

    supplied = Path(value)
    if supplied.exists():
        return str(supplied)
    parts = value.replace("\\", "/").rstrip("/").split("/")
    if len(parts) >= 2:
        relocated = Path(manifest).parent / parts[-2] / parts[-1]
        if relocated.exists():
            return str(relocated)
    return value


def record_from_conformer_cache(
    base_record: dict[str, Any],
    cache_path: str | Path,
    confidence_mode: str = "base",
) -> dict[str, Any]:
    payload = np.load(cache_path)
    sequence = str(payload["sequence"].item())
    if sequence != base_record["sequence"]:
        raise ValueError(
            f"Conformer sequence mismatch: base={base_record['sequence']}, cache={sequence}, "
            f"path={cache_path}"
        )
    item = dict(base_record)
    coords = torch.as_tensor(payload["coords"], dtype=torch.float32)
    if confidence_mode == "base":
        plddt = base_record["plddt"].detach().cpu().float().clone()
    else:
        plddt = torch.as_tensor(payload["plddt"], dtype=torch.float32)
    item.update(
        {
            "coords": coords,
            "plddt": plddt,
            "backbone_coords": torch.as_tensor(payload["backbone_coords"], dtype=torch.float32),
            "backbone_mask": torch.as_tensor(payload["backbone_mask"], dtype=torch.bool),
            "functional_group_coords": torch.as_tensor(
                payload["functional_group_coords"], dtype=torch.float32
            ),
            "functional_group_mask": torch.as_tensor(payload["functional_group_mask"], dtype=torch.bool),
            "chemical_site_coords": torch.as_tensor(payload["chemical_site_coords"], dtype=torch.float32),
            "chemical_site_types": torch.as_tensor(payload["chemical_site_types"], dtype=torch.float32),
            "chemical_site_orientations": torch.as_tensor(
                payload["chemical_site_orientations"], dtype=torch.float32
            ),
            "chemical_site_orientation_mask": torch.as_tensor(
                payload["chemical_site_orientation_mask"], dtype=torch.bool
            ),
            "chemical_site_mask": torch.as_tensor(payload["chemical_site_mask"], dtype=torch.bool),
            "structure_features": structure_feature_matrix(coords, plddt),
            "structure_source": str(cache_path),
        }
    )
    return item


def _metrics_from_probabilities(probabilities: list[float], labels: list[int], threshold: float):
    probs = torch.tensor(probabilities, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
    logits = torch.logit(probs)
    return binary_metrics(logits, torch.tensor(labels, dtype=torch.float32), threshold=threshold)


def evaluate_oof_conformer_ensemble(
    processed: str | Path,
    conformer_manifest: str | Path,
    fold_manifest: str | Path,
    checkpoint_pattern: str,
    output_dir: str | Path,
    threshold: float = 0.677819,
    confidence_mode: str = "base",
    device_name: str = "cuda",
    max_conformers: int | None = None,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = {str(row["sample_id"]): row for row in payload["records"]}
    record_order = {str(row["sample_id"]): index for index, row in enumerate(payload["records"])}
    folds = {str(row["sample_id"]): int(row["fold"]) for row in _read_csv(fold_manifest)}
    conformers: dict[str, list[str]] = {}
    for row in _read_csv(conformer_manifest):
        candidates = [str(row.get("original_id") or ""), str(row["sample_id"])]
        sample_id = next((value for value in candidates if value in records), candidates[-1])
        conformers.setdefault(sample_id, []).append(
            _local_cache_path(conformer_manifest, str(row["cache_path"]))
        )
    for sample_id in conformers:
        conformers[sample_id] = sorted(conformers[sample_id])
        if max_conformers is not None:
            conformers[sample_id] = conformers[sample_id][: max(int(max_conformers), 0)]

    models: dict[int, GHXToxModel] = {}
    output_rows: list[dict[str, Any]] = []
    for sample_id in sorted(conformers, key=lambda value: record_order.get(value, len(record_order))):
        if sample_id not in records or sample_id not in folds:
            raise ValueError(f"Sample {sample_id} is absent from the processed data or fold manifest.")
        fold = folds[sample_id]
        if fold not in models:
            checkpoint_path = checkpoint_pattern.format(fold=fold)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            config = checkpoint["config"]
            model = GHXToxModel(config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            validate_plm_feature_dim(
                payload["records"],
                int(config.get("model", {}).get("plm_embedding_dim", 0)),
                processed,
            )
            models[fold] = model
        model = models[fold]
        base = records[sample_id]
        bag = [base] + [
            record_from_conformer_cache(base, path, confidence_mode=confidence_mode)
            for path in conformers[sample_id]
        ]
        batch = move_batch_to_device(collate_peptides(bag, include_structure=True, include_atom=False), device)
        with torch.no_grad():
            output = model(batch)
            probabilities = torch.sigmoid(output["logits"]).detach().cpu().numpy()
            gates = output["global_gate"].detach().cpu().numpy()
        base_probability = float(probabilities[0])
        generated = probabilities[1:]
        output_rows.append(
            {
                "sample_id": sample_id,
                "fold": fold,
                "label": int(base["label"]),
                "sequence": base["sequence"],
                "num_generated_conformers": int(len(generated)),
                "base_probability": base_probability,
                "conformer_mean_probability": float(generated.mean()),
                "base_plus_conformer_mean_probability": float(probabilities.mean()),
                "conformer_probability_std": float(generated.std()),
                "conformer_probability_min": float(generated.min()),
                "conformer_probability_max": float(generated.max()),
                "base_gate": float(gates[0]),
                "conformer_mean_gate": float(gates[1:].mean()),
            }
        )

    labels = [int(row["label"]) for row in output_rows]
    if not output_rows:
        raise ValueError("The conformer manifest did not yield any evaluable processed sample.")
    methods = {
        "base_single_esmfold": [float(row["base_probability"]) for row in output_rows],
        "pepflow_conformer_mean": [float(row["conformer_mean_probability"]) for row in output_rows],
        "esmfold_plus_pepflow_mean": [
            float(row["base_plus_conformer_mean_probability"]) for row in output_rows
        ],
    }
    metrics = {
        name: _metrics_from_probabilities(values, labels, threshold) for name, values in methods.items()
    }
    base_metrics = metrics["base_single_esmfold"]
    deltas = {
        name: {
            metric: float(values[metric] - base_metrics[metric])
            for metric in ("balanced_accuracy", "mcc", "auroc", "auprc", "brier", "ece_10")
        }
        for name, values in metrics.items()
        if name != "base_single_esmfold"
    }
    summary = {
        "protocol": "fold-specific group-aware OOF; fixed threshold and equal conformer weights",
        "threshold": float(threshold),
        "confidence_mode": confidence_mode,
        "samples": len(output_rows),
        "metrics": metrics,
        "delta_vs_single_esmfold": deltas,
        "decision_boundary": (
            "Exploratory OOF evidence only. Do not replace the default unless the full predeclared "
            "OOF cohort improves MCC without material BACC/AUROC/AUPRC regression."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--conformer-manifest", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--checkpoint-pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.677819)
    parser.add_argument("--confidence-mode", choices=["base", "cache"], default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-conformers", type=int)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = evaluate_oof_conformer_ensemble(
        processed=args.processed,
        conformer_manifest=args.conformer_manifest,
        fold_manifest=args.fold_manifest,
        checkpoint_pattern=args.checkpoint_pattern,
        output_dir=args.output_dir,
        threshold=args.threshold,
        confidence_mode=args.confidence_mode,
        device_name=args.device,
        max_conformers=args.max_conformers,
    )
    print(summary["metrics"])


if __name__ == "__main__":
    main()
