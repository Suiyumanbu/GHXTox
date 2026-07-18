"""Leakage-controlled fusion of two aligned OOF prediction streams.

The fusion weight and decision threshold are selected only from training OOF
predictions.  A leave-one-fold-out screening pass estimates whether the
fusion generalizes before the final parameters are frozen on all OOF rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


def _read_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"sample_id", "sequence", "label", "toxicity_probability"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    return rows


def _aligned_arrays(
    baseline_path: str | Path,
    expert_path: str | Path,
    *,
    require_folds: bool,
    fold_manifest: str | Path | None = None,
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    baseline = _read_predictions(baseline_path)
    expert = _read_predictions(expert_path)
    if len(baseline) != len(expert):
        raise ValueError(f"Prediction row counts differ: {len(baseline)} versus {len(expert)}")
    for index, (left, right) in enumerate(zip(baseline, expert)):
        # Historical project assets use both ``peptide_1`` and
        # ``train_000001|1`` for the same row.  Permit that identifier alias
        # only after the immutable row index, sequence and label agree.
        for key in ("sequence", "label"):
            if left[key] != right[key]:
                raise ValueError(f"Prediction alignment mismatch at row {index}, column {key}.")
        if "source_index" in left and "source_index" in right:
            if int(left["source_index"]) != int(right["source_index"]):
                raise ValueError(
                    f"Prediction alignment mismatch at row {index}, column source_index."
                )
    labels = np.asarray([int(row["label"]) for row in baseline], dtype=np.int64)
    baseline_probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in baseline], dtype=np.float64
    )
    expert_probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in expert], dtype=np.float64
    )
    folds: np.ndarray | None = None
    if require_folds:
        if "fold" in baseline[0]:
            folds = np.asarray([int(row["fold"]) for row in baseline], dtype=np.int64)
        elif fold_manifest is not None:
            with Path(fold_manifest).open("r", encoding="utf-8", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
            by_index = {int(row["source_index"]): row for row in manifest_rows}
            derived_folds = []
            for position, row in enumerate(baseline):
                source_index = int(row.get("source_index", position))
                if source_index not in by_index:
                    raise ValueError(f"Fold manifest is missing source_index {source_index}.")
                manifest_row = by_index[source_index]
                for key in ("sequence", "label"):
                    if row[key] != manifest_row[key]:
                        raise ValueError(
                            f"Fold manifest mismatch at source_index {source_index}, column {key}."
                        )
                derived_folds.append(int(manifest_row["fold"]))
            folds = np.asarray(derived_folds, dtype=np.int64)
        else:
            raise ValueError(
                "The baseline OOF prediction file must contain a fold column, "
                "or fold_manifest must be provided."
            )
        if len(np.unique(folds)) < 2:
            raise ValueError("Cross-fitted fusion requires at least two folds.")
    return baseline, labels, baseline_probabilities, expert_probabilities, folds


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    labels_bool = np.asarray(labels, dtype=bool)
    predictions_bool = np.asarray(predictions, dtype=bool)
    tp = int(np.logical_and(predictions_bool, labels_bool).sum())
    tn = int(np.logical_and(~predictions_bool, ~labels_bool).sum())
    fp = int(np.logical_and(predictions_bool, ~labels_bool).sum())
    fn = int(np.logical_and(~predictions_bool, labels_bool).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    ranking = _probability_metrics(labels, probabilities, threshold=0.5)
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": float(ranking["auroc"]),
        "auprc": float(ranking["auprc"]),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _select_weight_threshold(
    labels: np.ndarray,
    baseline_probabilities: np.ndarray,
    expert_probabilities: np.ndarray,
    *,
    weight_step: float,
    threshold_step: float,
    objective: str,
) -> dict[str, float]:
    labels_bool = np.asarray(labels, dtype=bool)
    weights = np.arange(0.0, 1.0 + weight_step / 2.0, weight_step)
    thresholds = np.arange(threshold_step, 1.0, threshold_step)
    best: tuple[float, float, float, float, float] | None = None
    for expert_weight in weights:
        probabilities = (
            float(expert_weight) * expert_probabilities
            + (1.0 - float(expert_weight)) * baseline_probabilities
        )
        predictions = probabilities[:, None] >= thresholds[None, :]
        labels_column = labels_bool[:, None]
        tp = np.logical_and(predictions, labels_column).sum(axis=0, dtype=np.int64)
        tn = np.logical_and(~predictions, ~labels_column).sum(axis=0, dtype=np.int64)
        fp = np.logical_and(predictions, ~labels_column).sum(axis=0, dtype=np.int64)
        fn = np.logical_and(~predictions, labels_column).sum(axis=0, dtype=np.int64)
        denominator = np.sqrt(
            (tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = (tp * tn - fp * fn) / np.maximum(denominator, 1.0)
        balanced_accuracy = 0.5 * (
            tp / np.maximum(tp + fn, 1) + tn / np.maximum(tn + fp, 1)
        )
        if objective == "mcc":
            primary = mcc
            secondary = balanced_accuracy
        elif objective == "balanced_accuracy":
            primary = balanced_accuracy
            secondary = mcc
        elif objective == "balanced_mcc":
            primary = 0.5 * (balanced_accuracy + mcc)
            secondary = np.minimum(balanced_accuracy, mcc)
        else:
            raise ValueError(
                "objective must be 'mcc', 'balanced_accuracy', or 'balanced_mcc'."
            )
        index = int(np.lexsort((secondary, primary))[-1])
        candidate = (
            float(primary[index]),
            float(secondary[index]),
            -abs(float(expert_weight) - 0.5),
            float(expert_weight),
            float(thresholds[index]),
        )
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("Fusion grid search produced no candidate.")
    return {
        "expert_weight": best[3],
        "baseline_weight": 1.0 - best[3],
        "threshold": best[4],
        "selection_objective": objective,
        "training_primary_metric": best[0],
        "training_secondary_metric": best[1],
    }


def screen_oof_fusion(
    baseline_path: str | Path,
    expert_path: str | Path,
    output_dir: str | Path,
    *,
    weight_step: float = 0.01,
    threshold_step: float = 0.005,
    objective: str = "mcc",
    fold_manifest: str | Path | None = None,
    baseline_threshold: float = 0.5,
    expert_threshold: float = 0.5,
) -> dict[str, Any]:
    rows, labels, baseline_probabilities, expert_probabilities, folds = _aligned_arrays(
        baseline_path,
        expert_path,
        require_folds=True,
        fold_manifest=fold_manifest,
    )
    assert folds is not None
    crossfit_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    crossfit_predictions = np.zeros(len(labels), dtype=bool)
    fold_selection = []
    for fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != fold
        heldout_mask = folds == fold
        selected = _select_weight_threshold(
            labels[train_mask],
            baseline_probabilities[train_mask],
            expert_probabilities[train_mask],
            weight_step=weight_step,
            threshold_step=threshold_step,
            objective=objective,
        )
        probabilities = (
            selected["expert_weight"] * expert_probabilities[heldout_mask]
            + selected["baseline_weight"] * baseline_probabilities[heldout_mask]
        )
        crossfit_probabilities[heldout_mask] = probabilities
        crossfit_predictions[heldout_mask] = probabilities >= selected["threshold"]
        fold_selection.append({"heldout_fold": int(fold), **selected})
    if np.isnan(crossfit_probabilities).any():
        raise RuntimeError("At least one OOF row did not receive a cross-fitted fusion prediction.")

    frozen = _select_weight_threshold(
        labels,
        baseline_probabilities,
        expert_probabilities,
        weight_step=weight_step,
        threshold_step=threshold_step,
        objective=objective,
    )
    frozen_probabilities = (
        frozen["expert_weight"] * expert_probabilities
        + frozen["baseline_weight"] * baseline_probabilities
    )
    summary: dict[str, Any] = {
        "protocol": {
            "baseline_predictions": str(baseline_path),
            "expert_predictions": str(expert_path),
            "selection": f"leave-one-fold-out weight and threshold selection; {objective} primary",
            "weight_step": float(weight_step),
            "threshold_step": float(threshold_step),
            "num_samples": int(len(labels)),
            "num_folds": int(len(np.unique(folds))),
            "fold_manifest": None if fold_manifest is None else str(fold_manifest),
            "alignment": "rowwise source_index (when present), sequence and label; sample_id aliases permitted",
        },
        "baseline_at_declared_oof_threshold": _probability_metrics(
            labels, baseline_probabilities, threshold=float(baseline_threshold)
        ),
        "expert_at_declared_oof_threshold": _probability_metrics(
            labels, expert_probabilities, threshold=float(expert_threshold)
        ),
        "crossfit_fold_selection": fold_selection,
        "crossfit_metrics": _classification_metrics(
            labels, crossfit_predictions, crossfit_probabilities
        ),
        "frozen_parameters_from_all_oof": frozen,
        "frozen_oof_metrics": _probability_metrics(
            labels, frozen_probabilities, threshold=frozen["threshold"]
        ),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_path / "summary.json")
    with (output_path / "crossfit_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "source_index",
            "fold",
            "sample_id",
            "sequence",
            "label",
            "baseline_probability",
            "expert_probability",
            "fusion_probability",
            "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "fold": int(folds[index]),
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "label": int(labels[index]),
                    "baseline_probability": f"{baseline_probabilities[index]:.10g}",
                    "expert_probability": f"{expert_probabilities[index]:.10g}",
                    "fusion_probability": f"{crossfit_probabilities[index]:.10g}",
                    "prediction": int(crossfit_predictions[index]),
                }
            )
    return summary


def apply_frozen_fusion(
    baseline_path: str | Path,
    expert_path: str | Path,
    output_dir: str | Path,
    *,
    expert_weight: float,
    threshold: float,
) -> dict[str, Any]:
    rows, labels, baseline_probabilities, expert_probabilities, _ = _aligned_arrays(
        baseline_path, expert_path, require_folds=False
    )
    probabilities = (
        float(expert_weight) * expert_probabilities
        + (1.0 - float(expert_weight)) * baseline_probabilities
    )
    metrics = _probability_metrics(labels, probabilities, threshold=float(threshold))
    summary = {
        "protocol": {
            "baseline_predictions": str(baseline_path),
            "expert_predictions": str(expert_path),
            "expert_weight": float(expert_weight),
            "baseline_weight": 1.0 - float(expert_weight),
            "threshold": float(threshold),
            "parameters_selected_from_test_labels": False,
        },
        "metrics": metrics,
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_path / "metrics.json")
    with (output_path / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id",
            "sequence",
            "label",
            "baseline_probability",
            "expert_probability",
            "toxicity_probability",
            "prediction",
            "decision_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "label": int(labels[index]),
                    "baseline_probability": f"{baseline_probabilities[index]:.10g}",
                    "expert_probability": f"{expert_probabilities[index]:.10g}",
                    "toxicity_probability": f"{probabilities[index]:.10g}",
                    "prediction": int(probabilities[index] >= threshold),
                    "decision_threshold": f"{float(threshold):.10g}",
                }
            )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOF-selected two-model probability fusion.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen", help="Cross-fit and freeze OOF fusion parameters.")
    screen.add_argument("--baseline", required=True)
    screen.add_argument("--expert", required=True)
    screen.add_argument("--output-dir", required=True)
    screen.add_argument("--weight-step", type=float, default=0.01)
    screen.add_argument("--threshold-step", type=float, default=0.005)
    screen.add_argument("--fold-manifest")
    screen.add_argument("--baseline-threshold", type=float, default=0.5)
    screen.add_argument("--expert-threshold", type=float, default=0.5)
    screen.add_argument(
        "--objective",
        choices=["mcc", "balanced_accuracy", "balanced_mcc"],
        default="mcc",
    )
    apply = subparsers.add_parser("apply", help="Apply already frozen fusion parameters.")
    apply.add_argument("--baseline", required=True)
    apply.add_argument("--expert", required=True)
    apply.add_argument("--output-dir", required=True)
    apply.add_argument("--expert-weight", type=float, required=True)
    apply.add_argument("--threshold", type=float, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "screen":
        result = screen_oof_fusion(
            args.baseline,
            args.expert,
            args.output_dir,
            weight_step=args.weight_step,
            threshold_step=args.threshold_step,
            objective=args.objective,
            fold_manifest=args.fold_manifest,
            baseline_threshold=args.baseline_threshold,
            expert_threshold=args.expert_threshold,
        )
        print(json.dumps(result["crossfit_metrics"], indent=2))
        print(json.dumps(result["frozen_parameters_from_all_oof"], indent=2))
        return
    result = apply_frozen_fusion(
        args.baseline,
        args.expert,
        args.output_dir,
        expert_weight=args.expert_weight,
        threshold=args.threshold,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
