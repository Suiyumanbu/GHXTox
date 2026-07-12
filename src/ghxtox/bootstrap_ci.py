"""Stratified bootstrap confidence intervals for saved prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


METRIC_NAMES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc")


def _load_predictions(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"sample_id", "label", "toxicity_probability"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    return {
        "path": str(path),
        "sample_ids": [row["sample_id"] for row in rows],
        "labels": np.asarray([int(row["label"]) for row in rows], dtype=np.int64),
        "probabilities": np.asarray([float(row["toxicity_probability"]) for row in rows], dtype=np.float64),
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    tp = float(np.sum(predictions & positives))
    tn = float(np.sum(~predictions & negatives))
    fp = float(np.sum(predictions & negatives))
    fn = float(np.sum(~predictions & positives))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("Stratified bootstrap requires both negative and positive samples.")
    sampled = [rng.choice(group, size=group.size, replace=True) for group in groups]
    return np.concatenate(sampled)


def _interval(samples: dict[str, list[float]], confidence: float) -> dict[str, dict[str, float]]:
    tail = (1.0 - confidence) / 2.0
    return {
        name: {
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
        }
        for name, values in samples.items()
    }


def bootstrap_confidence_intervals(
    prediction_paths: list[str | Path],
    threshold: float = 0.85,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    runs = [_load_predictions(path) for path in prediction_paths]
    reference_ids = runs[0]["sample_ids"]
    reference_labels = runs[0]["labels"]
    for run in runs[1:]:
        if run["sample_ids"] != reference_ids or not np.array_equal(run["labels"], reference_labels):
            raise ValueError("All prediction files must contain the same samples, order, and labels.")

    rng = np.random.default_rng(seed)
    run_results = []
    for run in runs:
        samples = {name: [] for name in METRIC_NAMES}
        for _ in range(iterations):
            indices = _stratified_indices(reference_labels, rng)
            values = _metrics(reference_labels[indices], run["probabilities"][indices], threshold)
            for name in METRIC_NAMES:
                samples[name].append(values[name])
        run_results.append(
            {
                "path": run["path"],
                "point_estimate": _metrics(reference_labels, run["probabilities"], threshold),
                "confidence_interval": _interval(samples, confidence),
            }
        )

    hierarchical_samples = {name: [] for name in METRIC_NAMES}
    for _ in range(iterations):
        run = runs[int(rng.integers(len(runs)))]
        indices = _stratified_indices(reference_labels, rng)
        values = _metrics(reference_labels[indices], run["probabilities"][indices], threshold)
        for name in METRIC_NAMES:
            hierarchical_samples[name].append(values[name])

    aggregate_point = {
        name: float(np.mean([result["point_estimate"][name] for result in run_results]))
        for name in METRIC_NAMES
    }
    return {
        "protocol": {
            "method": "stratified_percentile_bootstrap",
            "aggregate_method": "hierarchical_seed_then_stratified_sample_bootstrap",
            "iterations": int(iterations),
            "confidence": float(confidence),
            "random_seed": int(seed),
            "threshold": float(threshold),
            "num_samples": int(reference_labels.size),
            "num_positive": int(reference_labels.sum()),
            "num_negative": int((reference_labels == 0).sum()),
            "num_model_seeds": len(runs),
        },
        "runs": run_results,
        "aggregate": {
            "point_estimate": aggregate_point,
            "confidence_interval": _interval(hierarchical_samples, confidence),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals from GHXTox prediction CSV files.")
    parser.add_argument("--predictions", nargs="+", required=True, help="Aligned prediction CSV files.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026, help="Bootstrap random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = bootstrap_confidence_intervals(
        prediction_paths=args.predictions,
        threshold=args.threshold,
        iterations=args.iterations,
        confidence=args.confidence,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Bootstrap confidence intervals saved to {output_path}")
    for name in ("mcc", "f1", "auroc", "auprc"):
        point = result["aggregate"]["point_estimate"][name]
        interval = result["aggregate"]["confidence_interval"][name]
        print(f"{name}: {point:.6f} [{interval['lower']:.6f}, {interval['upper']:.6f}]")


if __name__ == "__main__":
    main()
