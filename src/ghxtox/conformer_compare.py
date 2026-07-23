"""Paired comparison of a frozen 3D baseline and conformer-residual OOF predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ghxtox.bootstrap_ci import _load_predictions
from ghxtox.paper_validation import paired_model_comparison
from ghxtox.utils import save_json


def compare_prediction_files(
    baseline: str | Path,
    candidate: str | Path,
    threshold: float,
    bootstrap_iterations: int = 1000,
    permutation_iterations: int = 1000,
) -> dict:
    first = _load_predictions(baseline)
    second = _load_predictions(candidate)
    if first["sample_ids"] != second["sample_ids"] or not np.array_equal(
        first["labels"], second["labels"]
    ):
        raise ValueError("Prediction files are not aligned by sample ID and label.")
    result = paired_model_comparison(
        first["labels"],
        first["probabilities"],
        second["probabilities"],
        first["probabilities"] >= threshold,
        second["probabilities"] >= threshold,
        bootstrap_iterations=bootstrap_iterations,
        permutation_iterations=permutation_iterations,
    )
    result["protocol"].update(
        {
            "baseline": str(baseline),
            "candidate": str(candidate),
            "fixed_threshold": float(threshold),
            "scope": "training-only group-aware OOF; exploratory, not independent confirmation",
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--threshold", type=float, default=0.677819)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--permutation-iterations", type=int, default=1000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare_prediction_files(
        args.baseline,
        args.candidate,
        args.threshold,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
    )
    save_json(result, args.output)


if __name__ == "__main__":
    main()
