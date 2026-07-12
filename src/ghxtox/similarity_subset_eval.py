"""Evaluate saved predictions after excluding train-similar test sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.bootstrap_ci import _metrics


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_similarity_subsets(
    audit_rows_path: str | Path,
    prediction_paths: list[str | Path],
    thresholds: tuple[float, ...] = (0.9, 0.8),
    decision_threshold: float = 0.85,
) -> dict[str, Any]:
    audit_rows = _read_csv(audit_rows_path)
    results = []
    for prediction_path in prediction_paths:
        predictions = _read_csv(prediction_path)
        if len(predictions) != len(audit_rows):
            raise ValueError(
                f"Audit/prediction row count mismatch: {len(audit_rows)} != {len(predictions)}"
            )
        for index, (audit_row, prediction_row) in enumerate(zip(audit_rows, predictions)):
            if audit_row.get("query_sequence") != prediction_row.get("sequence"):
                raise ValueError(f"Audit/prediction sequence mismatch at row {index}.")
            if int(audit_row["query_label"]) != int(prediction_row["label"]):
                raise ValueError(f"Audit/prediction label mismatch at row {index}.")
        run = {"path": str(prediction_path), "subsets": {}}
        for threshold in thresholds:
            retained = [
                prediction_row
                for audit_row, prediction_row in zip(audit_rows, predictions)
                if float(audit_row["max_identity"]) < threshold
            ]
            labels = np.asarray([int(row["label"]) for row in retained], dtype=np.int64)
            probabilities = np.asarray(
                [float(row["toxicity_probability"]) for row in retained], dtype=np.float64
            )
            if labels.size == 0 or np.unique(labels).size < 2:
                raise ValueError(f"Subset identity<{threshold} must contain both classes.")
            run["subsets"][str(threshold)] = {
                "num_samples": int(labels.size),
                "num_positive": int(labels.sum()),
                "num_negative": int((labels == 0).sum()),
                "metrics": _metrics(labels, probabilities, decision_threshold),
            }
        results.append(run)

    aggregate: dict[str, Any] = {}
    for threshold in thresholds:
        key = str(threshold)
        metric_names = results[0]["subsets"][key]["metrics"]
        aggregate[key] = {
            "num_samples": results[0]["subsets"][key]["num_samples"],
            "num_positive": results[0]["subsets"][key]["num_positive"],
            "num_negative": results[0]["subsets"][key]["num_negative"],
            "metrics": {
                name: {
                    "mean": float(np.mean([run["subsets"][key]["metrics"][name] for run in results])),
                    "sample_std": float(np.std(
                        [run["subsets"][key]["metrics"][name] for run in results], ddof=1
                    )),
                }
                for name in metric_names
            },
        }
    return {
        "protocol": {
            "method": "exclude_test_samples_with_train_identity_at_or_above_threshold",
            "audit_rows": str(audit_rows_path),
            "identity_thresholds": list(thresholds),
            "decision_threshold": float(decision_threshold),
            "num_model_seeds": len(results),
        },
        "runs": results,
        "aggregate": aggregate,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate low-train-similarity test subsets.")
    parser.add_argument("--audit-rows", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.8])
    parser.add_argument("--decision-threshold", type=float, default=0.85)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = evaluate_similarity_subsets(
        audit_rows_path=args.audit_rows,
        prediction_paths=args.predictions,
        thresholds=tuple(args.thresholds),
        decision_threshold=args.decision_threshold,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Similarity-subset metrics saved to {output_path}")
    for threshold, subset in result["aggregate"].items():
        metrics = subset["metrics"]
        print(
            f"identity<{threshold}: n={subset['num_samples']} pos={subset['num_positive']} "
            f"mcc={metrics['mcc']['mean']:.4f}+/-{metrics['mcc']['sample_std']:.4f} "
            f"f1={metrics['f1']['mean']:.4f}+/-{metrics['f1']['sample_std']:.4f} "
            f"auprc={metrics['auprc']['mean']:.4f}+/-{metrics['auprc']['sample_std']:.4f}"
        )


if __name__ == "__main__":
    main()
