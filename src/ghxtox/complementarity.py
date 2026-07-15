"""Describe error complementarity between two frozen prediction systems."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ghxtox.metrics import binary_metrics


def _read_ensemble(paths: list[str | Path]) -> tuple[list[dict[str, str]], np.ndarray]:
    tables = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            tables.append(list(csv.DictReader(handle)))
    reference = tables[0]
    for table in tables[1:]:
        if len(table) != len(reference):
            raise ValueError("Prediction files have different row counts.")
        for first, second in zip(reference, table):
            if (first["sequence"], first["label"]) != (second["sequence"], second["label"]):
                raise ValueError("Prediction files are not row-aligned by sequence and label.")
    probabilities = np.mean(
        [[float(row["toxicity_probability"]) for row in table] for table in tables], axis=0
    )
    return reference, probabilities


def _metrics(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    logits = torch.logit(torch.tensor(probabilities, dtype=torch.float32).clamp(1e-7, 1 - 1e-7))
    return binary_metrics(logits, torch.tensor(labels, dtype=torch.float32), threshold=threshold)


def analyze_complementarity(
    first_paths: list[str | Path],
    second_paths: list[str | Path],
    processed: str | Path,
    first_threshold: float,
    second_threshold: float,
    first_name: str,
    second_name: str,
) -> dict:
    rows, first_probabilities = _read_ensemble(first_paths)
    second_rows, second_probabilities = _read_ensemble(second_paths)
    for first, second in zip(rows, second_rows):
        if (first["sequence"], first["label"]) != (second["sequence"], second["label"]):
            raise ValueError("The two systems are not row-aligned.")
    labels = np.asarray([int(row["label"]) for row in rows])
    first_predictions = first_probabilities >= first_threshold
    second_predictions = second_probabilities >= second_threshold
    first_correct = first_predictions == labels
    second_correct = second_predictions == labels

    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    if len(records) != len(rows):
        raise ValueError("Processed dataset and predictions have different lengths.")
    for record, row in zip(records, rows):
        if str(record["sequence"]) != row["sequence"]:
            raise ValueError("Processed dataset and predictions are not sequence-aligned.")
    mean_plddt = np.asarray(
        [float(record["plddt"].float().mean()) for record in records], dtype=float
    )
    lengths = np.asarray([len(row["sequence"]) for row in rows])

    def subset(mask: np.ndarray) -> dict:
        return {
            "num_samples": int(mask.sum()),
            "num_positive": int(labels[mask].sum()),
            first_name: _metrics(first_probabilities[mask], labels[mask], first_threshold),
            second_name: _metrics(second_probabilities[mask], labels[mask], second_threshold),
        }

    return {
        "protocol": {
            first_name: {"num_seeds": len(first_paths), "threshold": first_threshold},
            second_name: {"num_seeds": len(second_paths), "threshold": second_threshold},
            "ensemble_probability": "mean across seeds before thresholding",
            "note": "Descriptive frozen-test analysis; no ensemble weight is selected here.",
        },
        "probability_pearson_correlation": float(
            np.corrcoef(first_probabilities, second_probabilities)[0, 1]
        ),
        "prediction_disagreement": int((first_predictions != second_predictions).sum()),
        "error_overlap": {
            "both_correct": int((first_correct & second_correct).sum()),
            f"only_{first_name}_correct": int((first_correct & ~second_correct).sum()),
            f"only_{second_name}_correct": int((~first_correct & second_correct).sum()),
            "both_wrong": int((~first_correct & ~second_correct).sum()),
        },
        "full": subset(np.ones(len(labels), dtype=bool)),
        "strata": {
            "plddt_ge_0.70": subset(mean_plddt >= 0.70),
            "plddt_lt_0.70": subset(mean_plddt < 0.70),
            "length_11_20": subset(lengths <= 20),
            "length_21_35": subset((lengths >= 21) & (lengths <= 35)),
            "length_36_50": subset(lengths >= 36),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze two frozen multi-seed prediction systems.")
    parser.add_argument("--first", nargs="+", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--first-threshold", type=float, required=True)
    parser.add_argument("--second-threshold", type=float, required=True)
    parser.add_argument("--first-name", default="first")
    parser.add_argument("--second-name", default="second")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = analyze_complementarity(
        args.first,
        args.second,
        args.processed,
        args.first_threshold,
        args.second_threshold,
        args.first_name,
        args.second_name,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
