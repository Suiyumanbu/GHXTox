from __future__ import annotations

import csv

import pytest

from ghxtox.nested_folds import create_nested_group_folds
from ghxtox.nonnegative_stacking import (
    run_nested_calibrated_stacking,
    run_nonnegative_stacking,
)


def _write(path, probabilities, *, baseline=False) -> None:
    fieldnames = [
        "source_index",
        "fold",
        "sample_id",
        "sequence",
        "label",
        "toxicity_probability",
    ]
    if baseline:
        fieldnames.append("prediction")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            row = {
                "source_index": index,
                "fold": index % 5,
                "sample_id": f"sample_{index}",
                "sequence": f"SEQ{index}",
                "label": index % 2,
                "toxicity_probability": probability,
            }
            if baseline:
                row["prediction"] = int(probability >= 0.5)
            writer.writerow(row)


def test_nonnegative_stacking_crossfits_and_writes_auditable_summary(tmp_path) -> None:
    first = []
    second = []
    noise = []
    for index in range(50):
        label = index % 2
        first.append((0.82 if label else 0.18) if index < 25 else (0.45 if label else 0.55))
        second.append((0.45 if label else 0.55) if index < 25 else (0.82 if label else 0.18))
        noise.append(0.52 if label else 0.48)
    paths = []
    for name, values in (("first", first), ("second", second), ("noise", noise)):
        path = tmp_path / f"{name}.csv"
        _write(path, values)
        paths.append(path)
    baseline = tmp_path / "baseline.csv"
    _write(baseline, first, baseline=True)
    summary = run_nonnegative_stacking(
        paths,
        baseline,
        tmp_path / "stacking",
        member_names=["first", "second", "noise"],
        min_sn=None,
        min_sp=None,
        steps=120,
        restarts=1,
        seed=7,
    )
    assert summary["protocol"]["test1_or_test2_predictions_read"] is False
    assert len(summary["fold_results"]) == 5
    for fold in summary["fold_results"]:
        weights = list(fold["weights"].values())
        assert all(weight >= 0.0 for weight in weights)
        assert sum(weights) == pytest.approx(1.0)
    assert (tmp_path / "stacking" / "summary.json").exists()
    assert (tmp_path / "stacking" / "crossfit_predictions.csv").exists()


def test_nested_stacking_keeps_calibration_and_outer_test_disjoint(tmp_path) -> None:
    probabilities = []
    complement = []
    for index in range(100):
        label = index % 2
        probabilities.append(0.8 if label else 0.2)
        complement.append(0.7 if label else 0.3)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    baseline = tmp_path / "baseline.csv"
    _write(first, probabilities)
    _write(second, complement)
    _write(baseline, probabilities, baseline=True)

    base_manifest = tmp_path / "base_folds.csv"
    with base_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_index",
                "sample_id",
                "label",
                "sequence",
                "group_id",
                "fold",
            ],
        )
        writer.writeheader()
        for index in range(100):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": f"sample_{index}",
                    "label": index % 2,
                    "sequence": f"SEQ{index}",
                    "group_id": f"group_{index}",
                    "fold": index % 5,
                }
            )
    nested_csv = tmp_path / "nested.csv"
    create_nested_group_folds(
        base_manifest,
        nested_csv,
        tmp_path / "nested.json",
        inner_splits=4,
        seed=11,
    )
    summary = run_nested_calibrated_stacking(
        [first, second],
        baseline,
        nested_csv,
        tmp_path / "nested_stacking",
        member_names=["first", "second"],
        min_sn=None,
        min_sp=None,
        steps=80,
        restarts=1,
        seed=11,
    )
    assert summary["protocol"]["calibration_role"] == "calibration only"
    assert summary["protocol"]["evaluation_role"] == "outer test only"
    assert len(summary["fold_results"]) == 5
    assert sum(row["role_sizes"]["outer_test"] for row in summary["fold_results"]) == 100
