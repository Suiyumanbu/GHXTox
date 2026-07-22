from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from ghxtox.calibrate_predictions import (
    _read_predictions,
    apply_calibration,
    calibrate_predictions,
    fit_platt,
)
from ghxtox.metrics import binary_metrics, expected_calibration_error
from ghxtox.validated_ensemble import (
    apply_frozen_ensemble,
    screen_validated_ensemble,
)


def _write_predictions(path, probabilities, *, duplicate_id=False) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_index",
                "fold",
                "sample_id",
                "sequence",
                "label",
                "toxicity_probability",
            ],
        )
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            writer.writerow(
                {
                    "source_index": index,
                    "fold": index % 5,
                    "sample_id": "duplicate" if duplicate_id else f"sample_{index}",
                    "sequence": f"ACD{index}",
                    "label": index % 2,
                    "toxicity_probability": probability,
                }
            )


def test_binary_metrics_include_probability_reliability() -> None:
    probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    metrics = binary_metrics(torch.logit(probabilities), labels)
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["ece_10"] == pytest.approx(0.15)
    assert metrics["sn"] == metrics["recall"] == 1.0
    assert metrics["sp"] == metrics["specificity"] == 1.0
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.15)


def test_monotonic_platt_preserves_order_and_calibration_pipeline(tmp_path) -> None:
    probabilities = np.asarray(
        [0.36 + 0.002 * index if index % 2 == 0 else 0.56 + 0.002 * index for index in range(40)]
    )
    calibration_path = tmp_path / "calibration.csv"
    _write_predictions(calibration_path, probabilities)
    _, labels, raw = _read_predictions(calibration_path)
    fitted = fit_platt(raw, labels)
    calibrated = apply_calibration(raw, fitted)
    assert float(fitted["scale"]) > 0.0
    assert torch.equal(torch.argsort(raw), torch.argsort(calibrated))

    summary = calibrate_predictions(
        calibration_path,
        [calibration_path],
        tmp_path / "calibrated",
        tmp_path / "summary.json",
        method="platt",
        threshold_metric="mcc",
    )
    assert summary["protocol"]["required_role"].startswith("dedicated")
    assert summary["calibration_at_0.5_after"]["brier"] <= summary["calibration_at_0.5_before"]["brier"]
    assert abs(summary["applied"][0]["ranking_metric_delta"]["auroc"]) <= 1e-5
    assert (tmp_path / "calibrated" / "calibration_calibrated.csv").exists()


def test_calibration_rejects_duplicate_sample_ids(tmp_path) -> None:
    path = tmp_path / "duplicates.csv"
    _write_predictions(path, [0.1, 0.9], duplicate_id=True)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _read_predictions(path)


def test_validated_ensemble_crossfits_and_freezes_without_test_input(tmp_path) -> None:
    labels = np.arange(40) % 2
    first = []
    second = []
    inverted = []
    for index, label in enumerate(labels):
        first_half = index < 20
        if first_half:
            first.append(0.9 if label else 0.1)
            second.append(0.4 if label else 0.6)
        else:
            first.append(0.4 if label else 0.6)
            second.append(0.9 if label else 0.1)
        inverted.append(0.1 if label else 0.9)
    paths = []
    for name, probabilities in (
        ("first", first),
        ("second", second),
        ("inverted", inverted),
    ):
        path = tmp_path / f"{name}.csv"
        _write_predictions(path, probabilities)
        paths.append(path)

    summary = screen_validated_ensemble(
        paths,
        tmp_path / "screen",
        member_names=["first", "second", "inverted"],
        objective="mcc",
        max_members=3,
    )
    assert summary["protocol"]["screen_interface_has_no_separate_test_argument"] is True
    assert summary["crossfit_metrics"]["accuracy"] == pytest.approx(1.0)
    assert summary["frozen_spec"]["test_labels_used_for_selection"] is False
    assert "inverted" not in summary["frozen_spec"]["selected_member_names"]

    spec = tmp_path / "screen" / "frozen_spec.json"
    applied = apply_frozen_ensemble(spec, paths, tmp_path / "applied.csv")
    assert applied["protocol"]["parameters_selected_from_applied_labels"] is False
    assert applied["metrics"]["accuracy"] == pytest.approx(1.0)
    assert json.loads(spec.read_text(encoding="utf-8"))["threshold"] > 0.0
