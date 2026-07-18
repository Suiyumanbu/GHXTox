from __future__ import annotations

import csv

from ghxtox.oof_fusion import apply_frozen_fusion, screen_oof_fusion


def _write_predictions(path, probabilities, *, with_folds, sample_prefix="sample"):
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    fieldnames = ["sample_id", "sequence", "label", "toxicity_probability"]
    if with_folds:
        fieldnames.insert(0, "fold")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (label, probability) in enumerate(zip(labels, probabilities)):
            row = {
                "sample_id": f"{sample_prefix}_{index}",
                "sequence": "AC" + "D" * index,
                "label": label,
                "toxicity_probability": probability,
            }
            if with_folds:
                row["fold"] = index % 5
            writer.writerow(row)


def test_screen_and_apply_oof_fusion(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    _write_predictions(
        baseline,
        [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9],
        with_folds=True,
    )
    _write_predictions(
        expert,
        [0.2, 0.9, 0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.7],
        with_folds=False,
    )
    screened = screen_oof_fusion(
        baseline, expert, tmp_path / "screen", weight_step=0.25, threshold_step=0.1
    )
    assert screened["protocol"]["num_folds"] == 5
    assert 0.0 <= screened["frozen_parameters_from_all_oof"]["expert_weight"] <= 1.0
    applied = apply_frozen_fusion(
        baseline,
        expert,
        tmp_path / "applied",
        expert_weight=screened["frozen_parameters_from_all_oof"]["expert_weight"],
        threshold=screened["frozen_parameters_from_all_oof"]["threshold"],
    )
    assert applied["protocol"]["parameters_selected_from_test_labels"] is False
    assert (tmp_path / "applied" / "predictions.csv").exists()


def test_screen_accepts_verified_id_aliases_and_external_fold_manifest(tmp_path) -> None:
    baseline = tmp_path / "baseline.csv"
    expert = tmp_path / "expert.csv"
    manifest = tmp_path / "folds.csv"
    probabilities = [0.1, 0.7, 0.3, 0.6, 0.2, 0.8, 0.4, 0.55, 0.45, 0.9]
    _write_predictions(
        baseline, probabilities, with_folds=False, sample_prefix="peptide"
    )
    _write_predictions(
        expert, probabilities, with_folds=False, sample_prefix="train"
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_index", "sample_id", "sequence", "label", "fold"]
        )
        writer.writeheader()
        for index, label in enumerate([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": f"manifest_{index}",
                    "sequence": "AC" + "D" * index,
                    "label": label,
                    "fold": index % 5,
                }
            )
    screened = screen_oof_fusion(
        baseline,
        expert,
        tmp_path / "screen_alias",
        weight_step=0.25,
        threshold_step=0.1,
        fold_manifest=manifest,
    )
    assert screened["protocol"]["num_folds"] == 5
    assert screened["protocol"]["fold_manifest"] == str(manifest)
