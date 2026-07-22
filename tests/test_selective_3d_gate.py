from __future__ import annotations

import csv

import numpy as np

from ghxtox.nested_folds import create_nested_group_folds
from ghxtox.selective_3d_gate import (
    _apply_learned_gate,
    run_nested_selective_gate,
)


def _write_inputs(tmp_path, count: int = 100):
    prott5_path = tmp_path / "prott5.csv"
    structure_path = tmp_path / "structure.csv"
    default_path = tmp_path / "default.csv"
    common = ["source_index", "sample_id", "sequence", "label"]
    with (
        prott5_path.open("w", encoding="utf-8", newline="") as prott5_handle,
        structure_path.open("w", encoding="utf-8", newline="") as structure_handle,
        default_path.open("w", encoding="utf-8", newline="") as default_handle,
    ):
        prott5_writer = csv.DictWriter(
            prott5_handle,
            fieldnames=common
            + [
                "toxicity_probability",
                "member0_probability",
                "member1_probability",
            ],
        )
        structure_writer = csv.DictWriter(
            structure_handle,
            fieldnames=common
            + [
                "fold",
                "toxicity_probability",
                "mean_plddt",
                "min_plddt",
                "low_plddt_fraction",
                "global_gate",
                "mc_std_probability",
            ],
        )
        default_writer = csv.DictWriter(
            default_handle,
            fieldnames=common + ["fusion_probability", "prediction"],
        )
        for writer in (prott5_writer, structure_writer, default_writer):
            writer.writeheader()
        for index in range(count):
            label = index % 2
            high_quality = (index // 2) % 2 == 0
            prott5 = 0.78 if label else 0.22
            if high_quality:
                structure = 0.92 if label else 0.08
                mean_plddt, min_plddt, low_fraction = 0.88, 0.78, 0.02
            else:
                structure = 0.38 if label else 0.62
                mean_plddt, min_plddt, low_fraction = 0.52, 0.35, 0.55
            default = 0.62 * prott5 + 0.38 * structure
            base = {
                "source_index": index,
                "sample_id": f"sample_{index}",
                "sequence": f"SEQ{index}",
                "label": label,
            }
            prott5_writer.writerow(
                {
                    **base,
                    "toxicity_probability": prott5,
                    "member0_probability": prott5 - 0.01,
                    "member1_probability": prott5 + 0.01,
                }
            )
            structure_writer.writerow(
                {
                    **base,
                    "fold": index % 5,
                    "toxicity_probability": structure,
                    "mean_plddt": mean_plddt,
                    "min_plddt": min_plddt,
                    "low_plddt_fraction": low_fraction,
                    "global_gate": 0.40 if high_quality else 0.32,
                    "mc_std_probability": 0.01 if high_quality else 0.08,
                }
            )
            default_writer.writerow(
                {
                    **base,
                    "fusion_probability": default,
                    "prediction": int(default >= 0.5),
                }
            )
    return prott5_path, structure_path, default_path


def _write_manifest(tmp_path, count: int = 100):
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
        for index in range(count):
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
    nested = tmp_path / "nested.csv"
    create_nested_group_folds(
        base_manifest, nested, tmp_path / "nested.json", inner_splits=4, seed=11
    )
    return nested


def test_learned_gate_is_bounded_and_monotonic_in_safety_features() -> None:
    state = {
        "feature_mean": [0.0, 0.0],
        "feature_std": [1.0, 1.0],
        "positive_weights": [1.0, 0.5],
        "bias": -1.0,
        "max_weight": 0.4,
    }
    gate = _apply_learned_gate(np.asarray([[0.0, 0.0], [1.0, 1.0]]), state)
    assert 0.0 <= gate[0] < gate[1] <= 0.4


def test_nested_selective_gate_keeps_roles_disjoint_and_writes_summary(tmp_path) -> None:
    prott5, structure, default = _write_inputs(tmp_path)
    nested = _write_manifest(tmp_path)
    output = tmp_path / "selective_gate"
    summary = run_nested_selective_gate(
        prott5,
        structure,
        default,
        nested,
        output,
        max_weights=(0.3,),
        gate_penalties=(0.0,),
        steps=40,
        restarts=1,
        min_sn=None,
        min_sp=None,
        seed=11,
    )
    assert summary["protocol"]["gate_fit_role"] == "train"
    assert summary["protocol"]["evaluation_role"] == "outer test"
    assert summary["protocol"]["test1_or_test2_predictions_read"] is False
    assert len(summary["fold_results"]) == 5
    assert sum(row["role_sizes"]["outer_test"] for row in summary["fold_results"]) == 100
    assert summary["gate_distributions"]["learned_selective"]["max"] <= 0.3
    assert (output / "summary.json").exists()
    assert (output / "nested_outer_predictions.csv").exists()
