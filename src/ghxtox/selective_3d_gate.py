"""Nested, fallback-safe routing of a frozen 3D expert into a ProtT5 baseline."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ghxtox.calibrate_predictions import apply_calibration, fit_platt
from ghxtox.nested_folds import load_nested_indices
from ghxtox.utils import save_json, set_seed
from ghxtox.validated_ensemble import _crossfit_metrics, _threshold_search


FEATURE_NAMES = (
    "mean_plddt",
    "min_plddt",
    "one_minus_low_plddt_fraction",
    "global_3d_gate",
    "negative_expert_logit_disagreement",
    "negative_3d_mc_std",
    "negative_prott5_seed_std",
    "negative_structure_version_disagreement",
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    if "source_index" not in rows[0]:
        raise ValueError(f"Prediction file lacks source_index: {path}")
    rows.sort(key=lambda row: int(row["source_index"]))
    indices = [int(row["source_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError(f"source_index must cover 0..N-1 exactly: {path}")
    return rows


def _probability(rows: list[dict[str, str]], *columns: str) -> np.ndarray:
    for column in columns:
        if column in rows[0]:
            values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
            if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"Invalid probabilities in column {column!r}.")
            return values
    raise ValueError(f"None of the probability columns exists: {columns}")


def _logit_numpy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def _load_inputs(
    prott5_path: str | Path,
    structure_path: str | Path,
    default_path: str | Path,
    reference_structure_path: str | Path | None = None,
) -> dict[str, Any]:
    prott5_rows = _read_rows(prott5_path)
    structure_rows = _read_rows(structure_path)
    default_rows = _read_rows(default_path)
    if not (len(prott5_rows) == len(structure_rows) == len(default_rows)):
        raise ValueError("ProtT5, 3D and default OOF files have different row counts.")
    for index, (prott5, structure, default) in enumerate(
        zip(prott5_rows, structure_rows, default_rows)
    ):
        for key in ("source_index", "sequence", "label"):
            values = [row.get(key) for row in (prott5, structure, default)]
            if len(set(values)) != 1:
                raise ValueError(f"OOF alignment mismatch at row {index}, column {key}.")

    required_structure = {
        "mean_plddt",
        "min_plddt",
        "low_plddt_fraction",
        "global_gate",
        "mc_std_probability",
        "fold",
    }
    missing = required_structure.difference(structure_rows[0])
    if missing:
        raise ValueError(f"3D OOF diagnostics are missing: {sorted(missing)}")
    labels = np.asarray([int(row["label"]) for row in structure_rows], dtype=np.int64)
    folds = np.asarray([int(row["fold"]) for row in structure_rows], dtype=np.int64)
    prott5_probability = _probability(prott5_rows, "toxicity_probability")
    structure_probability = _probability(structure_rows, "toxicity_probability")
    default_probability = _probability(
        default_rows, "fusion_probability", "toxicity_probability"
    )
    if "prediction" not in default_rows[0]:
        raise ValueError("Current-default OOF file must include its frozen prediction column.")
    default_prediction = np.asarray(
        [int(row["prediction"]) != 0 for row in default_rows], dtype=bool
    )
    member_columns = sorted(
        column for column in prott5_rows[0] if column.startswith("member") and column.endswith("_probability")
    )
    if member_columns:
        member_matrix = np.asarray(
            [[float(row[column]) for column in member_columns] for row in prott5_rows],
            dtype=np.float64,
        )
        prott5_std = member_matrix.std(axis=1)
    else:
        prott5_std = np.zeros(len(labels), dtype=np.float64)
    if reference_structure_path is not None:
        reference_rows = _read_rows(reference_structure_path)
        if len(reference_rows) != len(structure_rows):
            raise ValueError("Reference-structure OOF file has a different row count.")
        for index, (reference, structure_row) in enumerate(
            zip(reference_rows, structure_rows)
        ):
            for key in ("source_index", "sequence", "label"):
                if reference.get(key) != structure_row.get(key):
                    raise ValueError(
                        f"Reference-structure alignment mismatch at row {index}, column {key}."
                    )
        reference_probability = _probability(reference_rows, "toxicity_probability")
        version_disagreement = np.abs(
            _logit_numpy(structure_probability) - _logit_numpy(reference_probability)
        )
    else:
        version_disagreement = np.zeros(len(labels), dtype=np.float64)
    features = np.column_stack(
        [
            np.asarray([float(row["mean_plddt"]) for row in structure_rows]),
            np.asarray([float(row["min_plddt"]) for row in structure_rows]),
            1.0 - np.asarray(
                [float(row["low_plddt_fraction"]) for row in structure_rows]
            ),
            np.asarray([float(row["global_gate"]) for row in structure_rows]),
            -np.abs(
                _logit_numpy(structure_probability) - _logit_numpy(prott5_probability)
            ),
            -np.asarray([float(row["mc_std_probability"]) for row in structure_rows]),
            -prott5_std,
            -version_disagreement,
        ]
    ).astype(np.float64, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Selective-gate features contain non-finite values.")
    return {
        "rows": structure_rows,
        "labels": labels,
        "folds": folds,
        "prott5_probability": prott5_probability,
        "structure_probability": structure_probability,
        "default_probability": default_probability,
        "default_prediction": default_prediction,
        "features": features,
        "member_columns": member_columns,
        "reference_structure_path": (
            None if reference_structure_path is None else str(reference_structure_path)
        ),
    }


def _fixed_quality_gate(features: np.ndarray, max_weight: float = 0.38) -> np.ndarray:
    """A label-free first-stage gate using only ESMFold quality summaries."""

    mean_plddt = features[:, 0]
    low_confidence_fraction = 1.0 - features[:, 2]
    quality = 1.0 / (1.0 + np.exp(-(mean_plddt - 0.70) / 0.12))
    quality *= np.clip(1.0 - 0.75 * low_confidence_fraction, 0.0, 1.0)
    return np.clip(float(max_weight) * quality, 0.0, float(max_weight))


def _mix(
    prott5_probability: np.ndarray,
    structure_probability: np.ndarray,
    gate: np.ndarray,
) -> np.ndarray:
    return np.clip(
        prott5_probability + gate * (structure_probability - prott5_probability),
        1e-6,
        1.0 - 1e-6,
    )


def _softplus_inverse(value: float) -> float:
    return float(np.log(np.expm1(value)))


def _apply_learned_gate(features: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(state["feature_mean"], dtype=np.float64)
    std = np.asarray(state["feature_std"], dtype=np.float64)
    normalized = (features - mean) / std
    weights = np.asarray(state["positive_weights"], dtype=np.float64)
    logits = float(state["bias"]) + normalized @ weights
    gate = float(state["max_weight"]) / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    return gate.astype(np.float64, copy=False)


def _fit_learned_gate(
    features: np.ndarray,
    prott5_probability: np.ndarray,
    structure_probability: np.ndarray,
    labels: np.ndarray,
    *,
    max_weight: float,
    gate_penalty: float,
    regularization: float,
    steps: int,
    restarts: int,
    seed: int,
) -> dict[str, Any]:
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0)
    active = feature_std >= 1e-6
    feature_std = np.where(active, feature_std, 1.0)
    normalized = torch.tensor(
        (features - feature_mean) / feature_std, dtype=torch.float32
    )
    base = torch.tensor(prott5_probability, dtype=torch.float32)
    structure = torch.tensor(structure_probability, dtype=torch.float32)
    target = torch.tensor(labels, dtype=torch.float32)
    active_tensor = torch.tensor(active, dtype=torch.float32)
    best: dict[str, Any] | None = None
    initial_raw = _softplus_inverse(0.05)
    for restart in range(max(int(restarts), 1)):
        set_seed(seed + restart)
        raw_weights = torch.nn.Parameter(
            torch.full((features.shape[1],), initial_raw)
            + (0.05 * torch.randn(features.shape[1]) if restart else 0.0)
        )
        bias = torch.nn.Parameter(torch.tensor(-1.5 + 0.2 * restart))
        optimizer = torch.optim.AdamW(
            [raw_weights, bias], lr=0.03, weight_decay=0.0
        )
        best_loss = float("inf")
        best_state: tuple[torch.Tensor, torch.Tensor] | None = None
        for _ in range(max(int(steps), 1)):
            optimizer.zero_grad()
            weights = F.softplus(raw_weights) * active_tensor
            gate = float(max_weight) * torch.sigmoid(bias + normalized @ weights)
            probability = base + gate * (structure - base)
            loss = F.binary_cross_entropy(probability.clamp(1e-6, 1.0 - 1e-6), target)
            loss = loss + float(gate_penalty) * gate.square().mean()
            loss = loss + float(regularization) * weights.square().mean()
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_state = (raw_weights.detach().clone(), bias.detach().clone())
        assert best_state is not None
        positive_weights = F.softplus(best_state[0]) * active_tensor
        candidate = {
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "active_features": active.tolist(),
            "positive_weights": positive_weights.tolist(),
            "bias": float(best_state[1]),
            "max_weight": float(max_weight),
            "gate_penalty": float(gate_penalty),
            "regularization": float(regularization),
            "training_regularized_bce": best_loss,
            "restart": restart,
        }
        if best is None or candidate["training_regularized_bce"] < best[
            "training_regularized_bce"
        ]:
            best = candidate
    assert best is not None
    return best


def _bce(probability: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return float(
        -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()
    )


def _fit_calibration_and_threshold(
    probability: np.ndarray,
    labels: np.ndarray,
    *,
    threshold_step: float,
    min_sn: float | None,
    min_sp: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    probability_tensor = torch.tensor(probability, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.float32)
    calibration = fit_platt(probability_tensor, label_tensor)
    calibrated = apply_calibration(probability_tensor, calibration).numpy()
    threshold = _threshold_search(
        labels,
        calibrated,
        metric="mcc",
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    return calibration, threshold


def _apply_numpy_calibration(
    probability: np.ndarray, calibration: dict[str, Any]
) -> np.ndarray:
    return apply_calibration(
        torch.tensor(probability, dtype=torch.float32), calibration
    ).numpy().astype(np.float64, copy=False)


def _summarize_gate(gate: np.ndarray, features: np.ndarray) -> dict[str, float]:
    high = features[:, 0] >= 0.70
    low = ~high
    return {
        "mean": float(gate.mean()),
        "std": float(gate.std()),
        "min": float(gate.min()),
        "max": float(gate.max()),
        "mean_high_plddt": float(gate[high].mean()) if high.any() else float("nan"),
        "mean_low_plddt": float(gate[low].mean()) if low.any() else float("nan"),
    }


def run_nested_selective_gate(
    prott5_path: str | Path,
    structure_path: str | Path,
    default_path: str | Path,
    nested_manifest: str | Path,
    output_dir: str | Path,
    *,
    reference_structure_path: str | Path | None = None,
    max_weights: tuple[float, ...] = (0.20, 0.30, 0.40),
    gate_penalties: tuple[float, ...] = (0.0, 0.002, 0.01),
    regularization: float = 0.01,
    steps: int = 500,
    restarts: int = 2,
    threshold_step: float = 0.005,
    min_sn: float | None = 0.875,
    min_sp: float | None = 0.96,
    seed: int = 2026,
) -> dict[str, Any]:
    data = _load_inputs(
        prott5_path,
        structure_path,
        default_path,
        reference_structure_path=reference_structure_path,
    )
    labels = data["labels"]
    features = data["features"]
    prott5 = data["prott5_probability"]
    structure = data["structure_probability"]
    current_default_probability = data["default_probability"]
    current_default_prediction = data["default_prediction"]
    outer_folds = sorted(np.unique(data["folds"]).tolist())

    methods = ("fixed_quality", "learned_selective")
    probabilities = {
        name: np.full(len(labels), np.nan, dtype=np.float64) for name in methods
    }
    predictions = {name: np.zeros(len(labels), dtype=bool) for name in methods}
    gate_values = {
        name: np.full(len(labels), np.nan, dtype=np.float64) for name in methods
    }
    nested_default_probability = np.full(len(labels), np.nan, dtype=np.float64)
    nested_default_prediction = np.zeros(len(labels), dtype=bool)
    nested_prott5_probability = np.full(len(labels), np.nan, dtype=np.float64)
    nested_prott5_prediction = np.zeros(len(labels), dtype=bool)
    fold_results = []
    selected_hyperparameters: list[tuple[float, float]] = []

    for outer_fold in outer_folds:
        roles = load_nested_indices(nested_manifest, outer_fold, len(labels))
        train = np.asarray(roles["train"], dtype=np.int64)
        validation = np.asarray(roles["validation"], dtype=np.int64)
        calibration = np.asarray(roles["calibration"], dtype=np.int64)
        test = np.asarray(roles["test"], dtype=np.int64)

        fixed_gate_calibration = _fixed_quality_gate(features[calibration])
        fixed_gate_test = _fixed_quality_gate(features[test])
        fixed_raw_calibration = _mix(
            prott5[calibration], structure[calibration], fixed_gate_calibration
        )
        fixed_calibrator, fixed_threshold = _fit_calibration_and_threshold(
            fixed_raw_calibration,
            labels[calibration],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        fixed_test = _apply_numpy_calibration(
            _mix(prott5[test], structure[test], fixed_gate_test), fixed_calibrator
        )
        probabilities["fixed_quality"][test] = fixed_test
        predictions["fixed_quality"][test] = fixed_test >= fixed_threshold["threshold"]
        gate_values["fixed_quality"][test] = fixed_gate_test

        candidate_states = []
        for max_weight in max_weights:
            for gate_penalty in gate_penalties:
                state = _fit_learned_gate(
                    features[train],
                    prott5[train],
                    structure[train],
                    labels[train],
                    max_weight=max_weight,
                    gate_penalty=gate_penalty,
                    regularization=regularization,
                    steps=steps,
                    restarts=restarts,
                    seed=seed + outer_fold * 100,
                )
                validation_gate = _apply_learned_gate(features[validation], state)
                validation_probability = _mix(
                    prott5[validation], structure[validation], validation_gate
                )
                candidate_states.append(
                    (
                        _bce(validation_probability, labels[validation]),
                        -float(validation_gate.std()),
                        state,
                    )
                )
        _, _, selected = min(candidate_states, key=lambda item: (item[0], item[1]))
        selected_pair = (
            float(selected["max_weight"]),
            float(selected["gate_penalty"]),
        )
        selected_hyperparameters.append(selected_pair)
        fit = np.asarray(sorted(roles["train"] + roles["validation"]), dtype=np.int64)
        refitted = _fit_learned_gate(
            features[fit],
            prott5[fit],
            structure[fit],
            labels[fit],
            max_weight=selected_pair[0],
            gate_penalty=selected_pair[1],
            regularization=regularization,
            steps=steps,
            restarts=restarts,
            seed=seed + outer_fold * 100 + 50,
        )
        learned_gate_calibration = _apply_learned_gate(features[calibration], refitted)
        learned_gate_test = _apply_learned_gate(features[test], refitted)
        learned_raw_calibration = _mix(
            prott5[calibration], structure[calibration], learned_gate_calibration
        )
        learned_calibrator, learned_threshold = _fit_calibration_and_threshold(
            learned_raw_calibration,
            labels[calibration],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        learned_test = _apply_numpy_calibration(
            _mix(prott5[test], structure[test], learned_gate_test), learned_calibrator
        )
        probabilities["learned_selective"][test] = learned_test
        predictions["learned_selective"][test] = learned_test >= learned_threshold["threshold"]
        gate_values["learned_selective"][test] = learned_gate_test

        default_calibrator, default_threshold = _fit_calibration_and_threshold(
            current_default_probability[calibration],
            labels[calibration],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        calibrated_default_test = _apply_numpy_calibration(
            current_default_probability[test], default_calibrator
        )
        nested_default_probability[test] = calibrated_default_test
        nested_default_prediction[test] = (
            calibrated_default_test >= default_threshold["threshold"]
        )

        prott5_calibrator, prott5_threshold = _fit_calibration_and_threshold(
            prott5[calibration],
            labels[calibration],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        calibrated_prott5_test = _apply_numpy_calibration(prott5[test], prott5_calibrator)
        nested_prott5_probability[test] = calibrated_prott5_test
        nested_prott5_prediction[test] = (
            calibrated_prott5_test >= prott5_threshold["threshold"]
        )

        fold_results.append(
            {
                "outer_fold": int(outer_fold),
                "role_sizes": {
                    "train": int(len(train)),
                    "validation": int(len(validation)),
                    "calibration": int(len(calibration)),
                    "outer_test": int(len(test)),
                },
                "selected_max_weight": selected_pair[0],
                "selected_gate_penalty": selected_pair[1],
                "fixed_threshold": fixed_threshold["threshold"],
                "learned_threshold": learned_threshold["threshold"],
                "current_default_threshold": default_threshold["threshold"],
                "prott5_threshold": prott5_threshold["threshold"],
                "fixed_quality": _crossfit_metrics(
                    labels[test], fixed_test, predictions["fixed_quality"][test]
                ),
                "learned_selective": _crossfit_metrics(
                    labels[test], learned_test, predictions["learned_selective"][test]
                ),
                "current_default": _crossfit_metrics(
                    labels[test],
                    current_default_probability[test],
                    current_default_prediction[test],
                ),
                "learned_gate": _summarize_gate(learned_gate_test, features[test]),
            }
        )

    for name in methods:
        if np.isnan(probabilities[name]).any() or np.isnan(gate_values[name]).any():
            raise RuntimeError(f"Nested outer folds did not cover every row for {name}.")
    current_default = _crossfit_metrics(
        labels, current_default_probability, current_default_prediction
    )
    nested_default = _crossfit_metrics(
        labels, nested_default_probability, nested_default_prediction
    )
    nested_prott5 = _crossfit_metrics(
        labels, nested_prott5_probability, nested_prott5_prediction
    )
    method_metrics = {
        name: _crossfit_metrics(labels, probabilities[name], predictions[name])
        for name in methods
    }
    learned = method_metrics["learned_selective"]
    deltas = {
        "mcc": learned["mcc"] - current_default["mcc"],
        "balanced_accuracy": learned["balanced_accuracy"]
        - current_default["balanced_accuracy"],
        "auroc": learned["auroc"] - current_default["auroc"],
        "auprc": learned["auprc"] - current_default["auprc"],
        "brier_lower_is_better": current_default["brier"] - learned["brier"],
        "ece_10_lower_is_better": current_default["ece_10"] - learned["ece_10"],
    }
    fold_wins = sum(
        row["learned_selective"]["mcc"] > row["current_default"]["mcc"]
        for row in fold_results
    )
    gates = {
        "mcc_strictly_improves_current_default": learned["mcc"] > current_default["mcc"],
        "balanced_accuracy_not_lower": learned["balanced_accuracy"]
        >= current_default["balanced_accuracy"],
        "auroc_within_0.0005": learned["auroc"] >= current_default["auroc"] - 0.0005,
        "auprc_not_lower": learned["auprc"] >= current_default["auprc"],
        "brier_within_0.0005": learned["brier"] <= current_default["brier"] + 0.0005,
        "ece_within_0.002": learned["ece_10"] <= current_default["ece_10"] + 0.002,
        "mcc_improves_in_at_least_three_folds": fold_wins >= 3,
    }
    replace_default = all(gates.values())

    selected_mode = Counter(selected_hyperparameters).most_common(1)[0][0]
    frozen_gate = _fit_learned_gate(
        features,
        prott5,
        structure,
        labels,
        max_weight=selected_mode[0],
        gate_penalty=selected_mode[1],
        regularization=regularization,
        steps=steps,
        restarts=restarts,
        seed=seed + 999,
    )
    frozen_values = _apply_learned_gate(features, frozen_gate)
    frozen_raw = _mix(prott5, structure, frozen_values)
    frozen_calibrator, frozen_threshold = _fit_calibration_and_threshold(
        frozen_raw,
        labels,
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    summary = {
        "protocol": {
            "selection_data": "frozen aligned OOF predictions only",
            "nested_manifest": str(nested_manifest),
            "gate_fit_role": "train",
            "hyperparameter_selection_role": "validation",
            "refit_roles": ["train", "validation"],
            "calibration_and_threshold_role": "calibration",
            "evaluation_role": "outer test",
            "feature_names": list(FEATURE_NAMES),
            "max_weight_candidates": list(max_weights),
            "gate_penalty_candidates": list(gate_penalties),
            "regularization": regularization,
            "test1_or_test2_predictions_read": False,
        },
        "diagnostics": {
            "num_samples": int(len(labels)),
            "member_columns_for_prott5_variance": data["member_columns"],
            "reference_structure_path": data["reference_structure_path"],
            "raw_global_gate": {
                "mean": float(features[:, 3].mean()),
                "std": float(features[:, 3].std()),
                "min": float(features[:, 3].min()),
                "max": float(features[:, 3].max()),
            },
        },
        "current_default": current_default,
        "nested_recalibrated_current_default": nested_default,
        "nested_recalibrated_prott5": nested_prott5,
        "fixed_quality_gate": method_metrics["fixed_quality"],
        "learned_selective_gate": learned,
        "delta_learned_minus_current_default": deltas,
        "gate_distributions": {
            name: _summarize_gate(gate_values[name], features) for name in methods
        },
        "fold_results": fold_results,
        "fold_mcc_wins": int(fold_wins),
        "selected_hyperparameter_counts": {
            f"max_weight={key[0]:.3g},gate_penalty={key[1]:.3g}": count
            for key, count in Counter(selected_hyperparameters).items()
        },
        "retention_gates": gates,
        "decision": {
            "replace_default": replace_default,
            "run_historical_tests": False,
            "reason": (
                "All predeclared nested outer-test gates passed."
                if replace_default
                else "At least one nested outer-test gate against the current default failed."
            ),
        },
        "frozen_all_oof_candidate": {
            "gate": frozen_gate,
            "gate_distribution": _summarize_gate(frozen_values, features),
            "platt": frozen_calibrator,
            "threshold": frozen_threshold,
            "warning": "Fit on all OOF rows for a future untouched external set; nested outer-test metrics control retention.",
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    with (output_dir / "nested_outer_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_index",
                "fold",
                "sample_id",
                "sequence",
                "label",
                "prott5_probability",
                "structure_probability",
                "current_default_probability",
                "current_default_prediction",
                "fixed_quality_gate",
                "fixed_quality_probability",
                "fixed_quality_prediction",
                "learned_3d_gate",
                "learned_probability",
                "learned_prediction",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(data["rows"]):
            writer.writerow(
                {
                    "source_index": index,
                    "fold": int(data["folds"][index]),
                    "sample_id": row["sample_id"],
                    "sequence": row["sequence"],
                    "label": int(labels[index]),
                    "prott5_probability": f"{prott5[index]:.10g}",
                    "structure_probability": f"{structure[index]:.10g}",
                    "current_default_probability": f"{current_default_probability[index]:.10g}",
                    "current_default_prediction": int(current_default_prediction[index]),
                    "fixed_quality_gate": f"{gate_values['fixed_quality'][index]:.10g}",
                    "fixed_quality_probability": f"{probabilities['fixed_quality'][index]:.10g}",
                    "fixed_quality_prediction": int(predictions["fixed_quality"][index]),
                    "learned_3d_gate": f"{gate_values['learned_selective'][index]:.10g}",
                    "learned_probability": f"{probabilities['learned_selective'][index]:.10g}",
                    "learned_prediction": int(predictions["learned_selective"][index]),
                }
            )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prott5", required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--current-default", required=True)
    parser.add_argument("--reference-structure")
    parser.add_argument("--nested-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_nested_selective_gate(
        args.prott5,
        args.structure,
        args.current_default,
        args.nested_manifest,
        args.output_dir,
        reference_structure_path=args.reference_structure,
        steps=args.steps,
        restarts=args.restarts,
        seed=args.seed,
    )
    print(json.dumps(result["learned_selective_gate"], indent=2))
    print(json.dumps(result["retention_gates"], indent=2))
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
