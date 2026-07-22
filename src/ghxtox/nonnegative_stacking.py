"""Cross-fit a non-negative probability stacker on aligned OOF predictions only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ghxtox.calibrate_predictions import apply_calibration, fit_platt
from ghxtox.nested_folds import load_nested_indices
from ghxtox.utils import save_json, set_seed
from ghxtox.validated_ensemble import (
    _aligned_matrix,
    _crossfit_metrics,
    _folds_from_rows,
    _threshold_search,
)


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))


def _apply_stacker(
    matrix: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    probabilities = torch.tensor(matrix.T, dtype=torch.float32)
    weights = torch.tensor(parameters["weights"], dtype=torch.float32)
    mixture = probabilities @ weights
    calibrated = torch.sigmoid(
        float(parameters["scale"]) * _logit(mixture)
        + float(parameters["bias"])
    )
    return calibrated.detach().cpu().numpy().astype(np.float64, copy=False)


def _fit_stacker(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    regularization: float,
    restarts: int,
    seed: int,
) -> dict[str, Any]:
    probabilities = torch.tensor(matrix.T, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    num_members = int(matrix.shape[0])
    best: dict[str, Any] | None = None
    for restart in range(max(int(restarts), 1)):
        set_seed(seed + restart)
        raw_weights = torch.nn.Parameter(
            torch.zeros(num_members)
            if restart == 0
            else 0.05 * torch.randn(num_members)
        )
        log_scale = torch.nn.Parameter(torch.zeros(()))
        bias = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.Adam(
            [raw_weights, log_scale, bias], lr=float(learning_rate)
        )
        best_restart_loss = float("inf")
        best_restart_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for _ in range(max(int(steps), 1)):
            optimizer.zero_grad()
            weights = torch.softmax(raw_weights, dim=0)
            mixture = probabilities @ weights
            scale = log_scale.exp().clamp(0.1, 10.0)
            logits = scale * _logit(mixture) + bias
            centered_weights = raw_weights - raw_weights.mean()
            penalty = (
                centered_weights.square().mean()
                + 0.25 * log_scale.square()
                + 0.05 * bias.square()
            )
            loss = F.binary_cross_entropy_with_logits(logits, targets) + float(
                regularization
            ) * penalty
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_restart_loss:
                best_restart_loss = value
                best_restart_state = (
                    raw_weights.detach().clone(),
                    log_scale.detach().clone(),
                    bias.detach().clone(),
                )
        assert best_restart_state is not None
        fitted_weights = torch.softmax(best_restart_state[0], dim=0)
        candidate = {
            "weights": fitted_weights.tolist(),
            "scale": float(best_restart_state[1].exp().clamp(0.1, 10.0)),
            "bias": float(best_restart_state[2]),
            "training_regularized_bce": best_restart_loss,
            "restart": restart,
        }
        if best is None or candidate["training_regularized_bce"] < best[
            "training_regularized_bce"
        ]:
            best = candidate
    assert best is not None
    return best


def _fit_weights_only(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    steps: int,
    learning_rate: float,
    regularization: float,
    restarts: int,
    seed: int,
) -> dict[str, Any]:
    probabilities = torch.tensor(matrix.T, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.float32)
    num_members = int(matrix.shape[0])
    best: dict[str, Any] | None = None
    for restart in range(max(int(restarts), 1)):
        set_seed(seed + restart)
        raw_weights = torch.nn.Parameter(
            torch.zeros(num_members)
            if restart == 0
            else 0.05 * torch.randn(num_members)
        )
        optimizer = torch.optim.Adam([raw_weights], lr=float(learning_rate))
        best_loss = float("inf")
        best_raw: torch.Tensor | None = None
        for _ in range(max(int(steps), 1)):
            optimizer.zero_grad()
            weights = torch.softmax(raw_weights, dim=0)
            mixture = (probabilities @ weights).clamp(1e-6, 1.0 - 1e-6)
            centered = raw_weights - raw_weights.mean()
            loss = F.binary_cross_entropy(mixture, targets) + float(
                regularization
            ) * centered.square().mean()
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best_loss:
                best_loss = value
                best_raw = raw_weights.detach().clone()
        assert best_raw is not None
        candidate = {
            "weights": torch.softmax(best_raw, dim=0).tolist(),
            "training_regularized_bce": best_loss,
            "restart": restart,
        }
        if best is None or candidate["training_regularized_bce"] < best[
            "training_regularized_bce"
        ]:
            best = candidate
    assert best is not None
    return best


def _weighted_probability(matrix: np.ndarray, weights: list[float]) -> np.ndarray:
    return np.average(
        matrix,
        axis=0,
        weights=np.asarray(weights, dtype=np.float64),
    ).astype(np.float64, copy=False)


def _calibrate_and_select_threshold(
    fit_probabilities: np.ndarray,
    fit_labels: np.ndarray,
    *,
    threshold_step: float,
    min_sn: float | None,
    min_sp: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    probability_tensor = torch.tensor(fit_probabilities, dtype=torch.float32)
    label_tensor = torch.tensor(fit_labels, dtype=torch.float32)
    calibration = fit_platt(probability_tensor, label_tensor)
    calibrated = apply_calibration(probability_tensor, calibration).numpy()
    threshold = _threshold_search(
        fit_labels,
        calibrated,
        metric="mcc",
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    return calibration, threshold


def _apply_numpy_calibration(
    probabilities: np.ndarray, calibration: dict[str, Any]
) -> np.ndarray:
    return apply_calibration(
        torch.tensor(probabilities, dtype=torch.float32), calibration
    ).numpy().astype(np.float64, copy=False)


def _read_baseline(
    path: str | Path,
    reference_rows: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(reference_rows):
        raise ValueError("Baseline and stacker OOF predictions have different lengths.")
    probability_column = (
        "fusion_probability"
        if rows and "fusion_probability" in rows[0]
        else "toxicity_probability"
    )
    for index, (baseline, reference) in enumerate(zip(rows, reference_rows)):
        for key in ("source_index", "sequence", "label"):
            if key in baseline and key in reference and baseline[key] != reference[key]:
                raise ValueError(f"Baseline alignment mismatch at row {index}, column {key}.")
    probabilities = np.asarray(
        [float(row[probability_column]) for row in rows], dtype=np.float64
    )
    if rows and "prediction" in rows[0]:
        predictions = np.asarray([int(row["prediction"]) != 0 for row in rows], dtype=bool)
    else:
        predictions = probabilities >= 0.5
    return probabilities, predictions


def run_nonnegative_stacking(
    prediction_paths: list[str | Path],
    baseline_path: str | Path,
    output_dir: str | Path,
    *,
    member_names: list[str] | None = None,
    fold_manifest: str | Path | None = None,
    threshold_step: float = 0.005,
    min_sn: float | None = 0.875,
    min_sp: float | None = 0.96,
    steps: int = 500,
    learning_rate: float = 0.03,
    regularization: float = 0.001,
    restarts: int = 3,
    seed: int = 2026,
) -> dict[str, Any]:
    rows, labels, matrix = _aligned_matrix(prediction_paths)
    names = member_names or [Path(path).stem for path in prediction_paths]
    if len(names) != len(prediction_paths) or len(names) != len(set(names)):
        raise ValueError("Member names must be unique and match the prediction file count.")
    folds = _folds_from_rows(rows, fold_manifest)
    baseline_probabilities, baseline_predictions = _read_baseline(baseline_path, rows)
    crossfit_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    crossfit_predictions = np.zeros(len(labels), dtype=bool)
    fold_results = []
    fold_wins = 0
    for heldout_fold in sorted(np.unique(folds).tolist()):
        training = folds != heldout_fold
        heldout = folds == heldout_fold
        parameters = _fit_stacker(
            matrix[:, training],
            labels[training],
            steps=steps,
            learning_rate=learning_rate,
            regularization=regularization,
            restarts=restarts,
            seed=seed + int(heldout_fold) * 100,
        )
        training_probabilities = _apply_stacker(matrix[:, training], parameters)
        threshold_result = _threshold_search(
            labels[training],
            training_probabilities,
            metric="mcc",
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        heldout_probabilities = _apply_stacker(matrix[:, heldout], parameters)
        heldout_predictions = heldout_probabilities >= threshold_result["threshold"]
        crossfit_probabilities[heldout] = heldout_probabilities
        crossfit_predictions[heldout] = heldout_predictions
        candidate_metrics = _crossfit_metrics(
            labels[heldout], heldout_probabilities, heldout_predictions
        )
        baseline_metrics = _crossfit_metrics(
            labels[heldout],
            baseline_probabilities[heldout],
            baseline_predictions[heldout],
        )
        if candidate_metrics["mcc"] > baseline_metrics["mcc"]:
            fold_wins += 1
        fold_results.append(
            {
                "heldout_fold": int(heldout_fold),
                "weights": {
                    name: float(weight)
                    for name, weight in zip(names, parameters["weights"])
                },
                "scale": parameters["scale"],
                "bias": parameters["bias"],
                "threshold": threshold_result["threshold"],
                "constraints_satisfied": threshold_result["constraints_satisfied"],
                "candidate_metrics": candidate_metrics,
                "baseline_metrics": baseline_metrics,
                "mcc_gain": candidate_metrics["mcc"] - baseline_metrics["mcc"],
            }
        )
    if np.isnan(crossfit_probabilities).any():
        raise RuntimeError("Stacking did not produce a prediction for every OOF row.")
    candidate = _crossfit_metrics(labels, crossfit_probabilities, crossfit_predictions)
    baseline = _crossfit_metrics(labels, baseline_probabilities, baseline_predictions)
    frozen_parameters = _fit_stacker(
        matrix,
        labels,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
        restarts=restarts,
        seed=seed + 999,
    )
    frozen_probabilities = _apply_stacker(matrix, frozen_parameters)
    frozen_threshold = _threshold_search(
        labels,
        frozen_probabilities,
        metric="mcc",
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    deltas = {
        "mcc": candidate["mcc"] - baseline["mcc"],
        "auroc": candidate["auroc"] - baseline["auroc"],
        "auprc": candidate["auprc"] - baseline["auprc"],
        "brier_lower_is_better": baseline["brier"] - candidate["brier"],
        "ece_10_lower_is_better": baseline["ece_10"] - candidate["ece_10"],
    }
    gates = {
        "mcc_strictly_improves": candidate["mcc"] > baseline["mcc"],
        "auroc_within_0.0005": candidate["auroc"] >= baseline["auroc"] - 0.0005,
        "auprc_not_lower": candidate["auprc"] >= baseline["auprc"],
        "brier_within_0.0005": candidate["brier"] <= baseline["brier"] + 0.0005,
        "ece_within_0.002": candidate["ece_10"] <= baseline["ece_10"] + 0.002,
        "mcc_improves_in_at_least_three_folds": fold_wins >= 3,
    }
    retain = all(gates.values())
    weight_matrix = np.asarray(
        [[row["weights"][name] for name in names] for row in fold_results],
        dtype=np.float64,
    )
    summary = {
        "protocol": {
            "selection_data": "aligned group-aware OOF predictions only",
            "prediction_paths": [str(path) for path in prediction_paths],
            "baseline_path": str(baseline_path),
            "member_names": names,
            "fold_manifest": None if fold_manifest is None else str(fold_manifest),
            "model": "softmax non-negative probability weights followed by monotonic affine-logit scaling",
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "regularization": float(regularization),
            "restarts": int(restarts),
            "threshold_step": float(threshold_step),
            "constraints": {"min_sn": min_sn, "min_sp": min_sp},
            "test1_or_test2_predictions_read": False,
        },
        "baseline_crossfit": baseline,
        "stacking_crossfit": candidate,
        "delta_stacking_minus_baseline": deltas,
        "fold_results": fold_results,
        "fold_mcc_wins": fold_wins,
        "weight_stability": {
            name: {
                "mean": float(weight_matrix[:, index].mean()),
                "std": float(weight_matrix[:, index].std()),
            }
            for index, name in enumerate(names)
        },
        "frozen_all_oof_candidate": {
            "parameters": frozen_parameters,
            "threshold": frozen_threshold,
            "warning": "Optimistic all-OOF fit; the cross-fit result controls retention.",
        },
        "retention_gates": gates,
        "decision": {
            "retain_stacking": retain,
            "run_historical_tests": False,
            "reason": (
                "All predeclared cross-fit gates passed."
                if retain
                else "At least one predeclared cross-fit gate failed."
            ),
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    with (output_dir / "crossfit_predictions.csv").open(
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
                "toxicity_probability",
                "prediction",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "fold": int(folds[index]),
                    "sample_id": row["sample_id"],
                    "sequence": row.get("sequence", ""),
                    "label": int(labels[index]),
                    "toxicity_probability": f"{crossfit_probabilities[index]:.10g}",
                    "prediction": int(crossfit_predictions[index]),
                }
            )
    return summary


def run_nested_calibrated_stacking(
    prediction_paths: list[str | Path],
    baseline_path: str | Path,
    nested_manifest: str | Path,
    output_dir: str | Path,
    *,
    member_names: list[str] | None = None,
    threshold_step: float = 0.005,
    min_sn: float | None = 0.875,
    min_sp: float | None = 0.96,
    steps: int = 500,
    learning_rate: float = 0.03,
    regularization: float = 0.001,
    restarts: int = 3,
    seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate weights, Platt scaling and threshold on disjoint nested roles."""

    rows, labels, matrix = _aligned_matrix(prediction_paths)
    names = member_names or [Path(path).stem for path in prediction_paths]
    if len(names) != len(prediction_paths) or len(names) != len(set(names)):
        raise ValueError("Member names must be unique and match the prediction file count.")
    baseline_probabilities, current_default_predictions = _read_baseline(
        baseline_path, rows
    )
    outer_folds = sorted({int(row["fold"]) for row in rows})
    candidate_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    candidate_predictions = np.zeros(len(labels), dtype=bool)
    nested_baseline_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    nested_baseline_predictions = np.zeros(len(labels), dtype=bool)
    fold_results = []
    fold_wins = 0
    weight_rows = []
    for outer_fold in outer_folds:
        roles = load_nested_indices(nested_manifest, outer_fold, len(labels))
        fit_indices = np.asarray(
            sorted(roles["train"] + roles["validation"]), dtype=np.int64
        )
        calibration_indices = np.asarray(roles["calibration"], dtype=np.int64)
        test_indices = np.asarray(roles["test"], dtype=np.int64)
        weights = _fit_weights_only(
            matrix[:, fit_indices],
            labels[fit_indices],
            steps=steps,
            learning_rate=learning_rate,
            regularization=regularization,
            restarts=restarts,
            seed=seed + outer_fold * 100,
        )
        weight_rows.append(weights["weights"])
        candidate_calibration_raw = _weighted_probability(
            matrix[:, calibration_indices], weights["weights"]
        )
        candidate_calibration, candidate_threshold = _calibrate_and_select_threshold(
            candidate_calibration_raw,
            labels[calibration_indices],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        candidate_test = _apply_numpy_calibration(
            _weighted_probability(matrix[:, test_indices], weights["weights"]),
            candidate_calibration,
        )
        candidate_test_predictions = candidate_test >= candidate_threshold["threshold"]
        candidate_probabilities[test_indices] = candidate_test
        candidate_predictions[test_indices] = candidate_test_predictions

        baseline_calibration, baseline_threshold = _calibrate_and_select_threshold(
            baseline_probabilities[calibration_indices],
            labels[calibration_indices],
            threshold_step=threshold_step,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        baseline_test = _apply_numpy_calibration(
            baseline_probabilities[test_indices], baseline_calibration
        )
        baseline_test_predictions = baseline_test >= baseline_threshold["threshold"]
        nested_baseline_probabilities[test_indices] = baseline_test
        nested_baseline_predictions[test_indices] = baseline_test_predictions

        candidate_metrics = _crossfit_metrics(
            labels[test_indices], candidate_test, candidate_test_predictions
        )
        nested_baseline_metrics = _crossfit_metrics(
            labels[test_indices], baseline_test, baseline_test_predictions
        )
        current_default_metrics = _crossfit_metrics(
            labels[test_indices],
            baseline_probabilities[test_indices],
            current_default_predictions[test_indices],
        )
        mcc_gain = candidate_metrics["mcc"] - current_default_metrics["mcc"]
        fold_wins += int(mcc_gain > 0.0)
        fold_results.append(
            {
                "outer_fold": outer_fold,
                "role_sizes": {
                    "weight_fit": int(len(fit_indices)),
                    "calibration": int(len(calibration_indices)),
                    "outer_test": int(len(test_indices)),
                },
                "weights": {
                    name: float(weight)
                    for name, weight in zip(names, weights["weights"])
                },
                "candidate_platt": candidate_calibration,
                "candidate_threshold": candidate_threshold["threshold"],
                "baseline_platt": baseline_calibration,
                "baseline_threshold": baseline_threshold["threshold"],
                "candidate_metrics": candidate_metrics,
                "nested_recalibrated_baseline_metrics": nested_baseline_metrics,
                "current_default_metrics": current_default_metrics,
                "mcc_gain_vs_current_default": mcc_gain,
            }
        )
    if np.isnan(candidate_probabilities).any() or np.isnan(
        nested_baseline_probabilities
    ).any():
        raise RuntimeError("Nested outer tests did not cover every OOF sample exactly once.")
    candidate = _crossfit_metrics(labels, candidate_probabilities, candidate_predictions)
    nested_baseline = _crossfit_metrics(
        labels, nested_baseline_probabilities, nested_baseline_predictions
    )
    current_default = _crossfit_metrics(
        labels, baseline_probabilities, current_default_predictions
    )
    deltas = {
        "mcc": candidate["mcc"] - current_default["mcc"],
        "auroc": candidate["auroc"] - current_default["auroc"],
        "auprc": candidate["auprc"] - current_default["auprc"],
        "brier_lower_is_better": current_default["brier"] - candidate["brier"],
        "ece_10_lower_is_better": current_default["ece_10"] - candidate["ece_10"],
    }
    nested_deltas = {
        "mcc": candidate["mcc"] - nested_baseline["mcc"],
        "auroc": candidate["auroc"] - nested_baseline["auroc"],
        "auprc": candidate["auprc"] - nested_baseline["auprc"],
        "brier_lower_is_better": nested_baseline["brier"] - candidate["brier"],
        "ece_10_lower_is_better": nested_baseline["ece_10"] - candidate["ece_10"],
    }
    gates = {
        "mcc_strictly_improves_current_default": candidate["mcc"]
        > current_default["mcc"],
        "auroc_within_0.0005_of_current_default": candidate["auroc"]
        >= current_default["auroc"] - 0.0005,
        "auprc_not_lower_than_current_default": candidate["auprc"]
        >= current_default["auprc"],
        "brier_within_0.0005_of_current_default": candidate["brier"]
        <= current_default["brier"] + 0.0005,
        "ece_within_0.002_of_current_default": candidate["ece_10"]
        <= current_default["ece_10"] + 0.002,
        "mcc_improves_in_at_least_three_folds": fold_wins >= 3,
    }
    retain = all(gates.values())
    frozen_weights = _fit_weights_only(
        matrix,
        labels,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
        restarts=restarts,
        seed=seed + 999,
    )
    frozen_raw = _weighted_probability(matrix, frozen_weights["weights"])
    frozen_calibration, frozen_threshold = _calibrate_and_select_threshold(
        frozen_raw,
        labels,
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    weight_matrix = np.asarray(weight_rows, dtype=np.float64)
    summary = {
        "protocol": {
            "selection_data": "aligned base-model OOF predictions",
            "nested_manifest": str(nested_manifest),
            "weight_fit_roles": ["train", "validation"],
            "calibration_role": "calibration only",
            "evaluation_role": "outer test only",
            "member_names": names,
            "model": "non-negative probability weights + monotonic Platt + calibration-only MCC threshold",
            "steps": int(steps),
            "learning_rate": float(learning_rate),
            "regularization": float(regularization),
            "restarts": int(restarts),
            "threshold_step": float(threshold_step),
            "constraints": {"min_sn": min_sn, "min_sp": min_sp},
            "test1_or_test2_predictions_read": False,
        },
        "current_default": current_default,
        "nested_recalibrated_baseline": nested_baseline,
        "nested_stacking": candidate,
        "delta_stacking_minus_current_default": deltas,
        "delta_stacking_minus_nested_recalibrated_baseline": nested_deltas,
        "fold_results": fold_results,
        "fold_mcc_wins": fold_wins,
        "weight_stability": {
            name: {
                "mean": float(weight_matrix[:, index].mean()),
                "std": float(weight_matrix[:, index].std()),
            }
            for index, name in enumerate(names)
        },
        "frozen_all_oof_pipeline": {
            "weights": {
                name: float(weight)
                for name, weight in zip(names, frozen_weights["weights"])
            },
            "platt": frozen_calibration,
            "threshold": frozen_threshold,
            "warning": "Fitted on all OOF rows for later untouched external application; nested outer-test metrics control retention.",
        },
        "retention_gates": gates,
        "decision": {
            "retain_stacking": retain,
            "replace_default": retain,
            "run_historical_tests": False,
            "reason": (
                "All gates against the current frozen default passed."
                if retain
                else "At least one gate against the current frozen default failed."
            ),
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
                "sample_id",
                "sequence",
                "label",
                "baseline_probability",
                "baseline_prediction",
                "toxicity_probability",
                "prediction",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "sample_id": row["sample_id"],
                    "sequence": row.get("sequence", ""),
                    "label": int(labels[index]),
                    "baseline_probability": f"{nested_baseline_probabilities[index]:.10g}",
                    "baseline_prediction": int(nested_baseline_predictions[index]),
                    "toxicity_probability": f"{candidate_probabilities[index]:.10g}",
                    "prediction": int(candidate_predictions[index]),
                }
            )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--nested-manifest",
        help="Use disjoint weight-fit/calibration/outer-test roles from this manifest.",
    )
    parser.add_argument("--member-names", nargs="+")
    parser.add_argument("--fold-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--min-sn", type=float, default=0.875)
    parser.add_argument("--min-sp", type=float, default=0.96)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    common = {
        "member_names": args.member_names,
        "threshold_step": args.threshold_step,
        "min_sn": args.min_sn,
        "min_sp": args.min_sp,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "regularization": args.regularization,
        "restarts": args.restarts,
        "seed": args.seed,
    }
    if args.nested_manifest:
        summary = run_nested_calibrated_stacking(
            args.predictions,
            args.baseline,
            args.nested_manifest,
            args.output_dir,
            **common,
        )
        print(json.dumps(summary["nested_stacking"], indent=2))
    else:
        summary = run_nonnegative_stacking(
            args.predictions,
            args.baseline,
            args.output_dir,
            fold_manifest=args.fold_manifest,
            **common,
        )
        print(json.dumps(summary["stacking_crossfit"], indent=2))
    print(json.dumps(summary["retention_gates"], indent=2))
    print(json.dumps(summary["decision"], indent=2))


if __name__ == "__main__":
    main()
