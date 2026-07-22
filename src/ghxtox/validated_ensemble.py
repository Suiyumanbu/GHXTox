"""Cross-fit and freeze a validation-only multi-model probability ensemble."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.plm_textcnn import _probability_metrics
from ghxtox.utils import save_json


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_table(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"sample_id", "label", "toxicity_probability"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Prediction file {path} contains duplicate sample_id values.")
    probabilities = np.asarray(
        [float(row["toxicity_probability"]) for row in rows], dtype=np.float64
    )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError(f"Prediction file {path} contains invalid probabilities.")
    return rows


def _aligned_matrix(
    paths: list[str | Path],
) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    if len(paths) < 2:
        raise ValueError("At least two prediction files are required.")
    tables = [_read_table(path) for path in paths]
    reference = tables[0]
    for table_index, table in enumerate(tables[1:], start=1):
        if len(table) != len(reference):
            raise ValueError(
                f"Prediction row counts differ for member 0 and member {table_index}."
            )
        for row_index, (left, right) in enumerate(zip(reference, table)):
            # Historical assets use both ``peptide_1`` and
            # ``train_000001|1`` for the same immutable source row.  Accept
            # that alias only when source_index, sequence and label agree.
            keys = ["label"]
            if "source_index" in left and "source_index" in right:
                keys.append("source_index")
            else:
                keys.append("sample_id")
            if "sequence" in left and "sequence" in right:
                keys.append("sequence")
            for key in keys:
                if left[key] != right[key]:
                    raise ValueError(
                        f"Prediction alignment mismatch at member {table_index}, "
                        f"row {row_index}, column {key}."
                    )
    labels = np.asarray([int(row["label"]) for row in reference], dtype=np.int64)
    if not set(labels.tolist()).issubset({0, 1}):
        raise ValueError("Prediction labels must be binary.")
    matrix = np.asarray(
        [
            [float(row["toxicity_probability"]) for row in table]
            for table in tables
        ],
        dtype=np.float64,
    )
    return reference, labels, matrix


def _folds_from_rows(
    rows: list[dict[str, str]], fold_manifest: str | Path | None
) -> np.ndarray:
    if "fold" in rows[0]:
        folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    elif fold_manifest is not None:
        with Path(fold_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
            manifest = list(csv.DictReader(handle))
        by_index = {int(row["source_index"]): row for row in manifest}
        folds_list = []
        for position, row in enumerate(rows):
            source_index = int(row.get("source_index", position))
            if source_index not in by_index:
                raise ValueError(f"Fold manifest is missing source_index {source_index}.")
            manifest_row = by_index[source_index]
            if int(manifest_row["label"]) != int(row["label"]):
                raise ValueError(f"Fold manifest label mismatch at source_index {source_index}.")
            if "sequence" in row and manifest_row.get("sequence") != row["sequence"]:
                raise ValueError(
                    f"Fold manifest sequence mismatch at source_index {source_index}."
                )
            folds_list.append(int(manifest_row["fold"]))
        folds = np.asarray(folds_list, dtype=np.int64)
    else:
        raise ValueError(
            "Validation-only ensemble search requires a fold column or --fold-manifest."
        )
    if np.unique(folds).size < 2:
        raise ValueError("Cross-fitted ensemble search requires at least two folds.")
    return folds


def _threshold_search(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    metric: str,
    threshold_step: float,
    min_sn: float | None,
    min_sp: float | None,
) -> dict[str, Any]:
    if not 0.0 < float(threshold_step) < 1.0:
        raise ValueError("threshold_step must be between 0 and 1.")
    candidates = np.arange(threshold_step, 1.0, threshold_step, dtype=np.float64)
    candidates = np.unique(np.concatenate([candidates, np.asarray([0.5])]))
    predictions = probabilities[:, None] >= candidates[None, :]
    positives = labels[:, None] == 1
    negatives = ~positives
    tp = np.sum(predictions & positives, axis=0, dtype=np.int64)
    tn = np.sum(~predictions & negatives, axis=0, dtype=np.int64)
    fp = np.sum(predictions & negatives, axis=0, dtype=np.int64)
    fn = np.sum(~predictions & positives, axis=0, dtype=np.int64)
    recall = tp / np.maximum(tp + fn, 1)
    specificity = tn / np.maximum(tn + fp, 1)
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    accuracy = (tp + tn) / np.maximum(tp + tn + fp + fn, 1)
    denominator = np.sqrt(
        (tp + fp).astype(np.float64) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = (tp * tn - fp * fn) / np.maximum(denominator, 1.0)
    metric_values = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
    }[metric]
    feasible = np.ones(len(candidates), dtype=bool)
    if min_sn is not None:
        feasible &= recall >= min_sn
    if min_sp is not None:
        feasible &= specificity >= min_sp
    margin = np.minimum(
        recall - (min_sn if min_sn is not None else 0.0),
        specificity - (min_sp if min_sp is not None else 0.0),
    )
    primary = np.where(feasible, metric_values, margin)
    eligible = np.flatnonzero(feasible) if np.any(feasible) else np.arange(len(candidates))
    chosen_index = max(
        eligible.tolist(),
        key=lambda index: (
            float(primary[index]),
            float(mcc[index]),
            float(balanced_accuracy[index]),
            -abs(float(candidates[index]) - 0.5),
        ),
    )
    chosen_threshold = float(candidates[chosen_index])
    metrics = _probability_metrics(labels, probabilities, chosen_threshold)
    return {
        "threshold": chosen_threshold,
        "metrics": metrics,
        "constraints_satisfied": bool(np.any(feasible)),
    }


def _subset_candidate(
    labels: np.ndarray,
    matrix: np.ndarray,
    subset: tuple[int, ...],
    *,
    objective: str,
    threshold_metric: str,
    threshold_step: float,
    min_sn: float | None,
    min_sp: float | None,
) -> dict[str, Any]:
    probabilities = matrix[list(subset)].mean(axis=0)
    threshold_result = _threshold_search(
        labels,
        probabilities,
        metric=threshold_metric,
        threshold_step=threshold_step,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    metrics = threshold_result["metrics"]
    score = (
        int(threshold_result["constraints_satisfied"]),
        float(metrics[objective]),
        float(metrics["mcc"]),
        float(metrics["auprc"]),
        -len(subset),
    )
    return {
        "member_indices": list(subset),
        "threshold": threshold_result["threshold"],
        "constraints_satisfied": threshold_result["constraints_satisfied"],
        "metrics": metrics,
        "score": score,
    }


def _greedy_search(
    labels: np.ndarray,
    matrix: np.ndarray,
    *,
    objective: str,
    threshold_metric: str,
    threshold_step: float,
    max_members: int,
    min_delta: float,
    min_sn: float | None,
    min_sp: float | None,
) -> dict[str, Any]:
    max_members = min(max(int(max_members), 1), matrix.shape[0])
    selected: tuple[int, ...] = ()
    best: dict[str, Any] | None = None
    trace = []
    while len(selected) < max_members:
        candidates = []
        for member in range(matrix.shape[0]):
            if member in selected:
                continue
            subset = tuple(sorted((*selected, member)))
            candidates.append(
                _subset_candidate(
                    labels,
                    matrix,
                    subset,
                    objective=objective,
                    threshold_metric=threshold_metric,
                    threshold_step=threshold_step,
                    min_sn=min_sn,
                    min_sp=min_sp,
                )
            )
        candidate = max(candidates, key=lambda item: item["score"])
        previous_objective = (
            None if best is None else float(best["metrics"][objective])
        )
        improvement = (
            None
            if previous_objective is None
            else float(candidate["metrics"][objective]) - previous_objective
        )
        trace.append(
            {
                "step": len(selected) + 1,
                "candidate_member_indices": candidate["member_indices"],
                "objective": float(candidate["metrics"][objective]),
                "objective_improvement": improvement,
                "threshold": candidate["threshold"],
                "constraints_satisfied": candidate["constraints_satisfied"],
            }
        )
        if best is not None and improvement is not None and improvement <= float(min_delta):
            break
        selected = tuple(candidate["member_indices"])
        best = candidate
    if best is None:
        raise RuntimeError("Ensemble search produced no candidate.")
    best = dict(best)
    best.pop("score", None)
    best["weights"] = [1.0 / len(best["member_indices"])] * len(
        best["member_indices"]
    )
    best["search_trace"] = trace
    return best


def _crossfit_metrics(
    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    base = _probability_metrics(labels, probabilities, 0.5)
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.sum(predictions & positives))
    tn = int(np.sum(~predictions & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    base.update(
        {
            "accuracy": (tp + tn) / max(len(labels), 1),
            "balanced_accuracy": 0.5 * (recall + specificity),
            "precision": precision,
            "sensitivity": recall,
            "sn": recall,
            "recall": recall,
            "specificity": specificity,
            "sp": specificity,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "mcc": (tp * tn - fp * fn) / denominator,
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
        }
    )
    base.pop("threshold", None)
    return base


def screen_validated_ensemble(
    prediction_paths: list[str | Path],
    output_dir: str | Path,
    *,
    member_names: list[str] | None = None,
    fold_manifest: str | Path | None = None,
    objective: str = "mcc",
    threshold_metric: str = "mcc",
    threshold_step: float = 0.005,
    max_members: int = 7,
    min_delta: float = 0.0,
    min_sn: float | None = None,
    min_sp: float | None = None,
) -> dict[str, Any]:
    rows, labels, matrix = _aligned_matrix(prediction_paths)
    names = member_names or [Path(path).stem for path in prediction_paths]
    if len(names) != len(prediction_paths) or len(names) != len(set(names)):
        raise ValueError("Member names must be unique and match the prediction file count.")
    folds = _folds_from_rows(rows, fold_manifest)
    crossfit_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    crossfit_predictions = np.zeros(len(labels), dtype=bool)
    fold_results = []
    for heldout_fold in sorted(np.unique(folds).tolist()):
        train_mask = folds != heldout_fold
        heldout_mask = folds == heldout_fold
        selected = _greedy_search(
            labels[train_mask],
            matrix[:, train_mask],
            objective=objective,
            threshold_metric=threshold_metric,
            threshold_step=threshold_step,
            max_members=max_members,
            min_delta=min_delta,
            min_sn=min_sn,
            min_sp=min_sp,
        )
        indices = selected["member_indices"]
        heldout_probs = matrix[indices][:, heldout_mask].mean(axis=0)
        crossfit_probabilities[heldout_mask] = heldout_probs
        crossfit_predictions[heldout_mask] = heldout_probs >= selected["threshold"]
        fold_results.append(
            {
                "heldout_fold": int(heldout_fold),
                "member_indices": indices,
                "member_names": [names[index] for index in indices],
                "threshold": selected["threshold"],
                "training_metrics": selected["metrics"],
                "constraints_satisfied": selected["constraints_satisfied"],
            }
        )
    if np.isnan(crossfit_probabilities).any():
        raise RuntimeError("Cross-fitted ensemble did not cover every validation row.")
    frozen = _greedy_search(
        labels,
        matrix,
        objective=objective,
        threshold_metric=threshold_metric,
        threshold_step=threshold_step,
        max_members=max_members,
        min_delta=min_delta,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    frozen_indices = frozen["member_indices"]
    frozen_spec = {
        "version": 1,
        "selection_data_role": "group-aware OOF/validation only",
        "test_labels_used_for_selection": False,
        "all_member_names": names,
        "selected_member_indices": frozen_indices,
        "selected_member_names": [names[index] for index in frozen_indices],
        "weights": frozen["weights"],
        "threshold": frozen["threshold"],
        "objective": objective,
        "threshold_metric": threshold_metric,
        "constraints": {"min_sn": min_sn, "min_sp": min_sp},
    }
    summary = {
        "protocol": {
            "prediction_paths": [str(path) for path in prediction_paths],
            "prediction_sha256": [_sha256(path) for path in prediction_paths],
            "member_names": names,
            "fold_manifest": None if fold_manifest is None else str(fold_manifest),
            "selection": "leave-one-fold-out greedy uniform-subset search",
            "objective": objective,
            "threshold_metric": threshold_metric,
            "threshold_step": float(threshold_step),
            "max_members": int(max_members),
            "min_delta": float(min_delta),
            "screen_interface_has_no_separate_test_argument": True,
        },
        "crossfit_fold_selection": fold_results,
        "crossfit_metrics": _crossfit_metrics(
            labels, crossfit_probabilities, crossfit_predictions
        ),
        "frozen_selection_from_all_oof": frozen,
        "frozen_spec": frozen_spec,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, output_dir / "summary.json")
    save_json(frozen_spec, output_dir / "frozen_spec.json")
    with (output_dir / "crossfit_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "source_index",
            "fold",
            "sample_id",
            "sequence",
            "label",
            "toxicity_probability",
            "prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def apply_frozen_ensemble(
    spec_path: str | Path,
    prediction_paths: list[str | Path],
    output_csv: str | Path,
) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows, labels, matrix = _aligned_matrix(prediction_paths)
    expected = list(spec["all_member_names"])
    if len(prediction_paths) != len(expected):
        raise ValueError(
            f"Frozen spec expects {len(expected)} member files, got {len(prediction_paths)}."
        )
    selected = [int(index) for index in spec["selected_member_indices"]]
    weights = np.asarray(spec["weights"], dtype=np.float64)
    if len(selected) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Frozen ensemble weights are invalid.")
    probabilities = np.average(matrix[selected], axis=0, weights=weights)
    threshold = float(spec["threshold"])
    metrics = _probability_metrics(labels, probabilities, threshold)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "sequence",
            "label",
            "toxicity_probability",
            "prediction",
            "decision_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "source_index": row.get("source_index", index),
                    "sample_id": row["sample_id"],
                    "sequence": row.get("sequence", ""),
                    "label": int(labels[index]),
                    "toxicity_probability": f"{probabilities[index]:.10g}",
                    "prediction": int(probabilities[index] >= threshold),
                    "decision_threshold": f"{threshold:.10g}",
                }
            )
    summary = {
        "protocol": {
            "frozen_spec": str(spec_path),
            "frozen_spec_sha256": _sha256(spec_path),
            "prediction_paths": [str(path) for path in prediction_paths],
            "selected_member_names": spec["selected_member_names"],
            "weights": weights.tolist(),
            "threshold": threshold,
            "parameters_selected_from_applied_labels": False,
        },
        "metrics": metrics,
    }
    save_json(summary, output_csv.with_suffix(".metrics.json"))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--predictions", nargs="+", required=True)
    screen.add_argument("--member-names", nargs="+")
    screen.add_argument("--fold-manifest")
    screen.add_argument("--output-dir", required=True)
    screen.add_argument(
        "--objective",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc", "auroc", "auprc"],
        default="mcc",
    )
    screen.add_argument(
        "--threshold-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc"],
        default="mcc",
    )
    screen.add_argument("--max-members", type=int, default=7)
    screen.add_argument("--threshold-step", type=float, default=0.005)
    screen.add_argument("--min-delta", type=float, default=0.0)
    screen.add_argument("--min-sn", type=float)
    screen.add_argument("--min-sp", type=float)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--spec", required=True)
    apply_parser.add_argument("--predictions", nargs="+", required=True)
    apply_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "screen":
        summary = screen_validated_ensemble(
            args.predictions,
            args.output_dir,
            member_names=args.member_names,
            fold_manifest=args.fold_manifest,
            objective=args.objective,
            threshold_metric=args.threshold_metric,
            threshold_step=args.threshold_step,
            max_members=args.max_members,
            min_delta=args.min_delta,
            min_sn=args.min_sn,
            min_sp=args.min_sp,
        )
        print(json.dumps(summary["crossfit_metrics"], indent=2))
        print(json.dumps(summary["frozen_spec"], indent=2))
        return
    summary = apply_frozen_ensemble(args.spec, args.predictions, args.output)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
