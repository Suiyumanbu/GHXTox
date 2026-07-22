"""Fit a frozen probability calibrator and threshold without touching test labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ghxtox.metrics import binary_metrics
from ghxtox.utils import save_json


def _logit(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities.float().clamp(1e-6, 1.0 - 1e-6))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_predictions(
    path: str | Path,
) -> tuple[list[dict[str, str]], torch.Tensor, torch.Tensor]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"sample_id", "label", "toxicity_probability"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Prediction file {path} contains duplicate sample_id values.")
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.float32)
    if not set(labels.tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"Prediction file {path} contains non-binary labels.")
    probabilities = torch.tensor(
        [float(row["toxicity_probability"]) for row in rows], dtype=torch.float32
    )
    if not torch.isfinite(probabilities).all() or bool(
        ((probabilities < 0.0) | (probabilities > 1.0)).any()
    ):
        raise ValueError(f"Prediction file {path} contains invalid probabilities.")
    return rows, labels, probabilities


def fit_temperature(
    probabilities: torch.Tensor, labels: torch.Tensor, steps: int = 200
) -> dict[str, float | str]:
    logits = _logit(probabilities)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max(int(steps), 1),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels.float())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "temperature",
        "temperature": float(log_temperature.detach().exp().clamp(0.05, 20.0)),
        "monotonic": True,
    }


def fit_platt(
    probabilities: torch.Tensor, labels: torch.Tensor, steps: int = 200
) -> dict[str, float | str | bool]:
    """Fit monotonic Platt scaling; positive scale preserves probability ranking."""

    logits = _logit(probabilities)
    log_scale = torch.nn.Parameter(torch.zeros(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_scale, bias],
        lr=0.1,
        max_iter=max(int(steps), 1),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scale = log_scale.exp().clamp(0.02, 50.0)
        loss = F.binary_cross_entropy_with_logits(
            scale * logits + bias, labels.float()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "method": "platt",
        "scale": float(log_scale.detach().exp().clamp(0.02, 50.0)),
        "bias": float(bias.detach()),
        "monotonic": True,
    }


def apply_calibration(
    probabilities: torch.Tensor, calibration: dict[str, Any]
) -> torch.Tensor:
    method = str(calibration["method"])
    logits = _logit(probabilities)
    if method == "temperature":
        return torch.sigmoid(logits / max(float(calibration["temperature"]), 1e-6))
    if method == "platt":
        return torch.sigmoid(
            max(float(calibration["scale"]), 1e-6) * logits
            + float(calibration["bias"])
        )
    if method in {"identity", "none"}:
        return probabilities.float().clamp(0.0, 1.0)
    raise ValueError(f"Unsupported calibration method: {method}")


def _best_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    metric: str,
    min_sn: float | None = None,
    min_sp: float | None = None,
) -> tuple[float, dict[str, float], bool]:
    candidates = torch.linspace(0.01, 0.99, 99).tolist()
    candidates.extend(float(value) for value in probabilities.unique().tolist())
    candidates = sorted(
        set(round(float(value), 6) for value in candidates if 0.0 < value < 1.0)
    )
    logits = _logit(probabilities)
    evaluated: list[tuple[bool, tuple[float, ...], float, dict[str, float]]] = []
    for threshold in candidates:
        metrics = binary_metrics(logits, labels, threshold=threshold)
        feasible = (min_sn is None or metrics["sn"] >= min_sn) and (
            min_sp is None or metrics["sp"] >= min_sp
        )
        constraint_margin = min(
            metrics["sn"] - (min_sn if min_sn is not None else 0.0),
            metrics["sp"] - (min_sp if min_sp is not None else 0.0),
        )
        score = (
            float(metrics[metric]),
            float(metrics["mcc"]),
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - 0.5),
        )
        if not feasible:
            score = (constraint_margin, *score[1:])
        evaluated.append((feasible, score, float(threshold), metrics))
    feasible_rows = [row for row in evaluated if row[0]]
    pool = feasible_rows if feasible_rows else evaluated
    selected = max(pool, key=lambda row: row[1])
    return selected[2], selected[3], bool(feasible_rows)


def _write_predictions(
    rows: list[dict[str, str]],
    calibrated: torch.Tensor,
    output_path: str | Path,
    threshold: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for column in (
        "raw_toxicity_probability",
        "toxicity_probability",
        "prediction",
        "decision_threshold",
    ):
        if column not in fieldnames:
            fieldnames.append(column)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, calibrated.tolist(), strict=True):
            output = dict(row)
            output["raw_toxicity_probability"] = row["toxicity_probability"]
            output["toxicity_probability"] = f"{float(probability):.10g}"
            output["prediction"] = int(float(probability) >= threshold)
            output["decision_threshold"] = f"{threshold:.10g}"
            writer.writerow(output)


def calibrate_predictions(
    calibration_path: str | Path,
    apply_paths: list[str | Path],
    output_dir: str | Path,
    output_json: str | Path,
    *,
    method: str = "platt",
    threshold_selection_path: str | Path | None = None,
    threshold_metric: str = "mcc",
    min_sn: float | None = None,
    min_sp: float | None = None,
    steps: int = 200,
) -> dict[str, Any]:
    calibration_rows, calibration_labels, calibration_probs = _read_predictions(
        calibration_path
    )
    if method == "platt":
        calibrator = fit_platt(calibration_probs, calibration_labels, steps=steps)
    elif method == "temperature":
        calibrator = fit_temperature(calibration_probs, calibration_labels, steps=steps)
    elif method == "identity":
        calibrator = {"method": "identity", "monotonic": True}
    else:
        raise ValueError(f"Unsupported calibration method: {method}")

    calibrated_fit = apply_calibration(calibration_probs, calibrator)
    threshold_path = threshold_selection_path or calibration_path
    _, threshold_labels, threshold_probs = _read_predictions(threshold_path)
    calibrated_threshold_probs = apply_calibration(threshold_probs, calibrator)
    threshold, threshold_metrics, constraints_satisfied = _best_threshold(
        calibrated_threshold_probs,
        threshold_labels,
        threshold_metric,
        min_sn=min_sn,
        min_sp=min_sp,
    )
    fit_before = binary_metrics(_logit(calibration_probs), calibration_labels, threshold=0.5)
    fit_after = binary_metrics(_logit(calibrated_fit), calibration_labels, threshold=0.5)
    summary: dict[str, Any] = {
        "protocol": {
            "calibration_predictions": str(calibration_path),
            "calibration_sha256": _sha256(calibration_path),
            "threshold_selection_predictions": str(threshold_path),
            "threshold_selection_sha256": _sha256(threshold_path),
            "shared_calibration_and_threshold_data": Path(calibration_path).resolve()
            == Path(threshold_path).resolve(),
            "required_role": "dedicated group-disjoint calibration/threshold split",
            "role_enforcement": (
                "The command has no separate test-selection argument; the caller must supply "
                "only a dedicated calibration split."
            ),
        },
        "calibrator": calibrator,
        "calibration_num_samples": len(calibration_rows),
        "calibration_at_0.5_before": fit_before,
        "calibration_at_0.5_after": fit_after,
        "threshold_selection": {
            "metric": threshold_metric,
            "threshold": threshold,
            "min_sn": min_sn,
            "min_sp": min_sp,
            "constraints_satisfied": constraints_satisfied,
            "metrics": threshold_metrics,
        },
        "applied": [],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in apply_paths:
        rows, labels, probabilities = _read_predictions(path)
        calibrated = apply_calibration(probabilities, calibrator)
        output_path = output_dir / f"{Path(path).stem}_calibrated.csv"
        _write_predictions(rows, calibrated, output_path, threshold)
        before = binary_metrics(_logit(probabilities), labels, threshold=0.5)
        after = binary_metrics(_logit(calibrated), labels, threshold=threshold)
        ranking_delta = {
            "auroc": float(after["auroc"] - before["auroc"]),
            "auprc": float(after["auprc"] - before["auprc"]),
        }
        if calibrator.get("monotonic") and any(
            math.isfinite(value) and abs(value) > 1e-5 for value in ranking_delta.values()
        ):
            raise RuntimeError(
                "A monotonic calibrator changed ranking metrics beyond numerical tolerance; "
                "check prediction alignment and serialization."
            )
        summary["applied"].append(
            {
                "input": str(path),
                "input_sha256": _sha256(path),
                "output": str(output_path),
                "num_samples": len(rows),
                "raw_at_0.5": before,
                "calibrated_at_frozen_threshold": after,
                "ranking_metric_delta": ranking_delta,
            }
        )
    save_json(summary, output_json)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--threshold-selection")
    parser.add_argument("--apply", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--method", choices=["platt", "temperature", "identity"], default="platt")
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "balanced_accuracy", "f1", "mcc"],
        default="mcc",
    )
    parser.add_argument("--min-sn", type=float)
    parser.add_argument("--min-sp", type=float)
    parser.add_argument("--steps", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = calibrate_predictions(
        args.calibration,
        args.apply,
        args.output_dir,
        args.output_json,
        method=args.method,
        threshold_selection_path=args.threshold_selection,
        threshold_metric=args.threshold_metric,
        min_sn=args.min_sn,
        min_sp=args.min_sp,
        steps=args.steps,
    )
    selection = summary["threshold_selection"]
    print(f"calibration={summary['calibrator']['method']}")
    print(f"threshold={selection['threshold']:.6f}")
    print(f"{selection['metric']}={selection['metrics'][selection['metric']]:.6f}")


if __name__ == "__main__":
    main()
