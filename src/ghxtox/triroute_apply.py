"""Apply a frozen OOF-trained confidence router to three expert prediction files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from ghxtox.metrics import binary_metrics
from ghxtox.triroute import (
    ConfidenceRouter,
    FittedReliability,
    ReliabilityRegressor,
    ROUTE_NAMES,
    _apply_temperature,
    _calibration_metrics,
    _feature_tensor,
    _logit,
    _read_rows,
)


def _restore_reliability(state: dict) -> FittedReliability:
    mean = torch.tensor(state["mean"], dtype=torch.float32)
    std = torch.tensor(state["std"], dtype=torch.float32)
    model = ReliabilityRegressor(len(mean))
    model.load_state_dict({key: torch.tensor(value, dtype=torch.float32) for key, value in state["model_state"].items()})
    model.eval()
    return FittedReliability(model=model, mean=mean, std=std)


def _restore_router(state: dict) -> ConfidenceRouter:
    router = ConfidenceRouter()
    with torch.no_grad():
        router.bias.copy_(torch.tensor(state["bias"], dtype=torch.float32))
        router.log_temperature.copy_(torch.tensor(math.log(float(state["temperature"]))))
        beta = max(float(state["fallback_beta"]), 1e-6)
        router.fallback_raw.copy_(torch.tensor(math.log(math.expm1(beta))))
    router.eval()
    return router


def apply_frozen_router(
    prediction_paths: list[str | Path],
    router_summary_path: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
) -> dict:
    if len(prediction_paths) != 3:
        raise ValueError("Exactly three prediction files are required in 1D, 2D, 3D order.")
    routes = [_read_rows(path) for path in prediction_paths]
    if any(len(rows) != len(routes[0]) for rows in routes):
        raise ValueError("Prediction files contain different sample counts.")
    for rows in routes:
        rows.sort(key=lambda row: int(row["source_index"]))
    for index in range(len(routes[0])):
        reference = routes[0][index]
        for route_index in (1, 2):
            for key in ("source_index", "sample_id", "sequence", "label"):
                if routes[route_index][index][key] != reference[key]:
                    raise ValueError(f"Route {route_index} is misaligned at row {index} for {key!r}.")

    summary = json.loads(Path(router_summary_path).read_text(encoding="utf-8"))
    frozen = summary["frozen_full_oof_fit"]
    raw_probability = torch.stack(
        [torch.tensor([float(row["toxicity_probability"]) for row in rows]) for rows in routes], dim=1
    ).float()
    calibrated = torch.stack(
        [
            _apply_temperature(raw_probability[:, index], float(frozen["temperatures"][index]))
            for index in range(3)
        ],
        dim=1,
    )
    confidence = torch.stack(
        [
            _restore_reliability(frozen["reliability"][index]).predict(_feature_tensor(routes[index], index))
            for index in range(3)
        ],
        dim=1,
    )
    router = _restore_router(frozen["router"])
    with torch.no_grad():
        probability, weights = router(calibrated, confidence)
    overall_confidence = (weights * confidence).sum(dim=1)
    labels = torch.tensor([float(row["label"]) for row in routes[0]], dtype=torch.float32)
    threshold = float(frozen["decision_threshold"])
    uniform_probability = calibrated.mean(dim=1)
    result = {
        "protocol": {
            "prediction_files": [str(path) for path in prediction_paths],
            "router_summary": str(router_summary_path),
            "threshold": threshold,
            "test_labels_used_for_fitting": False,
        },
        "dynamic_router_at_frozen_threshold": binary_metrics(_logit(probability), labels, threshold=threshold),
        "dynamic_router_at_0.5": binary_metrics(_logit(probability), labels, threshold=0.5),
        "dynamic_calibration": _calibration_metrics(probability, labels),
        "uniform_mean_at_frozen_threshold": binary_metrics(_logit(uniform_probability), labels, threshold=threshold),
        "uniform_mean_at_0.5": binary_metrics(_logit(uniform_probability), labels, threshold=0.5),
        "uniform_calibration": _calibration_metrics(uniform_probability, labels),
        "mean_weights": {name: float(weights[:, index].mean()) for index, name in enumerate(ROUTE_NAMES)},
        "mean_confidence": {name: float(confidence[:, index].mean()) for index, name in enumerate(ROUTE_NAMES)},
        "mean_overall_confidence": float(overall_confidence.mean()),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(routes[0][0]) + [
        "calibrated_probability_1d", "calibrated_probability_2d", "calibrated_probability_3d",
        "confidence_1d", "confidence_2d", "confidence_3d",
        "weight_1d", "weight_2d", "weight_3d",
        "routed_probability", "prediction", "overall_confidence",
    ]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, base in enumerate(routes[0]):
            row = dict(base)
            row.update({
                "calibrated_probability_1d": f"{float(calibrated[index, 0]):.9g}",
                "calibrated_probability_2d": f"{float(calibrated[index, 1]):.9g}",
                "calibrated_probability_3d": f"{float(calibrated[index, 2]):.9g}",
                "confidence_1d": f"{float(confidence[index, 0]):.9g}",
                "confidence_2d": f"{float(confidence[index, 1]):.9g}",
                "confidence_3d": f"{float(confidence[index, 2]):.9g}",
                "weight_1d": f"{float(weights[index, 0]):.9g}",
                "weight_2d": f"{float(weights[index, 1]):.9g}",
                "weight_3d": f"{float(weights[index, 2]):.9g}",
                "routed_probability": f"{float(probability[index]):.9g}",
                "prediction": int(probability[index] >= threshold),
                "overall_confidence": f"{float(overall_confidence[index]):.9g}",
            })
            writer.writerow(row)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a frozen three-route confidence router.")
    parser.add_argument("--one-dimensional", required=True)
    parser.add_argument("--two-dimensional", required=True)
    parser.add_argument("--three-dimensional", required=True)
    parser.add_argument("--router-summary", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = apply_frozen_router(
        [args.one_dimensional, args.two_dimensional, args.three_dimensional],
        args.router_summary,
        args.output_csv,
        args.output_json,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
