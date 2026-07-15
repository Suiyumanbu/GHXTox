"""Cross-fitted confidence routing for the 1D, 2D, and 3D experts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ghxtox.metrics import binary_metrics


ROUTE_NAMES = ("one_dimensional", "two_dimensional", "three_dimensional")
BASE_FEATURES = (
    "toxicity_probability",
    "mc_mean_probability",
    "probability_margin",
    "probability_entropy",
    "mc_std_probability",
    "sequence_length",
)
THREE_D_FEATURES = BASE_FEATURES + (
    "mean_plddt",
    "min_plddt",
    "low_plddt_fraction",
    "global_gate",
)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_routes(paths: list[str | Path]) -> tuple[list[list[dict[str, str]]], torch.Tensor, torch.Tensor]:
    routes = [_read_rows(path) for path in paths]
    if not routes or any(len(rows) != len(routes[0]) for rows in routes):
        raise ValueError("All route OOF files must contain the same number of rows.")
    for route_index, rows in enumerate(routes):
        rows.sort(key=lambda row: int(row["source_index"]))
        for index, row in enumerate(rows):
            reference = routes[0][index]
            for key in ("source_index", "sample_id", "sequence", "label", "fold"):
                if row[key] != reference[key]:
                    raise ValueError(f"Route {route_index} is misaligned at row {index} for field {key!r}.")
    labels = torch.tensor([float(row["label"]) for row in routes[0]], dtype=torch.float32)
    folds = torch.tensor([int(row["fold"]) for row in routes[0]], dtype=torch.long)
    return routes, labels, folds


def _logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    return torch.logit(probability)


def _fit_temperature(probabilities: torch.Tensor, labels: torch.Tensor, steps: int = 300) -> float:
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([log_temperature], lr=0.03)
    logits = _logit(probabilities)
    for _ in range(steps):
        temperature = log_temperature.exp().clamp(0.1, 10.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(log_temperature.detach().exp().clamp(0.1, 10.0))


def _apply_temperature(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.sigmoid(_logit(probabilities) / temperature)


def _feature_tensor(rows: list[dict[str, str]], route_index: int) -> torch.Tensor:
    names = THREE_D_FEATURES if route_index == 2 else BASE_FEATURES
    missing = [name for name in names if name not in rows[0]]
    if missing:
        raise ValueError(f"OOF diagnostics are missing {missing}; rerun ghxtox.oof with the current exporter.")
    values = [[float(row[name]) for name in names] for row in rows]
    result = torch.tensor(values, dtype=torch.float32)
    length_index = names.index("sequence_length")
    result[:, length_index] = result[:, length_index].log1p()
    return result


class ReliabilityRegressor(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


@dataclass
class FittedReliability:
    model: ReliabilityRegressor
    mean: torch.Tensor
    std: torch.Tensor

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model((features - self.mean) / self.std).clamp(0.01, 0.99)

    def state(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "model_state": {key: value.detach().tolist() for key, value in self.model.state_dict().items()},
        }


def _fit_reliability(
    features: torch.Tensor,
    calibrated_probability: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
    steps: int = 500,
) -> FittedReliability:
    torch.manual_seed(seed)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (features - mean) / std
    target = 1.0 - (calibrated_probability - labels).abs()
    model = ReliabilityRegressor(features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    for _ in range(steps):
        prediction = model(normalized)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return FittedReliability(model=model, mean=mean, std=std)


class ConfidenceRouter(nn.Module):
    """Interpretable three-way router with an explicit 1D fallback interaction."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(3))
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.fallback_raw = nn.Parameter(torch.tensor(-1.0))

    def weights(self, confidence: torch.Tensor) -> torch.Tensor:
        confidence = confidence.clamp(0.01, 0.99)
        score = torch.logit(confidence) + self.bias
        fallback = (1.0 - confidence[:, 1]) * (1.0 - confidence[:, 2])
        score = score.clone()
        score[:, 0] = score[:, 0] + F.softplus(self.fallback_raw) * fallback
        temperature = self.log_temperature.exp().clamp(0.2, 5.0)
        return torch.softmax(score / temperature, dim=1)

    def forward(self, probability: torch.Tensor, confidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(confidence)
        mixture = (weights * probability).sum(dim=1).clamp(1e-6, 1.0 - 1e-6)
        return mixture, weights

    def state(self) -> dict:
        return {
            "bias": self.bias.detach().tolist(),
            "temperature": float(self.log_temperature.detach().exp().clamp(0.2, 5.0)),
            "fallback_beta": float(F.softplus(self.fallback_raw.detach())),
        }


def _fit_router(
    probabilities: torch.Tensor,
    confidence: torch.Tensor,
    labels: torch.Tensor,
    steps: int = 800,
) -> ConfidenceRouter:
    router = ConfidenceRouter()
    optimizer = torch.optim.AdamW(router.parameters(), lr=0.02, weight_decay=0.01)
    for _ in range(steps):
        mixture, _ = router(probabilities, confidence)
        loss = F.binary_cross_entropy(mixture, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    router.eval()
    return router


def _calibration_metrics(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict[str, float]:
    brier = float(((probabilities - labels) ** 2).mean())
    ece = 0.0
    for lower in torch.linspace(0.0, 0.9, bins):
        upper = lower + 1.0 / bins
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1.0 else probabilities <= upper)
        if selected.any():
            ece += float(selected.float().mean() * (probabilities[selected].mean() - labels[selected].mean()).abs())
    return {"brier": brier, "ece_10": ece}


def _best_threshold(probabilities: torch.Tensor, labels: torch.Tensor) -> tuple[float, dict[str, float]]:
    candidates = torch.linspace(0.05, 0.95, 181).tolist()
    best_threshold = 0.5
    logits = _logit(probabilities)
    best_metrics = binary_metrics(logits, labels, threshold=best_threshold)
    for threshold in candidates:
        metrics = binary_metrics(logits, labels, threshold=float(threshold))
        if metrics["mcc"] > best_metrics["mcc"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _evaluate(probabilities: torch.Tensor, labels: torch.Tensor) -> dict:
    threshold, optimized = _best_threshold(probabilities, labels)
    return {
        "at_0.5": binary_metrics(_logit(probabilities), labels, threshold=0.5),
        "calibration": _calibration_metrics(probabilities, labels),
        "oof_mcc_threshold": threshold,
        "at_oof_mcc_threshold": optimized,
    }


def _selective_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    confidence: torch.Tensor,
) -> dict[str, dict[str, float]]:
    order = torch.argsort(confidence, descending=True)
    correct = ((probabilities >= 0.5) == labels.bool()).float()
    result = {}
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        count = max(1, int(round(len(labels) * coverage)))
        selected = order[:count]
        result[f"coverage_{coverage:.1f}"] = {
            "count": count,
            "accuracy": float(correct[selected].mean()),
            "mean_confidence": float(confidence[selected].mean()),
            "brier": float(((probabilities[selected] - labels[selected]) ** 2).mean()),
        }
    return result


def fit_cross_fitted_router(
    route_paths: list[str | Path],
    output_dir: str | Path,
    seed: int = 42,
) -> dict:
    if len(route_paths) != 3:
        raise ValueError("Exactly three route OOF CSV files are required in 1D, 2D, 3D order.")
    routes, labels, folds = _load_routes(route_paths)
    raw_probability = torch.stack(
        [torch.tensor([float(row["toxicity_probability"]) for row in rows]) for rows in routes], dim=1
    ).float()
    features = [_feature_tensor(rows, index) for index, rows in enumerate(routes)]
    routed_probability = torch.zeros_like(labels)
    routed_weights = torch.zeros((len(labels), 3), dtype=torch.float32)
    routed_confidence = torch.zeros((len(labels), 3), dtype=torch.float32)
    calibrated_probability = torch.zeros_like(raw_probability)
    fold_states = []

    for held_fold in sorted(folds.unique().tolist()):
        train_mask = folds != held_fold
        held_mask = folds == held_fold
        train_probability_columns = []
        held_probability_columns = []
        train_confidence_columns = []
        held_confidence_columns = []
        temperatures = []
        reliability_states = []
        for route_index in range(3):
            temperature = _fit_temperature(raw_probability[train_mask, route_index], labels[train_mask])
            train_probability = _apply_temperature(raw_probability[train_mask, route_index], temperature)
            held_probability = _apply_temperature(raw_probability[held_mask, route_index], temperature)
            reliability = _fit_reliability(
                features[route_index][train_mask],
                train_probability,
                labels[train_mask],
                seed=seed + held_fold * 10 + route_index,
            )
            train_probability_columns.append(train_probability)
            held_probability_columns.append(held_probability)
            train_confidence_columns.append(reliability.predict(features[route_index][train_mask]))
            held_confidence_columns.append(reliability.predict(features[route_index][held_mask]))
            temperatures.append(temperature)
            reliability_states.append(reliability.state())
        train_probability_matrix = torch.stack(train_probability_columns, dim=1)
        held_probability_matrix = torch.stack(held_probability_columns, dim=1)
        train_confidence_matrix = torch.stack(train_confidence_columns, dim=1)
        held_confidence_matrix = torch.stack(held_confidence_columns, dim=1)
        router = _fit_router(train_probability_matrix, train_confidence_matrix, labels[train_mask])
        with torch.no_grad():
            held_mixture, held_weights = router(held_probability_matrix, held_confidence_matrix)
        routed_probability[held_mask] = held_mixture
        routed_weights[held_mask] = held_weights
        routed_confidence[held_mask] = held_confidence_matrix
        calibrated_probability[held_mask] = held_probability_matrix
        fold_states.append(
            {
                "held_fold": held_fold,
                "temperatures": temperatures,
                "reliability": reliability_states,
                "router": router.state(),
            }
        )

    full_temperatures = []
    full_reliability = []
    full_probability_columns = []
    full_confidence_columns = []
    for route_index in range(3):
        temperature = _fit_temperature(raw_probability[:, route_index], labels)
        probability = _apply_temperature(raw_probability[:, route_index], temperature)
        reliability = _fit_reliability(
            features[route_index], probability, labels, seed=seed + 100 + route_index
        )
        full_temperatures.append(temperature)
        full_reliability.append(reliability.state())
        full_probability_columns.append(probability)
        full_confidence_columns.append(reliability.predict(features[route_index]))
    full_router = _fit_router(
        torch.stack(full_probability_columns, dim=1),
        torch.stack(full_confidence_columns, dim=1),
        labels,
    )

    baselines = {
        name: _evaluate(calibrated_probability[:, index], labels) for index, name in enumerate(ROUTE_NAMES)
    }
    baselines["uniform_mean"] = _evaluate(calibrated_probability.mean(dim=1), labels)
    routed_evaluation = _evaluate(routed_probability, labels)
    overall_confidence = (routed_weights * routed_confidence).sum(dim=1)
    fold_comparison = []
    winning_folds = 0
    for fold in sorted(folds.unique().tolist()):
        selected = folds == fold
        routed_fold = binary_metrics(_logit(routed_probability[selected]), labels[selected], threshold=0.5)
        branch_folds = {
            name: binary_metrics(
                _logit(calibrated_probability[selected, index]), labels[selected], threshold=0.5
            )
            for index, name in enumerate(ROUTE_NAMES)
        }
        strongest_fold_mcc = max(metrics["mcc"] for metrics in branch_folds.values())
        won = routed_fold["mcc"] > strongest_fold_mcc
        winning_folds += int(won)
        fold_comparison.append(
            {
                "fold": fold,
                "dynamic_router": routed_fold,
                "branches": branch_folds,
                "beats_strongest_branch": won,
            }
        )
    strongest_name = max(
        ROUTE_NAMES, key=lambda name: baselines[name]["at_oof_mcc_threshold"]["mcc"]
    )
    strongest = baselines[strongest_name]
    mcc_gain = (
        routed_evaluation["at_oof_mcc_threshold"]["mcc"]
        - strongest["at_oof_mcc_threshold"]["mcc"]
    )
    auprc_delta = routed_evaluation["at_0.5"]["auprc"] - strongest["at_0.5"]["auprc"]
    brier_delta = routed_evaluation["calibration"]["brier"] - strongest["calibration"]["brier"]
    uniform = baselines["uniform_mean"]
    uniform_mcc_delta = (
        routed_evaluation["at_oof_mcc_threshold"]["mcc"]
        - uniform["at_oof_mcc_threshold"]["mcc"]
    )
    uniform_auprc_delta = routed_evaluation["at_0.5"]["auprc"] - uniform["at_0.5"]["auprc"]
    uniform_brier_delta = routed_evaluation["calibration"]["brier"] - uniform["calibration"]["brier"]
    uniform_ece_delta = routed_evaluation["calibration"]["ece_10"] - uniform["calibration"]["ece_10"]
    adoption_checks = {
        "mcc_gain_at_least_0.005": mcc_gain >= 0.005,
        "auprc_drop_no_more_than_0.003": auprc_delta >= -0.003,
        "brier_not_worse": brier_delta <= 0.0,
        "wins_at_least_3_of_5_folds": winning_folds >= 3,
        "mcc_within_0.001_of_uniform_mean": uniform_mcc_delta >= -0.001,
        "auprc_within_0.003_of_uniform_mean": uniform_auprc_delta >= -0.003,
        "brier_better_than_uniform_mean": uniform_brier_delta < 0.0,
        "ece_better_than_uniform_mean": uniform_ece_delta < 0.0,
    }
    summary = {
        "protocol": {
            "route_files": [str(path) for path in route_paths],
            "num_samples": len(labels),
            "meta_folds": sorted(folds.unique().tolist()),
            "selection_data": "fixed group-aware OOF predictions only",
            "test_labels_used": False,
            "confidence_target": "probability assigned to the true label",
        },
        "baselines": baselines,
        "cross_fitted_dynamic_router": routed_evaluation,
        "fold_comparison_at_0.5": fold_comparison,
        "mean_weights": {
            name: float(routed_weights[:, index].mean()) for index, name in enumerate(ROUTE_NAMES)
        },
        "mean_route_confidence": {
            name: float(routed_confidence[:, index].mean()) for index, name in enumerate(ROUTE_NAMES)
        },
        "mean_overall_confidence": float(overall_confidence.mean()),
        "selective_performance": _selective_metrics(
            routed_probability, labels, overall_confidence
        ),
        "fold_states": fold_states,
        "frozen_full_oof_fit": {
            "route_names": list(ROUTE_NAMES),
            "feature_names": [list(BASE_FEATURES), list(BASE_FEATURES), list(THREE_D_FEATURES)],
            "temperatures": full_temperatures,
            "reliability": full_reliability,
            "router": full_router.state(),
            "decision_threshold": routed_evaluation["oof_mcc_threshold"],
        },
        "oof_screen_decision": {
            "strongest_single_route": strongest_name,
            "mcc_gain": mcc_gain,
            "auprc_delta": auprc_delta,
            "brier_delta": brier_delta,
            "uniform_mcc_delta": uniform_mcc_delta,
            "uniform_auprc_delta": uniform_auprc_delta,
            "uniform_brier_delta": uniform_brier_delta,
            "uniform_ece_delta": uniform_ece_delta,
            "winning_folds": winning_folds,
            "checks": adoption_checks,
            "advance_to_frozen_external_evaluation": all(adoption_checks.values()),
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(routes[0][0]) + [
        "calibrated_probability_1d", "calibrated_probability_2d", "calibrated_probability_3d",
        "confidence_1d", "confidence_2d", "confidence_3d",
        "weight_1d", "weight_2d", "weight_3d", "routed_probability", "overall_confidence",
    ]
    with (output_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, base_row in enumerate(routes[0]):
            row = dict(base_row)
            row.update({
                "calibrated_probability_1d": f"{float(calibrated_probability[index, 0]):.9g}",
                "calibrated_probability_2d": f"{float(calibrated_probability[index, 1]):.9g}",
                "calibrated_probability_3d": f"{float(calibrated_probability[index, 2]):.9g}",
                "confidence_1d": f"{float(routed_confidence[index, 0]):.9g}",
                "confidence_2d": f"{float(routed_confidence[index, 1]):.9g}",
                "confidence_3d": f"{float(routed_confidence[index, 2]):.9g}",
                "weight_1d": f"{float(routed_weights[index, 0]):.9g}",
                "weight_2d": f"{float(routed_weights[index, 1]):.9g}",
                "weight_3d": f"{float(routed_weights[index, 2]):.9g}",
                "routed_probability": f"{float(routed_probability[index]):.9g}",
                "overall_confidence": f"{float(overall_confidence[index]):.9g}",
            })
            writer.writerow(row)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit a leakage-controlled confidence router over three OOF experts.")
    parser.add_argument("--one-dimensional", required=True)
    parser.add_argument("--two-dimensional", required=True)
    parser.add_argument("--three-dimensional", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = fit_cross_fitted_router(
        [args.one_dimensional, args.two_dimensional, args.three_dimensional],
        args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
