"""Paper-oriented validation for the frozen 3D-v2 GHXTox model.

This module does not train or tune a model.  It compares the frozen 3D-v2
default against the frozen 3D-v1 fallback, reports paired uncertainty,
performs biologically relevant subgroup analyses, explains representative
cases by chemical-site deletion, and benchmarks the cached-feature classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ghxtox.biological_interpretation import _repeat_single_batch
from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_NAMES
from ghxtox.data import PeptideTensorDataset, collate_peptides
from ghxtox.features import sequence_global_features
from ghxtox.models import GHXToxModel
from ghxtox.utils import move_batch_to_device, resolve_device, save_json


METRIC_NAMES = (
    "balanced_accuracy",
    "f1",
    "mcc",
    "auroc",
    "auprc",
    "brier",
)


def _read_prediction_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    required = {"sample_id", "label", "toxicity_probability"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Prediction file {path} is missing columns: {sorted(missing)}")
    return rows


def _align_predictions(
    v1_path: str | Path,
    v2_path: str | Path,
    v1_threshold: float,
    v2_threshold: float,
) -> dict[str, Any]:
    v1_rows = _read_prediction_csv(v1_path)
    v2_rows = _read_prediction_csv(v2_path)
    v1_lookup = {str(row["sample_id"]): row for row in v1_rows}
    v2_lookup = {str(row["sample_id"]): row for row in v2_rows}
    if set(v1_lookup) != set(v2_lookup):
        only_v1 = sorted(set(v1_lookup) - set(v2_lookup))
        only_v2 = sorted(set(v2_lookup) - set(v1_lookup))
        raise ValueError(
            f"Prediction sample IDs do not match; only_v1={only_v1[:3]}, only_v2={only_v2[:3]}"
        )
    sample_ids = [str(row["sample_id"]) for row in v2_rows]
    labels = np.asarray([int(v2_lookup[sample_id]["label"]) for sample_id in sample_ids])
    v1_labels = np.asarray([int(v1_lookup[sample_id]["label"]) for sample_id in sample_ids])
    if not np.array_equal(labels, v1_labels):
        raise ValueError("Prediction labels do not match after sample-ID alignment.")
    v1_scores = np.asarray(
        [float(v1_lookup[sample_id]["toxicity_probability"]) for sample_id in sample_ids],
        dtype=np.float64,
    )
    v2_scores = np.asarray(
        [float(v2_lookup[sample_id]["toxicity_probability"]) for sample_id in sample_ids],
        dtype=np.float64,
    )
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "v1_scores": v1_scores,
        "v2_scores": v2_scores,
        "v1_predictions": v1_scores >= float(v1_threshold),
        "v2_predictions": v2_scores >= float(v2_threshold),
        "v1_rows": v1_lookup,
        "v2_rows": v2_lookup,
    }


def _metric_values(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = labels.astype(np.int64, copy=False)
    predictions = predictions.astype(bool, copy=False)
    positives = labels == 1
    negatives = ~positives
    tp = float(np.sum(predictions & positives))
    tn = float(np.sum(~predictions & negatives))
    fp = float(np.sum(predictions & negatives))
    fn = float(np.sum(~predictions & positives))
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    precision = tp / max(tp + fp, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    values = {
        "balanced_accuracy": 0.5 * (recall + specificity),
        "f1": f1,
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": float("nan"),
        "auprc": float("nan"),
        "brier": float(np.mean((scores - labels) ** 2)),
    }
    if np.unique(labels).size == 2:
        values["auroc"] = float(roc_auc_score(labels, scores))
        values["auprc"] = float(average_precision_score(labels, scores))
    return values


def _stratified_sample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(group.size == 0 for group in groups):
        raise ValueError("Both classes are required for stratified resampling.")
    return np.concatenate(
        [rng.choice(group, size=group.size, replace=True) for group in groups]
    )


def paired_model_comparison(
    labels: np.ndarray,
    v1_scores: np.ndarray,
    v2_scores: np.ndarray,
    v1_predictions: np.ndarray,
    v2_predictions: np.ndarray,
    bootstrap_iterations: int = 5000,
    permutation_iterations: int = 5000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Return paired bootstrap CIs and paired randomization p-values."""

    labels = np.asarray(labels, dtype=np.int64)
    v1_scores = np.asarray(v1_scores, dtype=np.float64)
    v2_scores = np.asarray(v2_scores, dtype=np.float64)
    v1_predictions = np.asarray(v1_predictions, dtype=bool)
    v2_predictions = np.asarray(v2_predictions, dtype=bool)
    lengths = {
        labels.size,
        v1_scores.size,
        v2_scores.size,
        v1_predictions.size,
        v2_predictions.size,
    }
    if len(lengths) != 1 or not labels.size:
        raise ValueError("Paired comparison arrays must be non-empty and aligned.")
    if np.unique(labels).size != 2:
        raise ValueError("Paired comparison requires both classes.")
    if bootstrap_iterations < 100 or permutation_iterations < 100:
        raise ValueError("Bootstrap and permutation iterations must both be at least 100.")

    point_v1 = _metric_values(labels, v1_scores, v1_predictions)
    point_v2 = _metric_values(labels, v2_scores, v2_predictions)
    point_delta = {name: point_v2[name] - point_v1[name] for name in METRIC_NAMES}

    rng = np.random.default_rng(seed)
    bootstrap_deltas = {name: np.empty(bootstrap_iterations) for name in METRIC_NAMES}
    for iteration in range(bootstrap_iterations):
        indices = _stratified_sample_indices(labels, rng)
        first = _metric_values(
            labels[indices], v1_scores[indices], v1_predictions[indices]
        )
        second = _metric_values(
            labels[indices], v2_scores[indices], v2_predictions[indices]
        )
        for name in METRIC_NAMES:
            bootstrap_deltas[name][iteration] = second[name] - first[name]

    null_deltas = {name: np.empty(permutation_iterations) for name in METRIC_NAMES}
    for iteration in range(permutation_iterations):
        swap = rng.random(labels.size) < 0.5
        perm_v1_scores = np.where(swap, v2_scores, v1_scores)
        perm_v2_scores = np.where(swap, v1_scores, v2_scores)
        perm_v1_predictions = np.where(swap, v2_predictions, v1_predictions)
        perm_v2_predictions = np.where(swap, v1_predictions, v2_predictions)
        first = _metric_values(labels, perm_v1_scores, perm_v1_predictions)
        second = _metric_values(labels, perm_v2_scores, perm_v2_predictions)
        for name in METRIC_NAMES:
            null_deltas[name][iteration] = second[name] - first[name]

    metrics: dict[str, Any] = {}
    for name in METRIC_NAMES:
        deltas = bootstrap_deltas[name]
        null = null_deltas[name]
        observed = point_delta[name]
        metrics[name] = {
            "v1": point_v1[name],
            "v2": point_v2[name],
            "delta_v2_minus_v1": observed,
            "paired_bootstrap_95_ci": {
                "lower": float(np.quantile(deltas, 0.025)),
                "upper": float(np.quantile(deltas, 0.975)),
            },
            "bootstrap_probability_v2_better": float(np.mean(deltas > 0.0)),
            "paired_randomization_two_sided_p": float(
                (np.sum(np.abs(null) >= abs(observed)) + 1)
                / (permutation_iterations + 1)
            ),
        }
    return {
        "protocol": {
            "paired_samples": int(labels.size),
            "num_positive": int(labels.sum()),
            "num_negative": int((labels == 0).sum()),
            "bootstrap": "class-stratified paired percentile bootstrap",
            "bootstrap_iterations": int(bootstrap_iterations),
            "randomization": "within-sample exchange of the two frozen model outputs",
            "permutation_iterations": int(permutation_iterations),
            "seed": int(seed),
            "interpretation": (
                "Existing Test1/Test2 labels were previously viewed; p-values are descriptive "
                "retrospective evidence, not independent confirmatory inference."
            ),
        },
        "metrics": metrics,
    }


def _bootstrap_subset_delta(
    labels: np.ndarray,
    v1_scores: np.ndarray,
    v2_scores: np.ndarray,
    v1_predictions: np.ndarray,
    v2_predictions: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    names = ("mcc", "auprc")
    samples = {name: np.empty(iterations) for name in names}
    for iteration in range(iterations):
        indices = _stratified_sample_indices(labels, rng)
        first = _metric_values(
            labels[indices], v1_scores[indices], v1_predictions[indices]
        )
        second = _metric_values(
            labels[indices], v2_scores[indices], v2_predictions[indices]
        )
        for name in names:
            samples[name][iteration] = second[name] - first[name]
    return {
        name: {
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def _record_characteristics(record: dict[str, Any]) -> dict[str, float]:
    sequence = str(record["sequence"])
    length = len(sequence)
    plddt = record["plddt"][:length].float()
    site_mask = record.get("chemical_site_mask")
    if torch.is_tensor(site_mask):
        site_coverage = float(site_mask[:length].any(dim=-1).float().mean())
    else:
        site_coverage = 0.0
    properties = sequence_global_features(sequence)
    return {
        **properties,
        "mean_plddt": float(plddt.mean()),
        "min_plddt": float(plddt.min()),
        "low_plddt_fraction": float((plddt < 0.55).float().mean()),
        "cysteine_count": float(sequence.count("C")),
        "chemical_site_coverage": site_coverage,
    }


def _assign_strata(characteristics: dict[str, float]) -> list[tuple[str, str]]:
    length = characteristics["length"]
    mean_plddt = characteristics["mean_plddt"]
    charge = characteristics["net_charge"]
    hydropathy = characteristics["mean_hydropathy"]
    cysteine_count = characteristics["cysteine_count"]
    site_coverage = characteristics["chemical_site_coverage"]
    return [
        (
            "length",
            "<=20" if length <= 20 else "21-30" if length <= 30 else "31-50" if length <= 50 else ">50",
        ),
        (
            "mean_plddt",
            "<0.55" if mean_plddt < 0.55 else "0.55-0.70" if mean_plddt < 0.70 else ">=0.70",
        ),
        (
            "net_charge",
            "negative" if charge < 0 else "neutral" if charge == 0 else "positive",
        ),
        (
            "mean_hydropathy",
            "<0" if hydropathy < 0 else "0-1" if hydropathy < 1.0 else ">=1",
        ),
        (
            "cysteine_count",
            "0" if cysteine_count == 0 else "1-2" if cysteine_count <= 2 else ">=3",
        ),
        (
            "chemical_site_coverage",
            "<0.80" if site_coverage < 0.80 else "0.80-0.95" if site_coverage < 0.95 else ">=0.95",
        ),
    ]


def stratified_comparison(
    aligned: dict[str, Any],
    records: list[dict[str, Any]],
    dataset_name: str,
    bootstrap_iterations: int,
    seed: int,
    minimum_samples: int = 20,
    minimum_per_class: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    record_lookup = {str(record["sample_id"]): record for record in records}
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    missing_records = []
    for index, sample_id in enumerate(aligned["sample_ids"]):
        record = record_lookup.get(sample_id)
        if record is None:
            missing_records.append(sample_id)
            continue
        for key in _assign_strata(_record_characteristics(record)):
            strata[key].append(index)
    if missing_records:
        raise ValueError(f"Missing processed records for samples: {missing_records[:3]}")

    rows: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    labels_all = aligned["labels"]
    for stratum_index, ((variable, bin_name), index_list) in enumerate(sorted(strata.items())):
        indices = np.asarray(index_list, dtype=np.int64)
        labels = labels_all[indices]
        counts = Counter(labels.tolist())
        if (
            indices.size < minimum_samples
            or counts[0] < minimum_per_class
            or counts[1] < minimum_per_class
        ):
            omitted.append(
                {
                    "dataset": dataset_name,
                    "variable": variable,
                    "bin": bin_name,
                    "num_samples": int(indices.size),
                    "num_positive": int(counts[1]),
                    "num_negative": int(counts[0]),
                    "reason": "insufficient total or per-class sample count",
                }
            )
            continue
        v1_scores = aligned["v1_scores"][indices]
        v2_scores = aligned["v2_scores"][indices]
        v1_predictions = aligned["v1_predictions"][indices]
        v2_predictions = aligned["v2_predictions"][indices]
        first = _metric_values(labels, v1_scores, v1_predictions)
        second = _metric_values(labels, v2_scores, v2_predictions)
        intervals = _bootstrap_subset_delta(
            labels,
            v1_scores,
            v2_scores,
            v1_predictions,
            v2_predictions,
            iterations=bootstrap_iterations,
            seed=seed + stratum_index,
        )
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "variable": variable,
            "bin": bin_name,
            "num_samples": int(indices.size),
            "num_positive": int(counts[1]),
            "num_negative": int(counts[0]),
        }
        for name in METRIC_NAMES:
            row[f"v1_{name}"] = first[name]
            row[f"v2_{name}"] = second[name]
            row[f"delta_{name}"] = second[name] - first[name]
        row["delta_mcc_ci_lower"] = intervals["mcc"]["lower"]
        row["delta_mcc_ci_upper"] = intervals["mcc"]["upper"]
        row["delta_auprc_ci_lower"] = intervals["auprc"]["lower"]
        row["delta_auprc_ci_upper"] = intervals["auprc"]["upper"]
        rows.append(row)
    return rows, omitted


def _case_group(label: int, prediction: int) -> str:
    if label == 1:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def select_representative_cases(
    rows: Iterable[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Select the most confident and nearest-threshold sample per confusion group."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        probability = float(row["toxicity_probability"])
        label = int(row["label"])
        prediction = int(probability >= threshold)
        normalized = dict(row)
        normalized["toxicity_probability"] = probability
        normalized["prediction"] = prediction
        normalized["case_group"] = _case_group(label, prediction)
        groups[normalized["case_group"]].append(normalized)

    selected: list[dict[str, Any]] = []
    for group_name in ("TP", "TN", "FP", "FN"):
        group = groups.get(group_name, [])
        if not group:
            continue
        if group_name in {"TP", "FP"}:
            extreme = max(group, key=lambda row: float(row["toxicity_probability"]))
        else:
            extreme = min(group, key=lambda row: float(row["toxicity_probability"]))
        boundary = min(
            group,
            key=lambda row: abs(float(row["toxicity_probability"]) - threshold),
        )
        for role, row in (("confidence_extreme", extreme), ("threshold_boundary", boundary)):
            if any(
                existing["sample_id"] == row["sample_id"]
                and existing["case_group"] == group_name
                for existing in selected
            ):
                continue
            selected.append({**row, "selection_role": role})
    return selected


def _chemical_type_text(types: torch.Tensor, mask: torch.Tensor) -> str:
    active = types[mask].amax(dim=0) if bool(mask.any()) else torch.zeros(types.shape[-1])
    return ";".join(
        name
        for index, name in enumerate(CHEMICAL_SITE_TYPE_NAMES)
        if float(active[index]) > 0.0
    )


def _chemical_case_attribution(
    model: GHXToxModel,
    record: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    batch = collate_peptides([record], include_structure=True, include_atom=False)
    batch = move_batch_to_device(batch, device)
    length = len(str(record["sequence"]))
    with torch.inference_mode():
        full_output = model(batch)
        full_logit = float(full_output["logits"][0].detach().cpu())
        full_probability = float(torch.sigmoid(full_output["logits"][0]).detach().cpu())

        no_site_batch = _repeat_single_batch(batch, 1)
        no_site_batch["chemical_site_mask"].zero_()
        no_site_output = model(no_site_batch)
        no_site_logit = float(no_site_output["logits"][0].detach().cpu())
        no_site_probability = float(torch.sigmoid(no_site_output["logits"][0]).detach().cpu())

        valid_positions = (
            batch["chemical_site_mask"][0, :length].any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
        )
        residue_rows: list[dict[str, Any]] = []
        if valid_positions.numel() > 0:
            residue_batch = _repeat_single_batch(batch, int(valid_positions.numel()))
            for row_index, position in enumerate(valid_positions.tolist()):
                residue_batch["chemical_site_mask"][row_index, position] = False
            residue_logits = model(residue_batch)["logits"].detach().cpu()
            for position, ablated_logit in zip(valid_positions.tolist(), residue_logits.tolist()):
                position_mask = batch["chemical_site_mask"][0, position].detach().cpu().bool()
                position_types = batch["chemical_site_types"][0, position].detach().cpu()
                residue_rows.append(
                    {
                        "position": position + 1,
                        "residue": str(record["sequence"])[position],
                        "plddt": float(batch["plddt"][0, position].detach().cpu()),
                        "site_count": int(position_mask.sum()),
                        "site_types": _chemical_type_text(position_types, position_mask),
                        "delta_logit_full_minus_residue_site_ablation": full_logit
                        - float(ablated_logit),
                    }
                )

        type_rows: list[dict[str, Any]] = []
        type_batch = _repeat_single_batch(batch, len(CHEMICAL_SITE_TYPE_NAMES))
        for type_index, type_name in enumerate(CHEMICAL_SITE_TYPE_NAMES):
            carries_type = type_batch["chemical_site_types"][
                type_index, :, :, type_index
            ] > 0.0
            type_batch["chemical_site_mask"][type_index] &= ~carries_type
        type_logits = model(type_batch)["logits"].detach().cpu()
        original_mask = batch["chemical_site_mask"][0]
        original_types = batch["chemical_site_types"][0]
        for type_index, (type_name, ablated_logit) in enumerate(
            zip(CHEMICAL_SITE_TYPE_NAMES, type_logits.tolist())
        ):
            affected = original_mask & (original_types[:, :, type_index] > 0.0)
            type_rows.append(
                {
                    "site_type": type_name,
                    "affected_site_count": int(affected.sum().detach().cpu()),
                    "delta_logit_full_minus_type_ablation": full_logit - float(ablated_logit),
                }
            )

    site_mask = batch["chemical_site_mask"][0, :length]
    sample_row = {
        "full_logit": full_logit,
        "full_probability": full_probability,
        "no_chemical_site_logit": no_site_logit,
        "no_chemical_site_probability": no_site_probability,
        "chemical_site_delta_logit": full_logit - no_site_logit,
        "chemical_site_delta_probability": full_probability - no_site_probability,
        "chemical_site_count": int(site_mask.sum().detach().cpu()),
        "chemical_site_residue_coverage": float(
            site_mask.any(dim=-1).float().mean().detach().cpu()
        ),
        "chemical_edge_count": float(
            full_output["chemical_edge_count"][0].detach().cpu()
        ),
        "chemical_residual_norm": float(
            full_output["chemical_residual_norm"][0].detach().cpu()
        ),
        "mean_plddt": float(batch["plddt"][0, :length].mean().detach().cpu()),
    }
    return sample_row, residue_rows, type_rows


def explain_representative_cases(
    dataset_name: str,
    processed_path: str | Path,
    aligned: dict[str, Any],
    checkpoint_path: str | Path,
    threshold: float,
    device: torch.device,
    excluded_sequences: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    record_lookup = {str(record["sample_id"]): record for record in dataset.records}
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GHXToxModel(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    excluded_sequences = excluded_sequences or set()
    candidate_rows = [
        row
        for row in aligned["v2_rows"].values()
        if str(row.get("sequence", "")) not in excluded_sequences
    ]
    selected = select_representative_cases(candidate_rows, threshold)

    sample_rows: list[dict[str, Any]] = []
    residue_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    for selection in selected:
        sample_id = str(selection["sample_id"])
        record = record_lookup[sample_id]
        sample_values, residue_values, type_values = _chemical_case_attribution(
            model, record, device
        )
        v1_probability = float(aligned["v1_rows"][sample_id]["toxicity_probability"])
        prefix = {
            "dataset": dataset_name,
            "sample_id": sample_id,
            "sequence": str(record["sequence"]),
            "label": int(record["label"]),
            "case_group": selection["case_group"],
            "selection_role": selection["selection_role"],
            "v1_probability": v1_probability,
            "v2_probability": float(selection["toxicity_probability"]),
            "v2_minus_v1_probability": float(selection["toxicity_probability"])
            - v1_probability,
        }
        sample_rows.append({**prefix, **sample_values})
        residue_rows.extend({**prefix, **row} for row in residue_values)
        type_rows.extend({**prefix, **row} for row in type_values)
    return sample_rows, residue_rows, type_rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_case_attributions(
    sample_rows: list[dict[str, Any]],
    residue_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    residue_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    type_lookup: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in residue_rows:
        residue_lookup[(str(row["dataset"]), str(row["sample_id"]))].append(row)
    for row in type_rows:
        type_lookup[(str(row["dataset"]), str(row["sample_id"]))].append(row)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for sample in sample_rows:
        key = (str(sample["dataset"]), str(sample["sample_id"]))
        residues = residue_lookup[key]
        types = type_lookup[key]
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
        axes[0].bar(
            [f"{row['residue']}{row['position']}" for row in residues],
            [float(row["delta_logit_full_minus_residue_site_ablation"]) for row in residues],
            color=[
                "#c44e52"
                if float(row["delta_logit_full_minus_residue_site_ablation"]) >= 0
                else "#4c72b0"
                for row in residues
            ],
        )
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set_ylabel("Full − residue-site ablated logit")
        axes[0].tick_params(axis="x", labelrotation=70)
        axes[0].set_title(
            f"{sample['dataset']} {sample['sample_id']} "
            f"{sample['case_group']} ({sample['selection_role']})"
        )
        axes[1].bar(
            [str(row["site_type"]) for row in types],
            [float(row["delta_logit_full_minus_type_ablation"]) for row in types],
            color="#55a868",
        )
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Full − site-type ablated logit")
        axes[1].tick_params(axis="x", labelrotation=35)
        safe_id = str(sample["sample_id"]).replace("/", "_").replace("\\", "_")
        figure.savefig(figure_dir / f"{sample['dataset']}_{safe_id}.png", dpi=180)
        plt.close(figure)


def _plot_validation_summary(
    paired_results: dict[str, Any],
    stratified_rows: list[dict[str, Any]],
    efficiency: dict[str, Any],
    output_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    datasets = ("oof", "test1", "test2")
    colors = {"oof": "#4c72b0", "test1": "#55a868", "test2": "#c44e52"}
    metric_names = ("mcc", "auroc", "auprc")
    x = np.arange(len(metric_names))
    width = 0.23
    for dataset_offset, dataset_name in enumerate(datasets):
        values = [
            paired_results[dataset_name]["metrics"][name]["delta_v2_minus_v1"]
            for name in metric_names
        ]
        axes[0].bar(
            x + (dataset_offset - 1) * width,
            values,
            width=width,
            label=dataset_name.upper(),
            color=colors[dataset_name],
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xticks(x, [name.upper() for name in metric_names])
    axes[0].set_ylabel("3D-v2 − 3D-v1")
    axes[0].set_title("Paired performance differences")
    axes[0].legend(frameon=False)

    ranked = sorted(
        stratified_rows,
        key=lambda row: abs(float(row["delta_auprc"])),
        reverse=True,
    )[:10]
    labels = [
        f"{row['dataset']}:{row['variable']}:{row['bin']}"
        for row in reversed(ranked)
    ]
    values = [float(row["delta_auprc"]) for row in reversed(ranked)]
    axes[1].barh(
        labels,
        values,
        color=["#55a868" if value >= 0 else "#c44e52" for value in values],
    )
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("AUPRC difference")
    axes[1].set_title("Largest retrospective subgroup shifts")

    parameter_values = [
        efficiency["v1"]["total_parameters"] / 1e6,
        efficiency["v2"]["total_parameters"] / 1e6,
    ]
    latency_values = [
        efficiency["v1"]["median_milliseconds_per_peptide"],
        efficiency["v2"]["median_milliseconds_per_peptide"],
    ]
    position = np.arange(2)
    axes[2].bar(
        position - 0.16,
        parameter_values,
        width=0.32,
        label="Parameters (million)",
        color="#8172b2",
    )
    latency_axis = axes[2].twinx()
    latency_axis.bar(
        position + 0.16,
        latency_values,
        width=0.32,
        label="CPU ms/peptide",
        color="#ccb974",
    )
    axes[2].set_xticks(position, ["3D-v1", "3D-v2"])
    axes[2].set_ylabel("Parameters (million)")
    latency_axis.set_ylabel("Cached CPU ms/peptide")
    axes[2].set_title("Incremental model cost")
    handles_left, labels_left = axes[2].get_legend_handles_labels()
    handles_right, labels_right = latency_axis.get_legend_handles_labels()
    axes[2].legend(
        handles_left + handles_right,
        labels_left + labels_right,
        frameon=False,
        loc="upper left",
    )
    figure.savefig(output_dir / "paper_validation_summary.png", dpi=200)
    plt.close(figure)


def _balanced_benchmark_records(
    records: list[dict[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[int(record["label"])].append(record)
    per_class = max(maximum // 2, 1)
    selected = by_label[0][:per_class] + by_label[1][:per_class]
    return selected[:maximum]


def _benchmark_checkpoint(
    checkpoint_path: str | Path,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(
        records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_peptides, include_structure=True, include_atom=False),
    )

    def run_once() -> None:
        with torch.inference_mode():
            for batch in loader:
                model(move_batch_to_device(batch, device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup_repeats):
        run_once()
    durations = []
    for _ in range(measured_repeats):
        start = time.perf_counter()
        run_once()
        durations.append(time.perf_counter() - start)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    chemical_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("chemical_site_branch.")
    )
    median_seconds = float(np.median(durations))
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "total_parameters": int(total_parameters),
        "chemical_site_branch_parameters": int(chemical_parameters),
        "num_samples": len(records),
        "batch_size": int(batch_size),
        "warmup_repeats": int(warmup_repeats),
        "measured_repeats": int(measured_repeats),
        "seconds_per_repeat": durations,
        "median_seconds_per_repeat": median_seconds,
        "median_milliseconds_per_peptide": 1000.0 * median_seconds / max(len(records), 1),
        "throughput_peptides_per_second": len(records) / max(median_seconds, 1e-12),
    }


def benchmark_models(
    v1_checkpoint: str | Path,
    v2_checkpoint: str | Path,
    processed_path: str | Path,
    device: torch.device,
    maximum_samples: int,
    batch_size: int,
    warmup_repeats: int,
    measured_repeats: int,
) -> dict[str, Any]:
    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    records = _balanced_benchmark_records(dataset.records, maximum_samples)
    v1 = _benchmark_checkpoint(
        v1_checkpoint,
        records,
        device,
        batch_size,
        warmup_repeats,
        measured_repeats,
    )
    v2 = _benchmark_checkpoint(
        v2_checkpoint,
        records,
        device,
        batch_size,
        warmup_repeats,
        measured_repeats,
    )
    return {
        "scope": (
            "Cached-feature classifier inference only. ESM2 embedding generation and ESMFold "
            "structure prediction are excluded."
        ),
        "device": str(device),
        "torch_num_threads": int(torch.get_num_threads()),
        "v1": v1,
        "v2": v2,
        "v2_minus_v1_parameters": v2["total_parameters"] - v1["total_parameters"],
        "latency_ratio_v2_over_v1": (
            v2["median_milliseconds_per_peptide"]
            / max(v1["median_milliseconds_per_peptide"], 1e-12)
        ),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    datasets = {
        "test1": {
            "v1_predictions": args.test1_v1_predictions,
            "v2_predictions": args.test1_v2_predictions,
            "processed": args.test1_processed,
        },
        "test2": {
            "v1_predictions": args.test2_v1_predictions,
            "v2_predictions": args.test2_v2_predictions,
            "processed": args.test2_processed,
        },
        "oof": {
            "v1_predictions": args.oof_v1_predictions,
            "v2_predictions": args.oof_v2_predictions,
            "processed": None,
        },
    }
    thresholds = {
        "test1": (args.v1_test_threshold, args.v2_threshold),
        "test2": (args.v1_test_threshold, args.v2_threshold),
        "oof": (args.v1_oof_threshold, args.v2_threshold),
    }

    aligned_by_dataset: dict[str, dict[str, Any]] = {}
    paired_results: dict[str, Any] = {}
    for dataset_index, (dataset_name, paths) in enumerate(datasets.items()):
        v1_threshold, v2_threshold = thresholds[dataset_name]
        aligned = _align_predictions(
            paths["v1_predictions"],
            paths["v2_predictions"],
            v1_threshold,
            v2_threshold,
        )
        aligned_by_dataset[dataset_name] = aligned
        result = paired_model_comparison(
            aligned["labels"],
            aligned["v1_scores"],
            aligned["v2_scores"],
            aligned["v1_predictions"],
            aligned["v2_predictions"],
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
            seed=args.seed + dataset_index,
        )
        if dataset_name == "oof":
            result["protocol"]["interpretation"] = (
                "Training-only group-aware out-of-fold comparison. It supports internal "
                "model-selection evidence but is not an independent external validation."
            )
        result["thresholds"] = {"v1": v1_threshold, "v2": v2_threshold}
        result["prediction_files"] = {
            "v1": paths["v1_predictions"],
            "v2": paths["v2_predictions"],
        }
        paired_results[dataset_name] = result
    save_json(paired_results, output_dir / "paired_comparison.json")

    stratified_rows: list[dict[str, Any]] = []
    omitted_rows: list[dict[str, Any]] = []
    for dataset_index, dataset_name in enumerate(("test1", "test2")):
        dataset = PeptideTensorDataset(
            datasets[dataset_name]["processed"], require_labels=True
        )
        rows, omitted = stratified_comparison(
            aligned_by_dataset[dataset_name],
            dataset.records,
            dataset_name,
            bootstrap_iterations=args.stratum_bootstrap_iterations,
            seed=args.seed + 100 + dataset_index * 100,
            minimum_samples=args.minimum_stratum_samples,
            minimum_per_class=args.minimum_stratum_per_class,
        )
        stratified_rows.extend(rows)
        omitted_rows.extend(omitted)
    _write_csv(output_dir / "stratified_metrics.csv", stratified_rows)
    save_json(
        {
            "protocol": {
                "variables": [
                    "length",
                    "mean_plddt",
                    "net_charge",
                    "mean_hydropathy",
                    "cysteine_count",
                    "chemical_site_coverage",
                ],
                "minimum_samples": args.minimum_stratum_samples,
                "minimum_per_class": args.minimum_stratum_per_class,
                "paired_bootstrap_iterations": args.stratum_bootstrap_iterations,
                "note": "Subgroups are descriptive and were not used for model selection.",
            },
            "num_reported_strata": len(stratified_rows),
            "omitted_strata": omitted_rows,
        },
        output_dir / "stratified_summary.json",
    )

    case_samples: list[dict[str, Any]] = []
    case_residues: list[dict[str, Any]] = []
    case_types: list[dict[str, Any]] = []
    selected_sequences: set[str] = set()
    for dataset_name in ("test1", "test2"):
        sample_rows, residue_rows, type_rows = explain_representative_cases(
            dataset_name,
            datasets[dataset_name]["processed"],
            aligned_by_dataset[dataset_name],
            args.v2_checkpoint,
            args.v2_threshold,
            device,
            excluded_sequences=selected_sequences,
        )
        case_samples.extend(sample_rows)
        case_residues.extend(residue_rows)
        case_types.extend(type_rows)
        selected_sequences.update(str(row["sequence"]) for row in sample_rows)
    _write_csv(output_dir / "case_sample_summary.csv", case_samples)
    _write_csv(output_dir / "case_residue_chemical_attribution.csv", case_residues)
    _write_csv(output_dir / "case_site_type_attribution.csv", case_types)
    _plot_case_attributions(case_samples, case_residues, case_types, output_dir)

    efficiency = benchmark_models(
        args.v1_checkpoint,
        args.v2_checkpoint,
        args.test1_processed,
        device,
        maximum_samples=args.benchmark_samples,
        batch_size=args.benchmark_batch_size,
        warmup_repeats=args.benchmark_warmup,
        measured_repeats=args.benchmark_repeats,
    )
    save_json(efficiency, output_dir / "efficiency.json")
    _plot_validation_summary(paired_results, stratified_rows, efficiency, output_dir)

    selected_ids = defaultdict(list)
    for row in case_samples:
        selected_ids[str(row["dataset"])].append(str(row["sample_id"]))
    summary = {
        "status": "completed_without_new_external_dataset",
        "frozen_models": {
            "v1": args.v1_checkpoint,
            "v2": args.v2_checkpoint,
        },
        "paired_comparison": "paired_comparison.json",
        "stratified_analysis": "stratified_metrics.csv",
        "chemical_case_analysis": {
            "sample_summary": "case_sample_summary.csv",
            "residue_attribution": "case_residue_chemical_attribution.csv",
            "site_type_attribution": "case_site_type_attribution.csv",
            "selected_sample_ids": dict(selected_ids),
        },
        "sequence_and_esm2_attribution": {
            "test1": "sequence_interpretation_test1/",
            "test2": "sequence_interpretation_test2/",
            "methods": "fixed-context residue occlusion and ESM2 integrated gradients",
        },
        "efficiency": "efficiency.json",
        "summary_figure": "paper_validation_summary.png",
        "limitations": [
            "No new independent external dataset was available.",
            "Test1 and Test2 analyses are retrospective because their labels were already viewed.",
            "Chemical-site deletion is a model intervention, not experimental proof of a physical interaction.",
            "A site-type deletion removes every pseudo-site carrying that type; overlapping site annotations mean it is not a pure isolated-type intervention.",
            "Timing excludes upstream ESM2 and ESMFold generation.",
        ],
    }
    save_json(summary, output_dir / "summary.json")
    print(f"Paper validation outputs saved to {output_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Complete paper-oriented frozen-model validation without new training."
    )
    parser.add_argument("--output-dir", default="runs/paper_validation")
    parser.add_argument(
        "--v1-checkpoint",
        default="runs/plm_fusion_esm2_geometry_confidence/best_model.pt",
    )
    parser.add_argument("--v2-checkpoint", default="runs/3d_v2_default/best_model.pt")
    parser.add_argument(
        "--test1-v1-predictions",
        default="runs/plm_fusion_esm2_geometry_confidence/test1_predictions.csv",
    )
    parser.add_argument(
        "--test1-v2-predictions",
        default="runs/3d_v2_default/test1_predictions.csv",
    )
    parser.add_argument(
        "--test2-v1-predictions",
        default="runs/plm_fusion_esm2_geometry_confidence/test2_predictions.csv",
    )
    parser.add_argument(
        "--test2-v2-predictions",
        default="runs/3d_v2_default/test2_predictions.csv",
    )
    parser.add_argument(
        "--oof-v1-predictions",
        default="runs/3d_v1_control_oof/predictions.csv",
    )
    parser.add_argument(
        "--oof-v2-predictions",
        default="runs/3d_v2_default/oof_predictions.csv",
    )
    parser.add_argument(
        "--test1-processed",
        default="data/processed/test1_chemical_sites_final_esm2.pt",
    )
    parser.add_argument(
        "--test2-processed",
        default="data/processed/test2_chemical_sites_final_esm2.pt",
    )
    parser.add_argument("--v1-test-threshold", type=float, default=0.85)
    parser.add_argument("--v1-oof-threshold", type=float, default=0.72987)
    parser.add_argument("--v2-threshold", type=float, default=0.677819)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--stratum-bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--minimum-stratum-samples", type=int, default=20)
    parser.add_argument("--minimum-stratum-per-class", type=int, default=5)
    parser.add_argument("--benchmark-samples", type=int, default=128)
    parser.add_argument("--benchmark-batch-size", type=int, default=32)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run_validation(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
