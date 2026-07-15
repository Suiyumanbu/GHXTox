"""Residue-level biological interpretation for frozen GHXTox checkpoints.

The analysis deliberately keeps model weights, thresholds, global sequence
features, and predicted geometry frozen.  Residue occlusion estimates the
effect of removing residue-level sequence evidence at a fixed position, while
integrated gradients only explains the ESM2 representation pathway.  Neither
quantity is presented as an experimental causal effect.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from ghxtox.constants import AA_TO_IDX, AMINO_ACIDS, UNK_TOKEN
from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.features import encode_amino_acids, encode_functional_groups, residue_feature_matrix
from ghxtox.models import GHXToxModel
from ghxtox.plm_embed import ESM2_MODELS, _load_esm2
from ghxtox.utils import move_batch_to_device, resolve_device, save_json


DEFAULT_INTERPRETATION_CHECKPOINTS = (
    "runs/plm_fusion_esm2_geometry_confidence/best_model.pt",
    "runs/plm_fusion_esm2_geometry_confidence_seed123/best_model.pt",
    "runs/plm_fusion_esm2_geometry_confidence_seed2025/best_model.pt",
)


def _include_structure(config: dict[str, Any]) -> bool:
    modality = str(config.get("model", {}).get("modality", "fusion")).lower()
    return modality not in {"sequence_only", "atom_only", "sequence_atom"}


def _include_atom(config: dict[str, Any]) -> bool:
    modality = str(config.get("model", {}).get("modality", "fusion")).lower()
    return modality in {"atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"}


def _repeat_single_batch(batch: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat a collated batch containing exactly one peptide."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    repeated: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    f"Cannot repeat non-sample tensor field '{key}' with shape {tuple(value.shape)}. "
                    "Atom-graph interpretation is not supported by this command."
                )
            repeated[key] = value.repeat((repeats,) + (1,) * (value.ndim - 1))
        elif isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"Expected one value for list field '{key}', got {len(value)}")
            repeated[key] = value * repeats
        elif value is None:
            repeated[key] = None
        else:
            repeated[key] = value
    return repeated


def _fixed_context_occlusion_batch(
    batch: dict[str, Any], positions: Sequence[int]
) -> dict[str, Any]:
    """Mask residue-level sequence evidence while preserving geometry/context.

    Global sequence descriptors and all structure tensors are intentionally
    unchanged.  This isolates local model evidence and is not a deletion or a
    physical mutation of the peptide.
    """

    occluded = _repeat_single_batch(batch, len(positions))
    for row, position in enumerate(positions):
        occluded["aa_ids"][row, position] = AA_TO_IDX[UNK_TOKEN]
        occluded["group_ids"][row, position] = 0
        occluded["residue_features"][row, position] = 0.0
        if "plm_features" in occluded:
            occluded["plm_features"][row, position] = 0.0
    return occluded


def residue_occlusion(
    model: torch.nn.Module,
    batch: dict[str, Any],
    length: int,
    chunk_size: int = 32,
) -> torch.Tensor:
    """Return baseline_logit - occluded_logit for each residue."""

    if length <= 0:
        return torch.empty(0)
    with torch.no_grad():
        baseline_logit = model(batch)["logits"][0]
        effects = []
        for start in range(0, length, chunk_size):
            positions = list(range(start, min(start + chunk_size, length)))
            occluded = _fixed_context_occlusion_batch(batch, positions)
            logits = model(occluded)["logits"]
            effects.append((baseline_logit - logits).detach().cpu())
    return torch.cat(effects)


def plm_integrated_gradients(
    model: torch.nn.Module,
    batch: dict[str, Any],
    length: int,
    steps: int = 16,
    baseline_mode: str = "sequence_mean",
) -> tuple[torch.Tensor, float, float]:
    """Integrated gradients for the ESM2 input pathway.

    Returns per-residue attribution, the endpoint logit difference, and the
    completeness residual ``delta_logit - sum(attribution)``.

    ``sequence_mean`` repeats the peptide's mean ESM2 vector at every position.
    It avoids the singular, far-out-of-distribution all-zero path through the
    ESM2 LayerNorm.  ``zero`` is retained for controlled tests only.
    """

    if steps < 2:
        raise ValueError("Integrated gradients requires at least two steps.")
    if "plm_features" not in batch:
        raise ValueError("PLM integrated gradients requires a plm_features tensor.")

    observed = batch["plm_features"].detach()
    if baseline_mode == "sequence_mean":
        baseline = observed[:, :length].mean(dim=1, keepdim=True).expand_as(observed).clone()
    elif baseline_mode == "zero":
        baseline = torch.zeros_like(observed)
    else:
        raise ValueError(f"Unsupported IG baseline mode: {baseline_mode}")
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=observed.device, dtype=observed.dtype)
    ig_batch = _repeat_single_batch(batch, steps + 1)
    scaled = baseline + alphas.view(-1, 1, 1) * (observed - baseline)
    scaled = scaled.detach().requires_grad_(True)
    ig_batch["plm_features"] = scaled

    logits = model(ig_batch)["logits"]
    gradients = torch.autograd.grad(logits.sum(), scaled, retain_graph=False)[0]
    trapezoid = (gradients[0] + gradients[-1] + 2.0 * gradients[1:-1].sum(dim=0)) / (2.0 * steps)
    attribution = ((observed[0] - baseline[0]) * trapezoid).sum(dim=-1)[:length]
    delta_logit = float((logits[-1] - logits[0]).detach().cpu())
    completeness_residual = delta_logit - float(attribution.detach().sum().cpu())
    return attribution.detach().cpu(), delta_logit, completeness_residual


def _attention_received(output: dict[str, torch.Tensor], length: int) -> torch.Tensor:
    attention = output.get("attention")
    if not torch.is_tensor(attention):
        return torch.full((length,), float("nan"))
    attention = attention.detach().cpu()
    if attention.ndim == 4:
        attention = attention.mean(dim=1)
    if attention.ndim == 3:
        attention = attention[0]
    if attention.ndim != 2:
        return torch.full((length,), float("nan"))
    return attention[:length, :length].mean(dim=0)


def ca_contact_features(
    coords: torch.Tensor,
    cutoff: float = 8.0,
    minimum_sequence_separation: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute all and long-range C-alpha contact counts per residue."""

    coords = coords.detach().float().cpu()
    length = coords.shape[0]
    if length == 0:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    distances = torch.cdist(coords, coords)
    valid = torch.isfinite(distances) & (distances > 0.0) & (distances <= cutoff)
    contact_count = valid.sum(dim=1)
    indices = torch.arange(length)
    separation = (indices[:, None] - indices[None, :]).abs()
    long_range_count = (valid & (separation >= minimum_sequence_separation)).sum(dim=1)
    return contact_count, long_range_count


def _sign_agreement(values: torch.Tensor, tolerance: float = 1e-8) -> torch.Tensor:
    positive = (values > tolerance).float().mean(dim=0)
    negative = (values < -tolerance).float().mean(dim=0)
    return torch.maximum(positive, negative)


def _l1_normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.abs().sum().clamp_min(1e-12)


def _mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return values.mean(dim=0), values.std(dim=0, unbiased=False)


def _safe_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _load_frozen_models(
    checkpoint_paths: Sequence[str | Path], device: torch.device
) -> list[dict[str, Any]]:
    loaded = []
    expected_family: tuple[str, int] | None = None
    for checkpoint_path in checkpoint_paths:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model_cfg = config.get("model", {})
        family = (
            str(model_cfg.get("modality", "fusion")).lower(),
            int(model_cfg.get("plm_embedding_dim", 0)),
        )
        if expected_family is None:
            expected_family = family
        elif family != expected_family:
            raise ValueError(
                "Interpretation checkpoints must use the same modality and PLM dimension; "
                f"expected {expected_family}, got {family} for {checkpoint_path}."
            )
        if _include_atom(config):
            raise ValueError(
                "This residue interpretation command does not support atom-graph checkpoints. "
                "Use the frozen 1D or pLDDT-aware 3D checkpoints."
            )
        model = GHXToxModel(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        loaded.append(
            {
                "path": str(checkpoint_path),
                "seed": int(config.get("seed", len(loaded))),
                "config": config,
                "model": model,
            }
        )
    if not loaded:
        raise ValueError("At least one checkpoint is required.")
    return loaded


def _per_checkpoint_attribution(
    model: torch.nn.Module,
    batch: dict[str, Any],
    length: int,
    ig_steps: int,
    occlusion_chunk_size: int,
    run_occlusion: bool,
    run_ig: bool,
    ig_baseline: str,
) -> dict[str, Any]:
    with torch.no_grad():
        output = model(batch)
        logit = output["logits"][0]
        probability = torch.sigmoid(logit)
        node_gate = output.get("node_gate")
        if torch.is_tensor(node_gate):
            node_gate = node_gate[0, :length].detach().cpu()
        else:
            node_gate = torch.full((length,), float("nan"))
        global_gate = output.get("global_gate")
        global_gate_value = _safe_float(global_gate[0]) if torch.is_tensor(global_gate) else float("nan")
        attention_received = _attention_received(output, length)

    occlusion = (
        residue_occlusion(model, batch, length, chunk_size=occlusion_chunk_size)
        if run_occlusion
        else torch.full((length,), float("nan"))
    )
    if run_ig:
        ig, ig_delta, ig_residual = plm_integrated_gradients(
            model, batch, length, steps=ig_steps, baseline_mode=ig_baseline
        )
    else:
        ig = torch.full((length,), float("nan"))
        ig_delta = float("nan")
        ig_residual = float("nan")
    return {
        "logit": _safe_float(logit),
        "probability": _safe_float(probability),
        "global_gate": global_gate_value,
        "node_gate": node_gate,
        "attention_received": attention_received,
        "occlusion": occlusion,
        "ig": ig,
        "ig_delta_logit": ig_delta,
        "ig_completeness_residual": ig_residual,
    }


def _nanmean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(values)
    count = finite.sum(dim=0).clamp_min(1)
    safe = torch.where(finite, values, torch.zeros_like(values))
    mean = safe.sum(dim=0) / count
    variance = torch.where(finite, (values - mean) ** 2, torch.zeros_like(values)).sum(dim=0) / count
    mean = torch.where(finite.any(dim=0), mean, torch.full_like(mean, float("nan")))
    std = torch.where(finite.any(dim=0), variance.sqrt(), torch.full_like(mean, float("nan")))
    return mean, std


def aggregate_attributions(
    results: Sequence[dict[str, Any]],
) -> dict[str, torch.Tensor | float]:
    """Aggregate residue explanations across frozen random seeds."""

    if not results:
        raise ValueError("No checkpoint results to aggregate.")
    occlusion = torch.stack([result["occlusion"] for result in results])
    ig = torch.stack([result["ig"] for result in results])
    node_gate = torch.stack([result["node_gate"] for result in results])
    attention = torch.stack([result["attention_received"] for result in results])
    occlusion_mean, occlusion_std = _nanmean_std(occlusion)
    ig_mean, ig_std = _nanmean_std(ig)
    node_gate_mean, node_gate_std = _nanmean_std(node_gate)
    attention_mean, attention_std = _nanmean_std(attention)

    has_occlusion = bool(torch.isfinite(occlusion_mean).any())
    has_ig = bool(torch.isfinite(ig_mean).any())
    if has_occlusion:
        normalized_occlusion = _l1_normalize(torch.nan_to_num(occlusion_mean))
    else:
        normalized_occlusion = torch.zeros_like(occlusion_mean)
    if has_ig:
        normalized_ig = _l1_normalize(torch.nan_to_num(ig_mean))
    else:
        normalized_ig = torch.zeros_like(ig_mean)
    if has_occlusion and has_ig:
        consensus = 0.5 * (normalized_occlusion + normalized_ig)
        same_direction = (occlusion_mean * ig_mean) > 0.0
        robust_direction = torch.where(same_direction, torch.sign(consensus), torch.zeros_like(consensus))
    elif has_occlusion:
        consensus = normalized_occlusion
        same_direction = torch.ones_like(consensus, dtype=torch.bool)
        robust_direction = torch.sign(consensus)
    else:
        consensus = normalized_ig
        same_direction = torch.ones_like(consensus, dtype=torch.bool)
        robust_direction = torch.sign(consensus)

    return {
        "probability_mean": float(torch.tensor([r["probability"] for r in results]).mean()),
        "probability_std": float(torch.tensor([r["probability"] for r in results]).std(unbiased=False)),
        "global_gate_mean": float(torch.tensor([r["global_gate"] for r in results]).mean()),
        "global_gate_std": float(torch.tensor([r["global_gate"] for r in results]).std(unbiased=False)),
        "occlusion_mean": occlusion_mean,
        "occlusion_std": occlusion_std,
        "occlusion_sign_agreement": _sign_agreement(torch.nan_to_num(occlusion)),
        "occlusion_normalized": normalized_occlusion,
        "ig_mean": ig_mean,
        "ig_std": ig_std,
        "ig_sign_agreement": _sign_agreement(torch.nan_to_num(ig)),
        "ig_normalized": normalized_ig,
        "node_gate_mean": node_gate_mean,
        "node_gate_std": node_gate_std,
        "attention_received_mean": attention_mean,
        "attention_received_std": attention_std,
        "consensus_score": consensus,
        "same_direction": same_direction,
        "robust_direction": robust_direction,
    }


def _spatial_clustering(coords: torch.Tensor, scores: torch.Tensor) -> tuple[float, float, float]:
    length = coords.shape[0]
    top_k = min(max(2, math.ceil(length * 0.2)), length)
    if length < 2:
        return float("nan"), float("nan"), float("nan")
    distances = torch.cdist(coords.detach().float().cpu(), coords.detach().float().cpu())
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    all_mean = float(distances[upper].mean())
    top = torch.topk(scores.abs(), k=top_k).indices
    top_distances = distances[top][:, top]
    top_upper = torch.triu(torch.ones_like(top_distances, dtype=torch.bool), diagonal=1)
    top_mean = float(top_distances[top_upper].mean())
    ratio = top_mean / all_mean if all_mean > 0.0 else float("nan")
    return top_mean, all_mean, ratio


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _amino_acid_summary(residue_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in residue_rows:
        grouped[str(row["residue"])].append(row)
    output = []
    for amino_acid in AMINO_ACIDS:
        rows = grouped.get(amino_acid, [])
        if not rows:
            continue
        output.append(
            {
                "residue": amino_acid,
                "count": len(rows),
                "mean_occlusion_delta_logit": sum(float(r["occlusion_delta_logit_mean"]) for r in rows)
                / len(rows),
                "mean_plm_ig": sum(float(r["plm_ig_mean"]) for r in rows) / len(rows),
                "mean_consensus_score": sum(float(r["consensus_score"]) for r in rows) / len(rows),
                "mean_abs_consensus_score": sum(abs(float(r["consensus_score"])) for r in rows)
                / len(rows),
                "promoting_hotspot_count": sum(
                    int(bool(r["is_hotspot"]) and float(r["robust_direction"]) > 0.0) for r in rows
                ),
                "suppressing_hotspot_count": sum(
                    int(bool(r["is_hotspot"]) and float(r["robust_direction"]) < 0.0) for r in rows
                ),
            }
        )
    return output


def _safe_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"


def _plot_sample(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting interpretation results requires matplotlib.") from exc

    positions = [int(row["position"]) for row in rows]
    residues = [str(row["residue"]) for row in rows]
    occlusion = [float(row["occlusion_normalized"]) for row in rows]
    ig = [float(row["plm_ig_normalized"]) for row in rows]
    plddt = [float(row["plddt"]) for row in rows]
    node_gate = [float(row["node_gate_mean"]) for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(max(8.0, len(rows) * 0.28), 5.8), sharex=True)
    width = 0.4
    axes[0].bar([p - width / 2 for p in positions], occlusion, width=width, label="Fixed-context occlusion")
    axes[0].bar([p + width / 2 for p in positions], ig, width=width, label="ESM2 integrated gradients")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("L1-normalized contribution")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(
        f"{rows[0]['sample_id']} | mean toxicity probability={float(rows[0]['probability_mean']):.3f}"
    )

    axes[1].plot(positions, plddt, marker="o", markersize=2.5, label="pLDDT")
    axes[1].plot(positions, node_gate, marker="o", markersize=2.5, label="3D node gate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("Structure confidence/use")
    axes[1].set_xlabel("Residue position")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"{aa}{position}" for aa, position in zip(residues, positions)], rotation=90)
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _select_plot_ids(sample_rows: Sequence[dict[str, Any]], threshold: float, limit: int) -> list[str]:
    if limit <= 0:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        label = row.get("label")
        probability = float(row["probability_mean"])
        if label is None or int(label) < 0:
            group = "unlabeled"
        elif int(label) == 1 and probability >= threshold:
            group = "tp"
        elif int(label) == 0 and probability < threshold:
            group = "tn"
        elif int(label) == 0:
            group = "fp"
        else:
            group = "fn"
        groups[group].append(row)

    for group, rows in groups.items():
        reverse = group in {"tp", "fp", "unlabeled"}
        rows.sort(key=lambda item: float(item["probability_mean"]), reverse=reverse)
    selected: list[str] = []
    order = ["tp", "tn", "fp", "fn", "unlabeled"]
    while len(selected) < limit:
        changed = False
        for group in order:
            rows = groups.get(group, [])
            if rows and len(selected) < limit:
                selected.append(str(rows.pop(0)["sample_id"]))
                changed = True
        if not changed:
            break
    return selected


def _embed_esm2_sequences(
    labeled_sequences: Sequence[tuple[str, str]],
    device: torch.device,
    model_name: str,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model, alphabet, layer, _ = _load_esm2(model_name, device)
    batch_converter = alphabet.get_batch_converter()
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(labeled_sequences), batch_size):
            chunk = labeled_sequences[start : start + batch_size]
            _, _, tokens = batch_converter(list(chunk))
            output = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
            representations = output["representations"][layer].detach().cpu()
            for (mutant_id, sequence), representation in zip(chunk, representations):
                embeddings[mutant_id] = representation[1 : len(sequence) + 1].contiguous().float()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embeddings


def _mutant_record(
    record: dict[str, Any], position: int, mutant_aa: str, embedding: torch.Tensor
) -> dict[str, Any]:
    sequence = str(record["sequence"])
    mutant_sequence = sequence[:position] + mutant_aa + sequence[position + 1 :]
    mutant = dict(record)
    mutant["sample_id"] = f"{record['sample_id']}__{sequence[position]}{position + 1}{mutant_aa}"
    mutant["sequence"] = mutant_sequence
    mutant["aa_ids"] = encode_amino_acids(mutant_sequence)
    mutant["group_ids"] = encode_functional_groups(mutant_sequence)
    mutant["residue_features"] = residue_feature_matrix(mutant_sequence)
    mutant["plm_features"] = embedding
    mutant["global_features"] = None
    return mutant


def _mutation_scan(
    selected_records: Sequence[dict[str, Any]],
    selected_positions: dict[str, Sequence[int]],
    loaded_models: Sequence[dict[str, Any]],
    device: torch.device,
    model_name: str,
    embedding_batch_size: int,
    classifier_batch_size: int,
) -> list[dict[str, Any]]:
    definitions: list[tuple[str, str, dict[str, Any], int, str]] = []
    for record in selected_records:
        sample_id = str(record["sample_id"])
        sequence = str(record["sequence"])
        for position in selected_positions.get(sample_id, []):
            for mutant_aa in AMINO_ACIDS:
                if mutant_aa == sequence[position]:
                    continue
                mutant_id = f"{sample_id}__{sequence[position]}{position + 1}{mutant_aa}"
                mutant_sequence = sequence[:position] + mutant_aa + sequence[position + 1 :]
                definitions.append((mutant_id, mutant_sequence, record, position, mutant_aa))
    if not definitions:
        return []

    embeddings = _embed_esm2_sequences(
        [(definition[0], definition[1]) for definition in definitions],
        device=device,
        model_name=model_name,
        batch_size=embedding_batch_size,
    )
    rows = []
    for loaded in loaded_models:
        model = loaded["model"]
        config = loaded["config"]
        seed = loaded["seed"]
        baseline_logits: dict[str, float] = {}
        for record in selected_records:
            batch = collate_peptides(
                [record], include_structure=_include_structure(config), include_atom=False
            )
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                baseline_logits[str(record["sample_id"])] = _safe_float(model(batch)["logits"][0])

        mutant_records = [
            _mutant_record(record, position, mutant_aa, embeddings[mutant_id])
            for mutant_id, _, record, position, mutant_aa in definitions
        ]
        offset = 0
        with torch.no_grad():
            for start in range(0, len(mutant_records), classifier_batch_size):
                chunk = mutant_records[start : start + classifier_batch_size]
                batch = collate_peptides(
                    chunk, include_structure=_include_structure(config), include_atom=False
                )
                logits = model(move_batch_to_device(batch, device))["logits"].detach().cpu().tolist()
                for mutant, mutant_logit in zip(chunk, logits):
                    definition = definitions[offset]
                    _, mutant_sequence, source_record, position, mutant_aa = definition
                    source_id = str(source_record["sample_id"])
                    rows.append(
                        {
                            "sample_id": source_id,
                            "seed": seed,
                            "position": position + 1,
                            "wild_type": str(source_record["sequence"])[position],
                            "mutant": mutant_aa,
                            "mutant_sequence": mutant_sequence,
                            "baseline_logit": baseline_logits[source_id],
                            "mutant_logit": mutant_logit,
                            "delta_logit": mutant_logit - baseline_logits[source_id],
                        }
                    )
                    offset += 1
        if offset != len(definitions):
            raise RuntimeError("Mutation scan bookkeeping mismatch.")

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["sample_id"]), int(row["position"]), str(row["wild_type"]), str(row["mutant"]))
        grouped[key].append(row)
    aggregated = []
    for (sample_id, position, wild_type, mutant), group in grouped.items():
        deltas = torch.tensor([float(row["delta_logit"]) for row in group])
        aggregated.append(
            {
                "sample_id": sample_id,
                "position": position,
                "wild_type": wild_type,
                "mutant": mutant,
                "mutant_sequence": group[0]["mutant_sequence"],
                "delta_logit_mean": float(deltas.mean()),
                "delta_logit_std": float(deltas.std(unbiased=False)),
                "seed_sign_agreement": float(_sign_agreement(deltas.unsqueeze(1))[0]),
                "analysis_scope": "sequence-consistent ESM2; fixed ESMFold geometry",
            }
        )
    return sorted(aggregated, key=lambda row: (row["sample_id"], row["position"], row["mutant"]))


def interpret(args: argparse.Namespace) -> dict[str, Any]:
    if not args.occlusion and not args.integrated_gradients:
        raise ValueError("Enable at least one of --occlusion or --integrated-gradients.")
    device = resolve_device(args.device)
    loaded_models = _load_frozen_models(args.checkpoints, device)
    first_config = loaded_models[0]["config"]
    processed_path = Path(args.processed)
    dataset = PeptideTensorDataset(processed_path, require_labels=False)
    required_plm_dim = int(first_config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_plm_dim, processed_path)

    records = dataset.records
    if args.sample_id:
        selected_ids = set(args.sample_id)
        records = [record for record in records if str(record["sample_id"]) in selected_ids]
        missing = sorted(selected_ids - {str(record["sample_id"]) for record in records})
        if missing:
            raise ValueError(f"Unknown sample IDs: {missing[:10]}")
    if args.max_samples_per_label is not None:
        grouped_records: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            label = record.get("label")
            grouped_records[int(label) if label is not None else -1].append(record)
        records = [
            record
            for label in sorted(grouped_records, reverse=True)
            for record in grouped_records[label][: args.max_samples_per_label]
        ]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records selected for interpretation.")

    output_dir = Path(args.output_dir)
    residue_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    selected_record_lookup: dict[str, dict[str, Any]] = {}
    mutation_positions: dict[str, list[int]] = {}

    for record_index, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        sequence = str(record["sequence"])
        length = len(sequence)
        selected_record_lookup[sample_id] = record
        checkpoint_results = []
        reference_batch: dict[str, Any] | None = None
        for loaded in loaded_models:
            config = loaded["config"]
            batch = collate_peptides(
                [record], include_structure=_include_structure(config), include_atom=False
            )
            batch = move_batch_to_device(batch, device)
            reference_batch = batch
            result = _per_checkpoint_attribution(
                loaded["model"],
                batch,
                length=length,
                ig_steps=args.ig_steps,
                occlusion_chunk_size=args.occlusion_chunk_size,
                run_occlusion=args.occlusion,
                run_ig=args.integrated_gradients,
                ig_baseline=args.ig_baseline,
            )
            result["seed"] = loaded["seed"]
            checkpoint_results.append(result)

        assert reference_batch is not None
        aggregate = aggregate_attributions(checkpoint_results)
        coords = reference_batch.get("coords")
        if torch.is_tensor(coords):
            coords_cpu = coords[0, :length].detach().cpu()
            contact_count, long_range_count = ca_contact_features(
                coords_cpu,
                cutoff=args.contact_cutoff,
                minimum_sequence_separation=args.minimum_sequence_separation,
            )
            top_distance, all_distance, clustering_ratio = _spatial_clustering(
                coords_cpu, aggregate["consensus_score"]
            )
        else:
            coords_cpu = torch.zeros(length, 3)
            contact_count = torch.full((length,), -1)
            long_range_count = torch.full((length,), -1)
            top_distance = all_distance = clustering_ratio = float("nan")

        consensus = aggregate["consensus_score"]
        ranks = torch.empty(length, dtype=torch.long)
        order = torch.argsort(consensus.abs(), descending=True)
        ranks[order] = torch.arange(1, length + 1)
        hotspot_count = max(1, math.ceil(length * args.hotspot_fraction))
        hotspot_mask = ranks <= hotspot_count
        mutation_positions[sample_id] = [int(index) for index in order[: args.mutation_top_positions]]
        plddt = reference_batch["plddt"][0, :length].detach().cpu()

        label = record.get("label")
        label_value = int(label) if label is not None else -1
        top_labels = [f"{sequence[index]}{index + 1}" for index in order[: min(5, length)].tolist()]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "sequence": sequence,
                "label": label_value,
                "length": length,
                "probability_mean": aggregate["probability_mean"],
                "probability_std": aggregate["probability_std"],
                "prediction": int(float(aggregate["probability_mean"]) >= args.threshold),
                "global_3d_gate_mean": aggregate["global_gate_mean"],
                "global_3d_gate_std": aggregate["global_gate_std"],
                "top_residues": ";".join(top_labels),
                "top_hotspot_mean_ca_distance": top_distance,
                "all_residue_mean_ca_distance": all_distance,
                "spatial_clustering_ratio": clustering_ratio,
                "mean_ig_completeness_residual": sum(
                    abs(float(result["ig_completeness_residual"])) for result in checkpoint_results
                )
                / len(checkpoint_results),
                "mean_ig_relative_completeness_error": sum(
                    abs(float(result["ig_completeness_residual"]))
                    / max(abs(float(result["ig_delta_logit"])), 1e-6)
                    for result in checkpoint_results
                )
                / len(checkpoint_results),
            }
        )

        for position, residue in enumerate(sequence):
            row = {
                "sample_id": sample_id,
                "sequence": sequence,
                "label": label_value,
                "position": position + 1,
                "residue": residue,
                "probability_mean": aggregate["probability_mean"],
                "probability_std": aggregate["probability_std"],
                "plddt": float(plddt[position]),
                "node_gate_mean": float(aggregate["node_gate_mean"][position]),
                "node_gate_std": float(aggregate["node_gate_std"][position]),
                "attention_received_mean": float(aggregate["attention_received_mean"][position]),
                "attention_received_std": float(aggregate["attention_received_std"][position]),
                "ca_contact_count": int(contact_count[position]),
                "long_range_ca_contact_count": int(long_range_count[position]),
                "occlusion_delta_logit_mean": float(aggregate["occlusion_mean"][position]),
                "occlusion_delta_logit_std": float(aggregate["occlusion_std"][position]),
                "occlusion_seed_sign_agreement": float(aggregate["occlusion_sign_agreement"][position]),
                "occlusion_normalized": float(aggregate["occlusion_normalized"][position]),
                "plm_ig_mean": float(aggregate["ig_mean"][position]),
                "plm_ig_std": float(aggregate["ig_std"][position]),
                "plm_ig_seed_sign_agreement": float(aggregate["ig_sign_agreement"][position]),
                "plm_ig_normalized": float(aggregate["ig_normalized"][position]),
                "same_direction_between_methods": bool(aggregate["same_direction"][position]),
                "consensus_score": float(consensus[position]),
                "absolute_consensus_rank": int(ranks[position]),
                "is_hotspot": bool(hotspot_mask[position]),
                "robust_direction": float(aggregate["robust_direction"][position]),
            }
            residue_rows.append(row)
        print(f"interpreted={record_index}/{len(records)} sample_id={sample_id}")

    residue_fields = list(residue_rows[0])
    sample_fields = list(sample_rows[0])
    _write_csv(output_dir / "residue_attributions.csv", residue_rows, residue_fields)
    _write_csv(output_dir / "sample_summary.csv", sample_rows, sample_fields)
    amino_rows = _amino_acid_summary(residue_rows)
    if amino_rows:
        _write_csv(output_dir / "amino_acid_summary.csv", amino_rows, list(amino_rows[0]))

    plot_ids = _select_plot_ids(sample_rows, args.threshold, args.plot_limit)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in residue_rows:
        by_sample[str(row["sample_id"])].append(row)
    for sample_id in plot_ids:
        _plot_sample(by_sample[sample_id], output_dir / "figures" / f"{_safe_stem(sample_id)}.png")

    mutation_rows: list[dict[str, Any]] = []
    if args.mutation_scan:
        mutation_records = [selected_record_lookup[sample_id] for sample_id in plot_ids]
        mutation_rows = _mutation_scan(
            mutation_records,
            mutation_positions,
            loaded_models,
            device=device,
            model_name=args.esm2_model,
            embedding_batch_size=args.mutation_embedding_batch_size,
            classifier_batch_size=args.mutation_classifier_batch_size,
        )
        if mutation_rows:
            _write_csv(output_dir / "mutation_scan.csv", mutation_rows, list(mutation_rows[0]))

    summary = {
        "processed": str(processed_path),
        "checkpoints": [loaded["path"] for loaded in loaded_models],
        "seeds": [loaded["seed"] for loaded in loaded_models],
        "num_samples": len(sample_rows),
        "num_residues": len(residue_rows),
        "num_mutations": len(mutation_rows),
        "threshold_for_case_labels_only": args.threshold,
        "methods": {
            "fixed_context_occlusion": bool(args.occlusion),
            "plm_integrated_gradients": bool(args.integrated_gradients),
            "integrated_gradients_steps": args.ig_steps,
            "integrated_gradients_baseline": args.ig_baseline,
            "mutation_scan": bool(args.mutation_scan),
        },
        "interpretation_scope": {
            "occlusion": (
                "AA identity, functional group, deterministic residue descriptors, and the ESM2 vector "
                "are masked at one position; geometry and global descriptors remain frozen."
            ),
            "integrated_gradients": (
                "Attribution of the ESM2 input pathway relative to a repeated within-peptide mean "
                "ESM2 vector by default; other inputs remain frozen."
            ),
            "attention_and_gate": (
                "Model-use diagnostics, not causal residue importance."
            ),
            "mutation_scan": (
                "Mutant AA descriptors and ESM2 representations are regenerated, but ESMFold geometry "
                "is kept fixed; results require structural or experimental validation."
            ),
        },
        "hotspot_fraction": args.hotspot_fraction,
        "contact_cutoff_angstrom": args.contact_cutoff,
        "minimum_sequence_separation": args.minimum_sequence_separation,
        "generated_files": [
            "residue_attributions.csv",
            "sample_summary.csv",
            "amino_acid_summary.csv",
            *( ["mutation_scan.csv"] if mutation_rows else [] ),
            "figures/*.png",
        ],
    }
    save_json(summary, output_dir / "summary.json")
    print(f"Biological interpretation saved to {output_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explain frozen GHXTox predictions at residue level across random seeds."
    )
    parser.add_argument("--processed", default="data/processed/test1_cached_func_esm2.pt")
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_INTERPRETATION_CHECKPOINTS))
    parser.add_argument("--output-dir", default="runs/biological_interpretation/test1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-samples-per-label",
        type=int,
        default=None,
        help="Deterministically keep at most N records from each label before applying --max-samples.",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--ig-baseline", choices=["sequence_mean", "zero"], default="sequence_mean")
    parser.add_argument("--occlusion-chunk-size", type=int, default=32)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--minimum-sequence-separation", type=int, default=3)
    parser.add_argument("--hotspot-fraction", type=float, default=0.20)
    parser.add_argument("--plot-limit", type=int, default=12)
    parser.add_argument("--mutation-scan", action="store_true")
    parser.add_argument("--mutation-top-positions", type=int, default=3)
    parser.add_argument("--mutation-embedding-batch-size", type=int, default=8)
    parser.add_argument("--mutation-classifier-batch-size", type=int, default=32)
    parser.add_argument("--esm2-model", default="esm2_t33_650M_UR50D", choices=sorted(ESM2_MODELS))
    parser.set_defaults(occlusion=True, integrated_gradients=True)
    parser.add_argument("--no-occlusion", dest="occlusion", action="store_false")
    parser.add_argument(
        "--no-integrated-gradients", dest="integrated_gradients", action="store_false"
    )
    return parser


def main() -> None:
    interpret(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
