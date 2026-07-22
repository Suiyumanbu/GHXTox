"""Inference for the retained 3D-v2 + ToxPLTC-derived ProtT5 default.

The two experts remain independently auditable. Their probability weight and
decision threshold are frozen from leave-one-group-fold-out OOF selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.plm_textcnn import PooledPLMTextCNN, _predict, _probability_metrics
from ghxtox.utils import load_json, move_batch_to_device, resolve_device, resolve_inference_checkpoint, save_json


DEFAULT_CONFIG = "configs/default_toxpltc_hybrid.json"


def combine_probabilities(
    structure_probabilities: np.ndarray,
    prott5_probabilities: np.ndarray,
    *,
    prott5_weight: float,
) -> np.ndarray:
    """Convexly combine aligned structure and ProtT5 probabilities."""

    structure = np.asarray(structure_probabilities, dtype=np.float64)
    prott5 = np.asarray(prott5_probabilities, dtype=np.float64)
    if structure.shape != prott5.shape:
        raise ValueError(
            f"Probability shapes differ: {structure.shape} versus {prott5.shape}."
        )
    if not 0.0 <= float(prott5_weight) <= 1.0:
        raise ValueError("prott5_weight must lie in [0, 1].")
    return float(prott5_weight) * prott5 + (1.0 - float(prott5_weight)) * structure


def validate_alignment(
    structure_sequences: list[str],
    structure_labels: list[int | None],
    prott5_payload: dict[str, Any],
) -> None:
    """Require immutable row, sequence and available-label alignment."""

    prott5_sequences = list(prott5_payload.get("sequences", []))
    if structure_sequences != prott5_sequences:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(structure_sequences, prott5_sequences)
                )
                if left != right
            ),
            min(len(structure_sequences), len(prott5_sequences)),
        )
        raise ValueError(f"3D/ProtT5 sequence alignment mismatch at row {mismatch}.")
    prott5_labels = prott5_payload.get("labels")
    if torch.is_tensor(prott5_labels):
        labels = prott5_labels.long().tolist()
        for index, (left, right) in enumerate(zip(structure_labels, labels)):
            if left is not None and int(left) != int(right):
                raise ValueError(f"3D/ProtT5 label alignment mismatch at row {index}.")


def _predict_structure(
    processed_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = PeptideTensorDataset(processed_path, require_labels=False)
    checkpoint, selected, _, fallback_used = resolve_inference_checkpoint(
        checkpoint_path, dataset.records, device, requested_threshold=None
    )
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_dim, processed_path)
    loader = DataLoader(
        dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_peptides
    )
    probabilities: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    sample_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int | None] = []
    with torch.inference_mode():
        for batch in loader:
            sample_ids.extend(batch["sample_id"])
            sequences.extend(batch["sequence"])
            batch_labels = batch["labels"]
            labels.extend(
                [None] * len(batch["sample_id"])
                if batch_labels is None
                else [int(value) for value in batch_labels.tolist()]
            )
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            probabilities.append(torch.sigmoid(output["logits"]).float().cpu())
            gates.append(output["global_gate"].float().cpu())
    return {
        "sample_ids": sample_ids,
        "sequences": sequences,
        "labels": labels,
        "probabilities": torch.cat(probabilities).numpy(),
        "gates": torch.cat(gates).numpy(),
        "checkpoint": selected,
        "fallback_used": fallback_used,
    }


def _predict_prott5(
    payload: dict[str, Any],
    checkpoint_dirs: list[str],
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    features = payload["features"].float().numpy()
    members = []
    for checkpoint_dir in checkpoint_dirs:
        directory = Path(checkpoint_dir)
        for fold in range(5):
            checkpoint_path = directory / f"fold{fold}_best_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing ProtT5 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model_config = dict(checkpoint["model"])
            model_config["window_sizes"] = tuple(model_config["window_sizes"])
            model = PooledPLMTextCNN(**model_config).to(device)
            model.load_state_dict(checkpoint["model_state"])
            members.append(_predict(model, features, batch_size, device))
    if len(members) != 15:
        raise ValueError(f"Expected 15 ProtT5 fold/seed members, found {len(members)}.")
    return np.mean(np.stack(members, axis=0), axis=0)


def predict_hybrid(
    config_path: str | Path,
    processed_path: str | Path,
    prott5_features_path: str | Path,
    output_path: str | Path,
    *,
    metrics_path: str | Path | None,
    device_name: str,
    batch_size: int,
) -> dict[str, Any]:
    config = load_json(config_path)
    device = resolve_device(device_name)
    structure = _predict_structure(
        processed_path,
        config["structure_expert"]["checkpoint"],
        device=device,
        batch_size=batch_size,
    )
    prott5_payload = torch.load(
        prott5_features_path, map_location="cpu", weights_only=False
    )
    validate_alignment(structure["sequences"], structure["labels"], prott5_payload)
    prott5_probabilities = _predict_prott5(
        prott5_payload,
        list(config["prott5_expert"]["checkpoint_dirs"]),
        device=device,
        batch_size=batch_size,
    )

    fallback_used = bool(structure["fallback_used"])
    if fallback_used and config["inference"].get("fallback_mode") == "prott5_only":
        probabilities = prott5_probabilities.astype(np.float64)
        threshold = float(config["prott5_expert"]["standalone_threshold"])
        prott5_weight = 1.0
    else:
        prott5_weight = float(config["fusion"]["prott5_weight"])
        probabilities = combine_probabilities(
            structure["probabilities"],
            prott5_probabilities,
            prott5_weight=prott5_weight,
        )
        threshold = float(config["fusion"]["threshold"])

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_index",
        "sample_id",
        "sequence",
        "label",
        "structure_probability",
        "prott5_probability",
        "toxicity_probability",
        "prediction",
        "decision_threshold",
        "prott5_weight",
        "global_3d_gate",
        "fallback_used",
    ]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": structure["sample_ids"][index],
                    "sequence": structure["sequences"][index],
                    "label": "" if structure["labels"][index] is None else structure["labels"][index],
                    "structure_probability": f"{float(structure['probabilities'][index]):.10g}",
                    "prott5_probability": f"{float(prott5_probabilities[index]):.10g}",
                    "toxicity_probability": f"{float(probability):.10g}",
                    "prediction": int(probability >= threshold),
                    "decision_threshold": f"{threshold:.10g}",
                    "prott5_weight": f"{prott5_weight:.10g}",
                    "global_3d_gate": f"{float(structure['gates'][index]):.10g}",
                    "fallback_used": int(fallback_used),
                }
            )

    available_labels = [label for label in structure["labels"] if label is not None]
    metrics = None
    if len(available_labels) == len(structure["labels"]):
        metrics = _probability_metrics(
            np.asarray(available_labels, dtype=np.int64), probabilities, threshold
        )
    summary = {
        "model": config["name"],
        "protocol": {
            "config": str(config_path),
            "processed": str(processed_path),
            "prott5_features": str(prott5_features_path),
            "structure_checkpoint": structure["checkpoint"],
            "num_prott5_members": 15,
            "prott5_weight": prott5_weight,
            "structure_weight": 1.0 - prott5_weight,
            "threshold": threshold,
            "fallback_used": fallback_used,
            "parameters_selected_from_test_labels": False,
        },
        "metrics": metrics,
    }
    if metrics_path is not None:
        save_json(summary, metrics_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--prott5-features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict_hybrid(
        args.config,
        args.processed,
        args.prott5_features,
        args.output,
        metrics_path=args.metrics,
        device_name=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
