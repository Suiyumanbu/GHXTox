"""Leakage-controlled PLM + Borderline-SMOTE + TextCNN experiments.

This module reproduces the useful part of the ToxPLTC training recipe while
keeping synthetic samples strictly inside each training fold.  It accepts any
fixed-length pooled protein-language-model representation, so the same code
can screen the existing ESM2 cache and later run the intended ProtT5 expert.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ghxtox.utils import resolve_device, save_json, set_seed


@dataclass(frozen=True)
class BorderlineSMOTEReport:
    original_positive: int
    original_negative: int
    danger_positive: int
    synthetic_positive: int
    output_positive: int
    output_negative: int


class PooledPLMTextCNN(nn.Module):
    """ToxPLTC-style classifier for one pooled PLM vector per peptide."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        window_sizes: tuple[int, ...] = (4, 5, 6),
        num_filters: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not window_sizes:
            raise ValueError("window_sizes must not be empty.")
        if max(window_sizes) > projection_dim:
            raise ValueError("Every convolution window must fit inside projection_dim.")
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.window_sizes = tuple(int(value) for value in window_sizes)
        self.num_filters = int(num_filters)
        self.feature_adjust = nn.Linear(self.input_dim, self.projection_dim)
        self.batch_norm = nn.BatchNorm1d(self.projection_dim)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(1, self.num_filters, kernel_size=size) for size in self.window_sizes]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.hidden = nn.Sequential(
            nn.Linear(len(self.window_sizes) * self.num_filters, 128),
            nn.Dropout(float(dropout)),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.feature_adjust.weight)
        nn.init.zeros_(self.feature_adjust.bias)
        for convolution in self.convolutions:
            nn.init.xavier_uniform_(convolution.weight)
            nn.init.zeros_(convolution.bias)
        nn.init.xavier_uniform_(self.hidden[0].weight)
        nn.init.zeros_(self.hidden[0].bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.batch_norm(self.feature_adjust(features))).unsqueeze(1)
        pooled = []
        for convolution in self.convolutions:
            convolved = torch.relu(convolution(hidden))
            pooled.append(convolved.amax(dim=-1))
        representation = self.dropout(torch.cat(pooled, dim=-1))
        return self.classifier(self.hidden(representation)).squeeze(-1)


def extract_pooled_features(processed: str | Path, output: str | Path, source_model: str) -> dict[str, Any]:
    """Mean-pool cached residue PLM embeddings without loading structure tensors onto GPU."""

    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    features = []
    labels = []
    sample_ids = []
    sequences = []
    expected_dim: int | None = None
    for index, record in enumerate(records):
        residue_features = record.get("plm_features")
        if not torch.is_tensor(residue_features) or residue_features.ndim != 2:
            raise ValueError(f"Record {index} has no two-dimensional plm_features tensor.")
        if residue_features.shape[0] == 0:
            raise ValueError(f"Record {index} has an empty plm_features tensor.")
        dimension = int(residue_features.shape[1])
        if expected_dim is None:
            expected_dim = dimension
        elif dimension != expected_dim:
            raise ValueError(f"Inconsistent PLM dimensions: expected {expected_dim}, found {dimension}.")
        features.append(residue_features.float().mean(dim=0))
        label = record.get("label")
        if label is None:
            raise ValueError(f"Record {index} is missing a label.")
        labels.append(int(label))
        sample_ids.append(str(record.get("sample_id", f"sample_{index}")))
        sequences.append(str(record.get("sequence", "")))

    result = {
        "features": torch.stack(features),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "sequences": sequences,
        "metadata": {
            "processed": str(processed),
            "source_model": str(source_model),
            "pooling": "mean over valid residue embeddings",
            "num_samples": len(records),
            "feature_dim": int(expected_dim or 0),
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return result["metadata"]


def _read_labeled_fasta(path: str | Path) -> tuple[list[str], list[str], list[int]]:
    sample_ids: list[str] = []
    sequences: list[str] = []
    labels: list[int] = []
    current_id: str | None = None
    current_sequence: list[str] = []

    def finish_record() -> None:
        if current_id is None:
            return
        sequence = "".join(current_sequence).replace(" ", "").upper()
        if not sequence:
            raise ValueError(f"Empty sequence for FASTA record {current_id!r}.")
        try:
            label = int(current_id.rsplit("|", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"FASTA header {current_id!r} must end in '|0' or '|1'."
            ) from exc
        if label not in {0, 1}:
            raise ValueError(f"Unsupported label {label} in FASTA header {current_id!r}.")
        sample_ids.append(current_id)
        sequences.append(sequence)
        labels.append(label)

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish_record()
                current_id = line[1:].strip()
                current_sequence = []
            else:
                if current_id is None:
                    raise ValueError("Sequence encountered before the first FASTA header.")
                current_sequence.append(line)
    finish_record()
    if not sample_ids:
        raise ValueError(f"No FASTA records found in {path}.")
    return sample_ids, sequences, labels


def import_feature_csv(
    csv_path: str | Path,
    fasta_path: str | Path,
    output: str | Path,
    source_model: str = "Rostlab/prot_t5_xl_half_uniref50-enc",
) -> dict[str, Any]:
    """Import a headerless feature-plus-label CSV after strict FASTA alignment checks."""

    sample_ids, sequences, fasta_labels = _read_labeled_fasta(fasta_path)
    matrix = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.shape[0] != len(sample_ids):
        raise ValueError(
            f"CSV/FASTA sample mismatch: {matrix.shape[0]} versus {len(sample_ids)}."
        )
    if matrix.shape[1] < 2:
        raise ValueError("Feature CSV must contain at least one feature column and one label column.")
    csv_labels = matrix[:, -1].astype(np.int64)
    if not np.array_equal(csv_labels, np.asarray(fasta_labels, dtype=np.int64)):
        mismatch = np.flatnonzero(csv_labels != np.asarray(fasta_labels, dtype=np.int64))[:5]
        raise ValueError(f"CSV labels do not match FASTA order at indices {mismatch.tolist()}.")
    metadata = {
        "source_csv": str(csv_path),
        "source_fasta": str(fasta_path),
        "source_model": source_model,
        "pooling": "official ToxPLTC per-protein embedding",
        "num_samples": len(sample_ids),
        "feature_dim": int(matrix.shape[1] - 1),
    }
    result = {
        "features": torch.from_numpy(matrix[:, :-1].copy()),
        "labels": torch.from_numpy(csv_labels.copy()).long(),
        "sample_ids": sample_ids,
        "sequences": sequences,
        "metadata": metadata,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    return metadata


def embed_prott5_pooled(
    fasta_path: str | Path,
    output: str | Path,
    *,
    model_path: str,
    device_name: str,
    batch_size: int,
    save_every: int,
    pooling: str,
    local_files_only: bool,
) -> dict[str, Any]:
    """Generate compact ProtT5 per-protein vectors with resumable checkpoints."""

    try:
        from transformers import T5EncoderModel, T5Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "ProtT5 extraction requires transformers and sentencepiece."
        ) from exc

    sample_ids, sequences, labels = _read_labeled_fasta(fasta_path)
    output_path = Path(output)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists():
        completed = torch.load(output_path, map_location="cpu", weights_only=False)
        if completed.get("sample_ids") == sample_ids:
            return completed["metadata"]

    features: list[torch.Tensor] = []
    start = 0
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if partial.get("sample_ids") != sample_ids:
            raise ValueError("Partial ProtT5 output belongs to a different FASTA file.")
        features = [row.float() for row in partial["features"]]
        start = len(features)
        if start > len(sample_ids):
            raise ValueError("Partial ProtT5 output contains too many samples.")

    device = resolve_device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = T5Tokenizer.from_pretrained(
        model_path, do_lower_case=False, local_files_only=local_files_only
    )
    model = T5EncoderModel.from_pretrained(
        model_path, torch_dtype=dtype, local_files_only=local_files_only
    )
    model.eval().to(device)

    def save_partial() -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": torch.stack(features) if features else torch.empty((0, 1024)),
                "sample_ids": sample_ids,
                "fasta": str(fasta_path),
                "model_path": model_path,
                "pooling": pooling,
            },
            partial_path,
        )

    with torch.inference_mode():
        for batch_start in range(start, len(sequences), int(batch_size)):
            batch_stop = min(batch_start + int(batch_size), len(sequences))
            batch_sequences = [" ".join(sequence) for sequence in sequences[batch_start:batch_stop]]
            encoded = tokenizer(
                batch_sequences,
                add_special_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            ).last_hidden_state
            for local_index, sequence in enumerate(sequences[batch_start:batch_stop]):
                if pooling == "official_with_eos":
                    valid_length = int(encoded["attention_mask"][local_index].sum().item())
                elif pooling == "residue_mean":
                    valid_length = len(sequence)
                else:
                    raise ValueError("pooling must be 'official_with_eos' or 'residue_mean'.")
                features.append(hidden[local_index, :valid_length].float().mean(dim=0).cpu())
            if batch_stop % int(save_every) == 0 or batch_stop == len(sequences):
                save_partial()
                print(f"embedded={batch_stop}/{len(sequences)}")

    feature_tensor = torch.stack(features)
    metadata = {
        "source_fasta": str(fasta_path),
        "source_model": model_path,
        "pooling": pooling,
        "includes_eos": pooling == "official_with_eos",
        "num_samples": len(sample_ids),
        "feature_dim": int(feature_tensor.shape[1]),
    }
    result = {
        "features": feature_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_ids": sample_ids,
        "sequences": sequences,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    partial_path.unlink(missing_ok=True)
    return metadata


def _nearest_indices_torch(
    query: np.ndarray,
    reference: np.ndarray,
    neighbors: int,
    query_reference_indices: np.ndarray | None = None,
    chunk_size: int = 256,
) -> np.ndarray:
    """Deterministic exact Euclidean k-NN fallback with bounded memory."""

    query_tensor = torch.from_numpy(np.asarray(query, dtype=np.float32))
    reference_tensor = torch.from_numpy(np.asarray(reference, dtype=np.float32))
    extra = 1 if query_reference_indices is not None else 0
    requested = min(int(neighbors) + extra, reference_tensor.shape[0])
    rows = []
    for start in range(0, query_tensor.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), query_tensor.shape[0])
        distances = torch.cdist(query_tensor[start:stop], reference_tensor)
        indices = distances.topk(requested, largest=False, dim=1).indices.cpu().numpy()
        if query_reference_indices is not None:
            filtered = []
            for local_index, candidates in enumerate(indices):
                self_index = int(query_reference_indices[start + local_index])
                kept = [int(value) for value in candidates if int(value) != self_index][:neighbors]
                if len(kept) < neighbors:
                    raise RuntimeError("Could not obtain enough non-self nearest neighbors.")
                filtered.append(kept)
            indices = np.asarray(filtered, dtype=np.int64)
        else:
            indices = indices[:, :neighbors]
        rows.append(indices)
    return np.concatenate(rows, axis=0)


def _nearest_indices(
    query: np.ndarray,
    reference: np.ndarray,
    neighbors: int,
    query_reference_indices: np.ndarray | None = None,
    backend: str = "auto",
) -> np.ndarray:
    if backend not in {"auto", "sklearn", "torch"}:
        raise ValueError("backend must be 'auto', 'sklearn', or 'torch'.")
    if backend in {"auto", "sklearn"}:
        try:
            from sklearn.neighbors import NearestNeighbors

            extra = 1 if query_reference_indices is not None else 0
            requested = min(int(neighbors) + extra, reference.shape[0])
            estimator = NearestNeighbors(n_neighbors=requested, metric="euclidean", n_jobs=-1)
            estimator.fit(reference)
            indices = estimator.kneighbors(query, return_distance=False)
            if query_reference_indices is None:
                return indices[:, :neighbors].astype(np.int64, copy=False)
            filtered = []
            for self_index, candidates in zip(query_reference_indices, indices):
                kept = [int(value) for value in candidates if int(value) != int(self_index)][:neighbors]
                if len(kept) < neighbors:
                    raise RuntimeError("Could not obtain enough non-self nearest neighbors.")
                filtered.append(kept)
            return np.asarray(filtered, dtype=np.int64)
        except ImportError:
            if backend == "sklearn":
                raise
    return _nearest_indices_torch(
        query,
        reference,
        neighbors,
        query_reference_indices=query_reference_indices,
    )


def borderline_smote(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    k_neighbors: int = 4,
    m_neighbors: int = 10,
    seed: int = 42,
    target_ratio: float = 1.0,
    neighbor_backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray, BorderlineSMOTEReport]:
    """Generate borderline-1 minority samples from one training fold only.

    A minority point is considered in danger when at least half, but not all,
    of its ``m_neighbors`` nearest full-training neighbors are majority-class.
    Synthetic points interpolate between a danger point and one of its nearest
    minority neighbors.  Validation or test features must never be passed here.
    """

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("features must be [N,D] and labels must be [N].")
    classes, counts = np.unique(y, return_counts=True)
    if classes.tolist() != [0, 1]:
        raise ValueError("borderline_smote requires binary labels encoded as 0/1.")
    positive_count = int(counts[1])
    negative_count = int(counts[0])
    if positive_count >= negative_count:
        report = BorderlineSMOTEReport(
            positive_count, negative_count, 0, 0, positive_count, negative_count
        )
        return x.copy(), y.copy(), report

    positive_indices = np.flatnonzero(y == 1)
    full_neighbors = _nearest_indices(
        x[positive_indices],
        x,
        int(m_neighbors),
        query_reference_indices=positive_indices,
        backend=neighbor_backend,
    )
    majority_counts = (y[full_neighbors] == 0).sum(axis=1)
    danger_mask = (majority_counts >= math.ceil(m_neighbors / 2)) & (
        majority_counts < m_neighbors
    )
    danger_positive_positions = np.flatnonzero(danger_mask)
    if danger_positive_positions.size == 0:
        raise RuntimeError("Borderline-SMOTE found no positive samples in danger.")

    positive_features = x[positive_indices]
    danger_features = positive_features[danger_positive_positions]
    minority_neighbors = _nearest_indices(
        danger_features,
        positive_features,
        int(k_neighbors),
        query_reference_indices=danger_positive_positions,
        backend=neighbor_backend,
    )

    desired_positive = min(
        negative_count,
        max(positive_count, int(round(negative_count * float(target_ratio)))),
    )
    synthetic_count = max(0, desired_positive - positive_count)
    rng = np.random.default_rng(int(seed))
    selected_danger = rng.integers(0, danger_features.shape[0], size=synthetic_count)
    selected_neighbor_slot = rng.integers(0, int(k_neighbors), size=synthetic_count)
    interpolation = rng.random((synthetic_count, 1), dtype=np.float32)
    starts = danger_features[selected_danger]
    neighbor_rows = minority_neighbors[selected_danger, selected_neighbor_slot]
    ends = positive_features[neighbor_rows]
    synthetic = starts + interpolation * (ends - starts)

    output_x = np.concatenate([x, synthetic.astype(np.float32, copy=False)], axis=0)
    output_y = np.concatenate([y, np.ones(synthetic_count, dtype=np.int64)], axis=0)
    report = BorderlineSMOTEReport(
        original_positive=positive_count,
        original_negative=negative_count,
        danger_positive=int(danger_positive_positions.size),
        synthetic_positive=int(synthetic_count),
        output_positive=int((output_y == 1).sum()),
        output_negative=int((output_y == 0).sum()),
    )
    return output_x, output_y, report


def _probability_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= float(threshold)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denominator = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))

    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and probabilities[order[stop]] == probabilities[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    auroc = (
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / max(positives * negatives, 1)
    )
    descending = np.argsort(-probabilities, kind="mergesort")
    sorted_labels = labels[descending]
    cumulative_positive = np.cumsum(sorted_labels)
    positive_ranks = np.flatnonzero(sorted_labels == 1)
    auprc = float(
        (cumulative_positive[positive_ranks] / (positive_ranks + 1)).sum() / max(positives, 1)
    )
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": (tp * tn - fp * fn) / denominator,
        "auroc": float(auroc),
        "auprc": auprc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "threshold": float(threshold),
    }


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> tuple[float, dict[str, float]]:
    candidates = np.unique(np.concatenate([probabilities, np.linspace(0.01, 0.99, 99)]))
    best_threshold = 0.5
    best_metrics = _probability_metrics(labels, probabilities, best_threshold)
    for threshold in candidates:
        metrics = _probability_metrics(labels, probabilities, float(threshold))
        score = (float(metrics[metric]), float(metrics["mcc"]))
        best_score = (float(best_metrics[metric]), float(best_metrics["mcc"]))
        if score > best_score:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _predict(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    probabilities = []
    tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    with torch.no_grad():
        for chunk in tensor.split(int(batch_size)):
            probabilities.append(torch.sigmoid(model(chunk.to(device))).cpu())
    return torch.cat(probabilities).numpy()


def predict_fold_ensemble(
    features_path: str | Path,
    checkpoint_dir: str | Path,
    output_csv: str | Path,
    *,
    device_name: str,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    """Average the five fixed-fold TextCNN checkpoints on an aligned split."""

    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].long().numpy()
    sample_ids = list(payload["sample_ids"])
    sequences = list(payload["sequences"])
    device = resolve_device(device_name)
    checkpoint_path = Path(checkpoint_dir)
    fold_probabilities = []
    checkpoints = []
    for fold in range(5):
        path = checkpoint_path / f"fold{fold}_best_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_config = dict(checkpoint["model"])
        model_config["window_sizes"] = tuple(model_config["window_sizes"])
        model = PooledPLMTextCNN(**model_config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        fold_probabilities.append(_predict(model, features, batch_size, device))
        checkpoints.append(str(path))
    probability_matrix = np.stack(fold_probabilities, axis=0)
    probabilities = probability_matrix.mean(axis=0)
    metrics = _probability_metrics(labels, probabilities, threshold=float(threshold))
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_index",
            "sample_id",
            "sequence",
            "label",
            "toxicity_probability",
            "prediction",
            "decision_threshold",
        ] + [f"fold{fold}_probability" for fold in range(5)]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, probability in enumerate(probabilities):
            row: dict[str, Any] = {
                "source_index": index,
                "sample_id": sample_ids[index],
                "sequence": sequences[index],
                "label": int(labels[index]),
                "toxicity_probability": f"{float(probability):.10g}",
                "prediction": int(probability >= threshold),
                "decision_threshold": f"{float(threshold):.10g}",
            }
            for fold in range(5):
                row[f"fold{fold}_probability"] = f"{float(probability_matrix[fold, index]):.10g}"
            writer.writerow(row)
    summary = {
        "protocol": {
            "features": str(features_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoints": checkpoints,
            "ensemble": "arithmetic mean of five fixed-fold probabilities",
            "threshold": float(threshold),
            "feature_metadata": payload.get("metadata", {}),
        },
        "metrics": metrics,
    }
    save_json(summary, output_path.with_suffix(".metrics.json"))
    return summary


def ensemble_prediction_files(
    input_paths: list[str | Path],
    output_csv: str | Path,
    *,
    threshold: float,
) -> dict[str, Any]:
    """Average aligned probability CSVs while preserving OOF fold metadata."""

    if len(input_paths) < 2:
        raise ValueError("At least two prediction files are required for an ensemble.")
    tables = []
    for path in input_paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Prediction file is empty: {path}")
        tables.append(rows)
    reference = tables[0]
    for table_index, table in enumerate(tables[1:], start=1):
        if len(table) != len(reference):
            raise ValueError(
                f"Prediction row counts differ for input 0 and input {table_index}."
            )
        for row_index, (left, right) in enumerate(zip(reference, table)):
            for key in ("sample_id", "sequence", "label"):
                if left[key] != right[key]:
                    raise ValueError(
                        f"Prediction alignment mismatch at input {table_index}, "
                        f"row {row_index}, column {key}."
                    )
    probability_matrix = np.asarray(
        [
            [float(row["toxicity_probability"]) for row in table]
            for table in tables
        ],
        dtype=np.float64,
    )
    probabilities = probability_matrix.mean(axis=0)
    labels = np.asarray([int(row["label"]) for row in reference], dtype=np.int64)
    metrics = _probability_metrics(labels, probabilities, threshold=float(threshold))
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    include_fold = "fold" in reference[0]
    fieldnames = ["source_index"]
    if include_fold:
        fieldnames.append("fold")
    fieldnames.extend(
        [
            "sample_id",
            "sequence",
            "label",
            "toxicity_probability",
            "prediction",
            "decision_threshold",
        ]
    )
    fieldnames.extend(f"member{index}_probability" for index in range(len(tables)))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, probability) in enumerate(zip(reference, probabilities)):
            output_row: dict[str, Any] = {
                "source_index": row.get("source_index", index),
                "sample_id": row["sample_id"],
                "sequence": row["sequence"],
                "label": int(labels[index]),
                "toxicity_probability": f"{float(probability):.10g}",
                "prediction": int(probability >= threshold),
                "decision_threshold": f"{float(threshold):.10g}",
            }
            if include_fold:
                output_row["fold"] = row["fold"]
            for member_index in range(len(tables)):
                output_row[f"member{member_index}_probability"] = (
                    f"{float(probability_matrix[member_index, index]):.10g}"
                )
            writer.writerow(output_row)
    summary = {
        "protocol": {
            "inputs": [str(path) for path in input_paths],
            "ensemble": "unweighted arithmetic mean of aligned probabilities",
            "threshold": float(threshold),
            "threshold_selected_from_test_labels": False,
        },
        "metrics": metrics,
    }
    save_json(summary, output_path.with_suffix(".metrics.json"))
    return summary


def _train_fold(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    augmentation: str,
    neighbor_backend: str,
    smote_target_ratio: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    smote_report: BorderlineSMOTEReport | None = None
    fit_features = np.asarray(train_features, dtype=np.float32)
    fit_labels = np.asarray(train_labels, dtype=np.int64)
    if augmentation == "borderline_smote":
        fit_features, fit_labels, smote_report = borderline_smote(
            fit_features,
            fit_labels,
            seed=seed,
            target_ratio=smote_target_ratio,
            neighbor_backend=neighbor_backend,
        )
    elif augmentation != "none":
        raise ValueError("augmentation must be 'none' or 'borderline_smote'.")

    model = PooledPLMTextCNN(input_dim=fit_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(fit_features),
            torch.from_numpy(fit_labels.astype(np.float32)),
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_labels.shape[0]
            total_items += int(batch_labels.shape[0])
        validation_probabilities = _predict(model, validation_features, batch_size, device)
        validation_metrics = _probability_metrics(validation_labels, validation_probabilities, 0.5)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_items, 1),
                "validation": validation_metrics,
            }
        )
        score = (validation_metrics["balanced_accuracy"], validation_metrics["mcc"])
        best_score = (
            (-float("inf"), -float("inf"))
            if best_metrics is None
            else (best_metrics["balanced_accuracy"], best_metrics["mcc"])
        )
        if score > best_score:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = validation_metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    probabilities = _predict(model, validation_features, batch_size, device)
    checkpoint = {
        "model_state": best_state,
        "model": {
            "input_dim": int(fit_features.shape[1]),
            "projection_dim": 128,
            "window_sizes": [4, 5, 6],
            "num_filters": 64,
            "dropout": 0.2,
        },
        "augmentation": augmentation,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_metrics_at_0_5": best_metrics,
        "smote_report": None if smote_report is None else smote_report.__dict__,
    }
    details = {"history": history, "smote_report": checkpoint["smote_report"]}
    return checkpoint, probabilities, details


def cross_validate(
    features_path: str | Path,
    fold_manifest: str | Path,
    output_dir: str | Path,
    *,
    augmentation: str,
    device_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    neighbor_backend: str,
    smote_target_ratio: float,
) -> dict[str, Any]:
    # Fold utilities depend on scikit-learn.  Import them only for CV so the
    # feature-extraction command stays usable in minimal inference environments.
    from ghxtox.folds import load_fold_indices

    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    features = payload["features"].float().numpy()
    labels = payload["labels"].long().numpy()
    sample_ids = list(payload["sample_ids"])
    sequences = list(payload["sequences"])
    device = resolve_device(device_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    oof_probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    fold_results = []

    for fold in range(5):
        train_indices, validation_indices = load_fold_indices(
            fold_manifest, fold, len(labels)
        )
        checkpoint, probabilities, details = _train_fold(
            features[np.asarray(train_indices)],
            labels[np.asarray(train_indices)],
            features[np.asarray(validation_indices)],
            labels[np.asarray(validation_indices)],
            seed=int(seed),
            device=device,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            patience=int(patience),
            augmentation=augmentation,
            neighbor_backend=neighbor_backend,
            smote_target_ratio=smote_target_ratio,
        )
        oof_probabilities[np.asarray(validation_indices)] = probabilities
        torch.save(checkpoint, output_path / f"fold{fold}_best_model.pt")
        save_json(details, output_path / f"fold{fold}_history.json")
        fold_metrics = _probability_metrics(
            labels[np.asarray(validation_indices)], probabilities, 0.5
        )
        fold_results.append(
            {
                "fold": fold,
                "num_train": len(train_indices),
                "num_validation": len(validation_indices),
                "best_epoch": checkpoint["best_epoch"],
                "metrics_at_0_5": fold_metrics,
                "smote_report": checkpoint["smote_report"],
            }
        )
        print(
            f"fold={fold} epoch={checkpoint['best_epoch']} "
            f"bacc={fold_metrics['balanced_accuracy']:.4f} "
            f"mcc={fold_metrics['mcc']:.4f} auprc={fold_metrics['auprc']:.4f}"
        )

    if np.isnan(oof_probabilities).any():
        raise RuntimeError("At least one training sample is missing an OOF prediction.")
    metrics_at_half = _probability_metrics(labels, oof_probabilities, 0.5)
    threshold, optimized_metrics = _best_threshold(
        labels, oof_probabilities, metric="balanced_accuracy"
    )
    with (output_path / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_index", "sample_id", "sequence", "label", "toxicity_probability"],
        )
        writer.writeheader()
        for index, probability in enumerate(oof_probabilities):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": sample_ids[index],
                    "sequence": sequences[index],
                    "label": int(labels[index]),
                    "toxicity_probability": f"{float(probability):.10g}",
                }
            )
    summary = {
        "protocol": {
            "features": str(features_path),
            "fold_manifest": str(fold_manifest),
            "augmentation": augmentation,
            "augmentation_scope": "training partition of each fold only",
            "checkpoint_selection": "validation balanced_accuracy at threshold 0.5; MCC tie-break",
            "seed": int(seed),
            "epochs": int(epochs),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "neighbor_backend": neighbor_backend,
            "smote_target_ratio": float(smote_target_ratio),
            "feature_metadata": payload.get("metadata", {}),
        },
        "folds": fold_results,
        "oof_at_0_5": metrics_at_half,
        "oof_bacc_threshold": float(threshold),
        "oof_at_bacc_threshold": optimized_metrics,
    }
    save_json(summary, output_path / "summary.json")
    print(json.dumps(summary["oof_at_bacc_threshold"], indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-controlled pooled-PLM TextCNN experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Mean-pool cached residue PLM features.")
    extract.add_argument("--processed", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--source-model", default="ESM2-650M")

    import_csv = subparsers.add_parser(
        "import-csv", help="Import a headerless pooled-feature CSV with strict FASTA checks."
    )
    import_csv.add_argument("--csv", required=True)
    import_csv.add_argument("--fasta", required=True)
    import_csv.add_argument("--output", required=True)
    import_csv.add_argument(
        "--source-model", default="Rostlab/prot_t5_xl_half_uniref50-enc"
    )

    embed_prott5 = subparsers.add_parser(
        "embed-prott5", help="Generate compact, resumable ProtT5 per-protein features."
    )
    embed_prott5.add_argument("--fasta", required=True)
    embed_prott5.add_argument("--output", required=True)
    embed_prott5.add_argument(
        "--model-path", default="Rostlab/prot_t5_xl_half_uniref50-enc"
    )
    embed_prott5.add_argument("--device", default="cuda")
    embed_prott5.add_argument("--batch-size", type=int, default=1)
    embed_prott5.add_argument("--save-every", type=int, default=100)
    embed_prott5.add_argument(
        "--pooling", choices=["official_with_eos", "residue_mean"], default="official_with_eos"
    )
    embed_prott5.add_argument("--local-files-only", action="store_true")

    cv = subparsers.add_parser("cv", help="Run fixed five-fold group-aware CV.")
    cv.add_argument("--features", required=True)
    cv.add_argument("--fold-manifest", required=True)
    cv.add_argument("--output-dir", required=True)
    cv.add_argument("--augmentation", choices=["none", "borderline_smote"], default="none")
    cv.add_argument("--device", default="auto")
    cv.add_argument("--epochs", type=int, default=150)
    cv.add_argument("--batch-size", type=int, default=32)
    cv.add_argument("--learning-rate", type=float, default=5e-4)
    cv.add_argument("--weight-decay", type=float, default=0.0)
    cv.add_argument("--patience", type=int, default=20)
    cv.add_argument("--seed", type=int, default=42)
    cv.add_argument("--neighbor-backend", choices=["auto", "sklearn", "torch"], default="auto")
    cv.add_argument("--smote-target-ratio", type=float, default=1.0)
    predict = subparsers.add_parser("predict", help="Predict with the five-fold checkpoint ensemble.")
    predict.add_argument("--features", required=True)
    predict.add_argument("--checkpoint-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--batch-size", type=int, default=256)
    predict.add_argument("--threshold", type=float, default=0.35)
    ensemble = subparsers.add_parser("ensemble", help="Average aligned prediction CSV files.")
    ensemble.add_argument("--inputs", nargs="+", required=True)
    ensemble.add_argument("--output", required=True)
    ensemble.add_argument("--threshold", type=float, required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "extract":
        metadata = extract_pooled_features(args.processed, args.output, args.source_model)
        print(json.dumps(metadata, indent=2))
        return
    if args.command == "import-csv":
        metadata = import_feature_csv(args.csv, args.fasta, args.output, args.source_model)
        print(json.dumps(metadata, indent=2))
        return
    if args.command == "embed-prott5":
        metadata = embed_prott5_pooled(
            args.fasta,
            args.output,
            model_path=args.model_path,
            device_name=args.device,
            batch_size=args.batch_size,
            save_every=args.save_every,
            pooling=args.pooling,
            local_files_only=args.local_files_only,
        )
        print(json.dumps(metadata, indent=2))
        return
    if args.command == "predict":
        summary = predict_fold_ensemble(
            args.features,
            args.checkpoint_dir,
            args.output,
            device_name=args.device,
            batch_size=args.batch_size,
            threshold=args.threshold,
        )
        print(json.dumps(summary["metrics"], indent=2))
        return
    if args.command == "ensemble":
        summary = ensemble_prediction_files(
            args.inputs,
            args.output,
            threshold=args.threshold,
        )
        print(json.dumps(summary["metrics"], indent=2))
        return
    cross_validate(
        features_path=args.features,
        fold_manifest=args.fold_manifest,
        output_dir=args.output_dir,
        augmentation=args.augmentation,
        device_name=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        neighbor_backend=args.neighbor_backend,
        smote_target_ratio=args.smote_target_ratio,
    )


if __name__ == "__main__":
    main()
