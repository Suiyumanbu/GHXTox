"""General project helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch


DEFAULT_DEVICE = "cuda"
DEFAULT_TRAIN_FASTA = "dataset/train_data or benchmark_data.fasta"
DEFAULT_TEST_FASTA = "dataset/test1.fasta"
DEFAULT_TRAIN_PROCESSED = "data/processed/train_cached_func_esm2.pt"
DEFAULT_TEST_PROCESSED = "data/processed/test1_cached_func_esm2.pt"
DEFAULT_CHECKPOINT = "runs/plm_fusion_esm2_geometry_confidence/best_model.pt"
DEFAULT_STRUCTURE_CACHE_DIR = "data/structures"
DEFAULT_THRESHOLD = 0.85


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return torch.device("cuda")
    return torch.device(requested)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved

