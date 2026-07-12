"""Structure providers for pseudo or cached peptide geometries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import torch

from ghxtox.features import clean_sequence


@dataclass(frozen=True)
class StructureData:
    coords: torch.Tensor
    plddt: torch.Tensor
    source: str
    backbone_coords: torch.Tensor | None = None
    backbone_mask: torch.Tensor | None = None


def _synthesize_backbone(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    backbone_coords = coords.unsqueeze(1).expand(-1, 5, -1).clone()
    backbone_mask = torch.zeros(coords.shape[0], 5, dtype=torch.bool, device=coords.device)
    if coords.shape[0] > 0:
        backbone_mask[:, 1] = True
    return backbone_coords, backbone_mask


class StructureProvider:
    def get(self, sample_id: str, sequence: str) -> StructureData:
        raise NotImplementedError


class HeuristicStructureProvider(StructureProvider):
    """Deterministic fallback geometry for pipeline development."""

    def get(self, sample_id: str, sequence: str) -> StructureData:
        sequence = clean_sequence(sequence)
        coords = []
        plddt = []
        length = max(len(sequence), 1)
        for i, aa in enumerate(sequence):
            angle = i * 1.745
            radius = 2.1 + 0.15 * ((ord(aa) % 5) - 2)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = i * 1.48
            coords.append([x, y, z])

            edge_penalty = 0.08 * (math.exp(-i / 4.0) + math.exp(-(length - 1 - i) / 4.0))
            cysteine_bonus = 0.03 if aa == "C" else 0.0
            plddt.append(max(0.35, min(0.88, 0.72 - edge_penalty + cysteine_bonus)))

        tensor = torch.tensor(coords, dtype=torch.float32)
        tensor = tensor - tensor.mean(dim=0, keepdim=True)
        backbone_coords, backbone_mask = _synthesize_backbone(tensor)
        return StructureData(
            coords=tensor,
            plddt=torch.tensor(plddt, dtype=torch.float32),
            source="heuristic",
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
        )


class CachedStructureProvider(StructureProvider):
    """Load ESMFold-style structures saved as per-sample NPZ/PT files."""

    def __init__(
        self,
        cache_dir: str | Path,
        fallback: StructureProvider | None = None,
        allow_fallback: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.fallback = fallback or HeuristicStructureProvider()
        self.allow_fallback = allow_fallback

    def get(self, sample_id: str, sequence: str) -> StructureData:
        for suffix in (".pt", ".pth", ".npz"):
            path = self.cache_dir / f"{sample_id}{suffix}"
            if path.exists():
                return self._load(path)
        if self.allow_fallback:
            return self.fallback.get(sample_id, sequence)
        raise FileNotFoundError(f"Missing cached structure for {sample_id} in {self.cache_dir}")

    def _load(self, path: Path) -> StructureData:
        if path.suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            coords = torch.as_tensor(payload["coords"], dtype=torch.float32)
            plddt = torch.as_tensor(payload["plddt"], dtype=torch.float32)
            backbone_coords_raw = payload.get("backbone_coords")
            backbone_mask_raw = payload.get("backbone_mask")
        elif path.suffix == ".npz":
            import numpy as np

            payload = np.load(path)
            coords = torch.as_tensor(payload["coords"], dtype=torch.float32)
            plddt = torch.as_tensor(payload["plddt"], dtype=torch.float32)
            backbone_coords_raw = payload["backbone_coords"] if "backbone_coords" in payload.files else None
            backbone_mask_raw = payload["backbone_mask"] if "backbone_mask" in payload.files else None
        else:
            raise ValueError(f"Unsupported structure file: {path}")

        if plddt.numel() and float(plddt.max()) > 1.5:
            plddt = plddt / 100.0
        if backbone_coords_raw is None or backbone_mask_raw is None:
            backbone_coords, backbone_mask = _synthesize_backbone(coords)
        else:
            backbone_coords = torch.as_tensor(backbone_coords_raw, dtype=torch.float32)
            backbone_mask = torch.as_tensor(backbone_mask_raw, dtype=torch.bool)
        return StructureData(
            coords=coords,
            plddt=plddt.clamp(0.0, 1.0),
            source=str(path),
            backbone_coords=backbone_coords,
            backbone_mask=backbone_mask,
        )


def make_structure_provider(
    mode: str,
    cache_dir: str | Path | None = None,
    allow_cached_fallback: bool = False,
) -> StructureProvider:
    mode = mode.lower()
    if mode == "heuristic":
        return HeuristicStructureProvider()
    if mode == "cached":
        if cache_dir is None:
            raise ValueError("cache_dir is required when structure_mode='cached'")
        return CachedStructureProvider(cache_dir, allow_fallback=allow_cached_fallback)
    raise ValueError(f"Unknown structure mode: {mode}")
