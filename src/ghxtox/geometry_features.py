"""Geometry descriptors derived from residue C-alpha coordinates."""

from __future__ import annotations

import torch

from ghxtox.constants import AROMATIC_AA, HYDROPATHY, NEGATIVE_AA, POLAR_AA, POSITIVE_AA, SULFUR_AA
from ghxtox.features import AMIDE_AA, CARBOXYLATE_AA, HYDROXYL_AA, clean_sequence

STRUCTURE_FEATURE_DIM = 16
CHEMICAL_STRUCTURE_FEATURE_DIM = 28


def _safe_normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def _torsion_sin_cos(points: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    p0 = points[index - 1]
    p1 = points[index]
    p2 = points[index + 1]
    p3 = points[index + 2]

    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_unit = _safe_normalize(b1)
    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    v = _safe_normalize(v)
    w = _safe_normalize(w)

    x = (v * w).sum().clamp(-1.0, 1.0)
    y = (torch.cross(b1_unit, v, dim=-1) * w).sum().clamp(-1.0, 1.0)
    return y, x


def structure_feature_matrix(coords: torch.Tensor, plddt: torch.Tensor) -> torch.Tensor:
    """Return per-residue C-alpha geometry descriptors.

    Features are scale-normalized and deterministic. They are intentionally
    lightweight because peptide structures are short and often predicted rather
    than experimentally resolved.
    """

    coords = coords.float()
    plddt = plddt.float().flatten()
    length = int(coords.shape[0])
    if length == 0:
        return torch.zeros((0, STRUCTURE_FEATURE_DIM), dtype=torch.float32)

    if coords.shape[-1] != 3:
        raise ValueError(f"Expected coords shape [L, 3], got {tuple(coords.shape)}.")
    if plddt.shape[0] != length:
        raise ValueError(f"Expected plddt length {length}, got {plddt.shape[0]}.")

    confidence = plddt.clamp(0.0, 1.0)
    distances = torch.cdist(coords, coords).clamp_min(0.0)
    eye = torch.eye(length, dtype=torch.bool, device=coords.device)
    pair_mask = ~eye
    denom = max(length - 1, 1)

    centroid = coords.mean(dim=0, keepdim=True)
    radial = (coords - centroid).norm(dim=-1)
    radial_norm = radial / radial.max().clamp_min(1.0)

    contact_6 = ((distances < 6.0) & pair_mask).float().sum(dim=1) / denom
    contact_8 = ((distances < 8.0) & pair_mask).float().sum(dim=1) / denom
    contact_10 = ((distances < 10.0) & pair_mask).float().sum(dim=1) / denom
    soft_density = (torch.exp(-distances / 8.0) * pair_mask.float()).sum(dim=1) / denom

    masked_dist = distances.masked_fill(~pair_mask, 1e6)
    k = min(4, max(length - 1, 1))
    knn_mean = masked_dist.topk(k=k, largest=False, dim=1).values.mean(dim=1) / 20.0

    positions = torch.arange(length, device=coords.device)
    seq_sep = (positions.view(-1, 1) - positions.view(1, -1)).abs()
    nonlocal_mask = pair_mask & (seq_sep > 2)
    nearest_nonlocal = distances.masked_fill(~nonlocal_mask, 1e6).min(dim=1).values
    nearest_nonlocal = torch.where(nearest_nonlocal >= 1e5, torch.full_like(nearest_nonlocal, 20.0), nearest_nonlocal)
    nearest_nonlocal = nearest_nonlocal / 20.0

    local_dist_sum = torch.zeros(length, dtype=torch.float32, device=coords.device)
    local_dist_count = torch.zeros(length, dtype=torch.float32, device=coords.device)
    for offset in (1, 2):
        values = (coords[:-offset] - coords[offset:]).norm(dim=-1)
        local_dist_sum[:-offset] += values
        local_dist_count[:-offset] += 1.0
        local_dist_sum[offset:] += values
        local_dist_count[offset:] += 1.0
    local_distance = local_dist_sum / local_dist_count.clamp_min(1.0) / 8.0

    angle_cos = torch.zeros(length, dtype=torch.float32, device=coords.device)
    torsion_sin = torch.zeros(length, dtype=torch.float32, device=coords.device)
    torsion_cos = torch.zeros(length, dtype=torch.float32, device=coords.device)
    for index in range(1, length - 1):
        left = _safe_normalize(coords[index - 1] - coords[index])
        right = _safe_normalize(coords[index + 1] - coords[index])
        angle_cos[index] = (left * right).sum().clamp(-1.0, 1.0)
    for index in range(1, length - 2):
        torsion_sin[index], torsion_cos[index] = _torsion_sin_cos(coords, index)

    plddt_window_mean = torch.zeros(length, dtype=torch.float32, device=coords.device)
    plddt_window_min = torch.zeros(length, dtype=torch.float32, device=coords.device)
    for index in range(length):
        left = max(0, index - 2)
        right = min(length, index + 3)
        window = confidence[left:right]
        plddt_window_mean[index] = window.mean()
        plddt_window_min[index] = window.min()

    rows = torch.stack(
        [
            confidence,
            (confidence < 0.55).float(),
            plddt_window_mean,
            plddt_window_min,
            radial_norm,
            contact_6,
            contact_8,
            contact_10,
            soft_density,
            knn_mean.clamp(0.0, 5.0),
            nearest_nonlocal.clamp(0.0, 5.0),
            local_distance.clamp(0.0, 5.0),
            angle_cos,
            torsion_sin,
            torsion_cos,
            radial / max(float(length), 1.0),
        ],
        dim=-1,
    )
    return rows.cpu()


def chemical_structure_feature_matrix(sequence: str, coords: torch.Tensor, plddt: torch.Tensor) -> torch.Tensor:
    """Return geometry descriptors plus spatial chemical-neighborhood descriptors."""

    sequence = clean_sequence(sequence)
    base = structure_feature_matrix(coords, plddt)
    length = base.shape[0]
    if length == 0:
        return torch.zeros((0, CHEMICAL_STRUCTURE_FEATURE_DIM), dtype=torch.float32)
    if len(sequence) != length:
        sequence = sequence[:length].ljust(length, "X")

    coords = coords.float()
    distances = torch.cdist(coords, coords).clamp_min(0.0)
    eye = torch.eye(length, dtype=torch.bool, device=coords.device)
    neighbor_mask = (distances < 8.0) & ~eye
    soft_weight = torch.exp(-distances / 8.0) * (~eye).float()
    hard_count = neighbor_mask.float().sum(dim=1).clamp_min(1.0)
    soft_count = soft_weight.sum(dim=1).clamp_min(1e-6)

    def indicator(values: set[str]) -> torch.Tensor:
        return torch.tensor([1.0 if aa in values else 0.0 for aa in sequence], dtype=torch.float32, device=coords.device)

    charge = torch.tensor(
        [1.0 if aa in POSITIVE_AA else -1.0 if aa in NEGATIVE_AA else 0.0 for aa in sequence],
        dtype=torch.float32,
        device=coords.device,
    )
    hydropathy = torch.tensor([HYDROPATHY.get(aa, 0.0) / 4.5 for aa in sequence], dtype=torch.float32, device=coords.device)
    positive = indicator(POSITIVE_AA)
    negative = indicator(NEGATIVE_AA)
    polar = indicator(POLAR_AA)
    aromatic = indicator(AROMATIC_AA)
    sulfur = indicator(SULFUR_AA)
    hydroxyl = indicator(HYDROXYL_AA)
    amide = indicator(AMIDE_AA)
    carboxylate = indicator(CARBOXYLATE_AA)

    def hard_fraction(values: torch.Tensor) -> torch.Tensor:
        return (neighbor_mask.float() @ values) / hard_count

    def soft_average(values: torch.Tensor) -> torch.Tensor:
        return (soft_weight @ values) / soft_count

    chemical = torch.stack(
        [
            hard_fraction(positive),
            hard_fraction(negative),
            hard_fraction(polar),
            hard_fraction(aromatic),
            hard_fraction(sulfur),
            hard_fraction(hydroxyl),
            hard_fraction(amide),
            hard_fraction(carboxylate),
            soft_average(charge),
            soft_average(hydropathy),
            soft_average(positive - negative),
            hard_fraction((hydropathy > 0).float()),
        ],
        dim=-1,
    )
    return torch.cat([base, chemical.cpu()], dim=-1)
