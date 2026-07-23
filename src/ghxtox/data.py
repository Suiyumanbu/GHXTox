"""Dataset and padding utilities for tensorized peptide records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_DIM, MAX_CHEMICAL_SITES
from ghxtox.constants import AA_TO_IDX
from ghxtox.features import RESIDUE_FEATURE_DIM, sequence_global_features
from ghxtox.geometry_features import (
    CHEMICAL_STRUCTURE_FEATURE_DIM,
    STRUCTURE_FEATURE_DIM,
    chemical_structure_feature_matrix,
    structure_feature_matrix,
)
from ghxtox.conformer_features import (
    CONFORMER_GLOBAL_FEATURE_DIM,
    CONFORMER_RESIDUE_FEATURE_DIM,
)


GLOBAL_FEATURE_DIM = 5
GLOBAL_FEATURE_KEYS = (
    "length",
    "net_charge",
    "aromatic_fraction",
    "cysteine_fraction",
    "mean_hydropathy",
)


def _infer_plm_feature_dim(batch: list[dict[str, Any]]) -> int:
    for item in batch:
        features = item.get("plm_features")
        if torch.is_tensor(features) and features.ndim == 2:
            return int(features.shape[1])
    return 0


def infer_records_plm_feature_dim(records: list[dict[str, Any]]) -> int:
    for item in records:
        features = item.get("plm_features")
        if torch.is_tensor(features) and features.ndim == 2:
            return int(features.shape[1])
    return 0


def validate_plm_feature_dim(records: list[dict[str, Any]], required_dim: int, source: str | Path) -> None:
    required_dim = max(int(required_dim), 0)
    if required_dim <= 0:
        return
    actual_dim = infer_records_plm_feature_dim(records)
    if actual_dim == 0:
        raise ValueError(
            f"Config requires plm_embedding_dim={required_dim}, but {source} has no plm_features. "
            "Use the *_esm2.pt processed files or run `python -m ghxtox.plm_embed` first."
        )
    if actual_dim != required_dim:
        raise ValueError(
            f"Config requires plm_embedding_dim={required_dim}, but {source} contains "
            f"plm_features with dim={actual_dim}."
        )


class PeptideTensorDataset(Dataset):
    def __init__(self, path: str | Path, require_labels: bool = False) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.records: list[dict[str, Any]] = payload["records"]
        self.plm_feature_dim = infer_records_plm_feature_dim(self.records)
        if require_labels:
            missing = [item["sample_id"] for item in self.records if item["label"] is None]
            if missing:
                raise ValueError(f"Missing labels for {len(missing)} records, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _pad_1d(tensor: torch.Tensor, length: int, value: float | int = 0) -> torch.Tensor:
    out = tensor.new_full((length,), value)
    out[: tensor.shape[0]] = tensor
    return out


def _pad_2d(tensor: torch.Tensor, length: int, dim: int, value: float | int | bool = 0.0) -> torch.Tensor:
    out = tensor.new_full((length, dim), value)
    rows = min(tensor.shape[0], length)
    cols = min(tensor.shape[1], dim)
    out[:rows, :cols] = tensor[:rows, :cols]
    return out


def _pad_3d(
    tensor: torch.Tensor,
    length: int,
    dim1: int,
    dim2: int,
    value: float | int | bool = 0.0,
) -> torch.Tensor:
    out = tensor.new_full((length, dim1, dim2), value)
    rows = min(tensor.shape[0], length)
    cols1 = min(tensor.shape[1], dim1)
    cols2 = min(tensor.shape[2], dim2)
    out[:rows, :cols1, :cols2] = tensor[:rows, :cols1, :cols2]
    return out


def _global_feature_tensor(item: dict[str, Any]) -> torch.Tensor:
    sequence = item.get("sequence", "")
    features = sequence_global_features(sequence) if sequence else {}
    features.update(item.get("global_features") or {})
    length = float(features.get("length", item["aa_ids"].shape[0]))
    length = max(length, 1.0)
    values = [
        length / 128.0,
        float(features.get("net_charge", 0.0)) / length,
        float(features.get("aromatic_fraction", 0.0)),
        float(features.get("cysteine_fraction", 0.0)),
        float(features.get("mean_hydropathy", 0.0)) / 4.5,
    ]
    return torch.tensor(values, dtype=torch.float32)


def collate_peptides(
    batch: list[dict[str, Any]],
    include_structure: bool = True,
    include_atom: bool = True,
) -> dict[str, Any]:
    max_len = max(item["aa_ids"].shape[0] for item in batch)
    residue_dim = RESIDUE_FEATURE_DIM
    structure_dim = max(
        max(
            (
                int(item["structure_features"].shape[1])
                for item in batch
                if torch.is_tensor(item.get("structure_features"))
            ),
            default=0,
        ),
        STRUCTURE_FEATURE_DIM,
    )
    plm_dim = _infer_plm_feature_dim(batch)

    aa_ids = torch.stack([_pad_1d(item["aa_ids"], max_len, AA_TO_IDX["<pad>"]) for item in batch])
    group_ids = torch.stack([_pad_1d(item["group_ids"], max_len, 0) for item in batch])
    residue_features = torch.stack(
        [_pad_2d(item["residue_features"], max_len, residue_dim) for item in batch]
    )
    plddt = torch.stack([_pad_1d(item["plddt"], max_len, 0.0) for item in batch])
    global_features = torch.stack([_global_feature_tensor(item) for item in batch])
    lengths = torch.tensor([item["aa_ids"].shape[0] for item in batch], dtype=torch.long)
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    labels = [item["label"] for item in batch]
    if all(label is not None for label in labels):
        label_tensor = torch.tensor(labels, dtype=torch.float32)
    else:
        label_tensor = None

    result = {
        "sample_id": [item["sample_id"] for item in batch],
        "sequence": [item["sequence"] for item in batch],
        "aa_ids": aa_ids,
        "group_ids": group_ids,
        "residue_features": residue_features,
        "plddt": plddt,
        "global_features": global_features,
        "lengths": lengths,
        "mask": mask,
        "labels": label_tensor,
    }
    conformer_residue_features = []
    conformer_global_features = []
    conformer_available = []
    for item in batch:
        length = int(item["aa_ids"].shape[0])
        residue = item.get("conformer_residue_features")
        global_value = item.get("conformer_global_features")
        available = bool(item.get("conformer_available", False))
        if not torch.is_tensor(residue):
            residue = torch.zeros(length, CONFORMER_RESIDUE_FEATURE_DIM, dtype=torch.float32)
            available = False
        if not torch.is_tensor(global_value):
            global_value = torch.zeros(CONFORMER_GLOBAL_FEATURE_DIM, dtype=torch.float32)
            available = False
        conformer_residue_features.append(
            _pad_2d(residue.float(), max_len, CONFORMER_RESIDUE_FEATURE_DIM)
        )
        conformer_global_features.append(global_value.float()[:CONFORMER_GLOBAL_FEATURE_DIM])
        conformer_available.append(available)
    result["conformer_residue_features"] = torch.stack(conformer_residue_features)
    result["conformer_global_features"] = torch.stack(conformer_global_features)
    result["conformer_available"] = torch.tensor(conformer_available, dtype=torch.bool)
    if include_structure:
        result["coords"] = torch.stack([_pad_2d(item["coords"], max_len, 3) for item in batch])
        backbone_coords = []
        backbone_mask = []
        for item in batch:
            item_coords = item["coords"].float()
            item_backbone_coords = item.get("backbone_coords")
            item_backbone_mask = item.get("backbone_mask")
            if not torch.is_tensor(item_backbone_coords):
                item_backbone_coords = item_coords.unsqueeze(1).expand(-1, 5, -1).clone()
            else:
                item_backbone_coords = item_backbone_coords.float()
            if not torch.is_tensor(item_backbone_mask):
                item_backbone_mask = torch.zeros(item_backbone_coords.shape[:2], dtype=torch.bool)
                if item_backbone_mask.shape[0] > 0:
                    item_backbone_mask[:, 1] = True
            else:
                item_backbone_mask = item_backbone_mask.bool()
            backbone_coords.append(_pad_3d(item_backbone_coords, max_len, 5, 3))
            backbone_mask.append(_pad_2d(item_backbone_mask, max_len, 5, False))
        result["backbone_coords"] = torch.stack(backbone_coords)
        result["backbone_mask"] = torch.stack(backbone_mask)
        functional_group_coords = []
        functional_group_mask = []
        for item in batch:
            item_coords = item["coords"].float()
            group_coords = item.get("functional_group_coords")
            group_mask = item.get("functional_group_mask")
            if not torch.is_tensor(group_coords):
                group_coords = item_coords.clone()
            else:
                group_coords = group_coords.float()
            if not torch.is_tensor(group_mask):
                group_mask = torch.zeros(item_coords.shape[0], dtype=torch.bool)
            else:
                group_mask = group_mask.bool()
            functional_group_coords.append(_pad_2d(group_coords, max_len, 3))
            functional_group_mask.append(_pad_1d(group_mask, max_len, False))
        result["functional_group_coords"] = torch.stack(functional_group_coords)
        result["functional_group_mask"] = torch.stack(functional_group_mask)
        site_slots = max(
            max(
                (
                    int(item["chemical_site_coords"].shape[1])
                    for item in batch
                    if torch.is_tensor(item.get("chemical_site_coords"))
                ),
                default=0,
            ),
            MAX_CHEMICAL_SITES,
        )
        site_type_dim = max(
            max(
                (
                    int(item["chemical_site_types"].shape[2])
                    for item in batch
                    if torch.is_tensor(item.get("chemical_site_types"))
                ),
                default=0,
            ),
            CHEMICAL_SITE_TYPE_DIM,
        )
        chemical_site_coords = []
        chemical_site_types = []
        chemical_site_orientations = []
        chemical_site_orientation_mask = []
        chemical_site_mask = []
        for item in batch:
            item_length = item["coords"].shape[0]
            site_coords = item.get("chemical_site_coords")
            site_types = item.get("chemical_site_types")
            site_orientations = item.get("chemical_site_orientations")
            site_orientation_mask = item.get("chemical_site_orientation_mask")
            site_mask = item.get("chemical_site_mask")
            if not torch.is_tensor(site_coords):
                site_coords = torch.zeros(item_length, site_slots, 3, dtype=torch.float32)
            if not torch.is_tensor(site_types):
                site_types = torch.zeros(item_length, site_slots, site_type_dim, dtype=torch.float32)
            if not torch.is_tensor(site_orientations):
                site_orientations = torch.zeros(item_length, site_slots, 3, dtype=torch.float32)
            if not torch.is_tensor(site_orientation_mask):
                site_orientation_mask = torch.zeros(item_length, site_slots, dtype=torch.bool)
            if not torch.is_tensor(site_mask):
                site_mask = torch.zeros(item_length, site_slots, dtype=torch.bool)
            chemical_site_coords.append(_pad_3d(site_coords.float(), max_len, site_slots, 3))
            chemical_site_types.append(_pad_3d(site_types.float(), max_len, site_slots, site_type_dim))
            chemical_site_orientations.append(
                _pad_3d(site_orientations.float(), max_len, site_slots, 3)
            )
            chemical_site_orientation_mask.append(
                _pad_2d(site_orientation_mask.bool(), max_len, site_slots, False)
            )
            chemical_site_mask.append(_pad_2d(site_mask.bool(), max_len, site_slots, False))
        result["chemical_site_coords"] = torch.stack(chemical_site_coords)
        result["chemical_site_types"] = torch.stack(chemical_site_types)
        result["chemical_site_orientations"] = torch.stack(chemical_site_orientations)
        result["chemical_site_orientation_mask"] = torch.stack(chemical_site_orientation_mask)
        result["chemical_site_mask"] = torch.stack(chemical_site_mask)
        result["structure_features"] = torch.stack(
            [
                _pad_2d(
                    item["structure_features"]
                    if torch.is_tensor(item.get("structure_features"))
                    else (
                        chemical_structure_feature_matrix(item.get("sequence", ""), item["coords"], item["plddt"])
                        if structure_dim == CHEMICAL_STRUCTURE_FEATURE_DIM
                        else structure_feature_matrix(item["coords"], item["plddt"])
                    ),
                    max_len,
                    structure_dim,
                )
                for item in batch
            ]
        )
    if plm_dim > 0:
        plm_features = []
        for item in batch:
            features = item.get("plm_features")
            if not torch.is_tensor(features):
                features = torch.zeros(item["aa_ids"].shape[0], plm_dim, dtype=torch.float32)
            plm_features.append(_pad_2d(features.float(), max_len, plm_dim))
        result["plm_features"] = torch.stack(plm_features)
    if include_atom and all(torch.is_tensor(item.get("atom_features")) for item in batch):
        atom_features = []
        edge_indices = []
        edge_features = []
        atom_batch = []
        atom_residue_index = []
        atom_offset = 0
        for batch_index, item in enumerate(batch):
            features = item["atom_features"].float()
            edges = item["atom_edge_index"].long()
            atom_features.append(features)
            edge_indices.append(edges + atom_offset)
            edge_features.append(item["atom_edge_features"].float())
            atom_batch.append(torch.full((features.shape[0],), batch_index, dtype=torch.long))
            residue_index = item.get("atom_residue_index")
            if not torch.is_tensor(residue_index):
                sequence_length = max(len(item.get("sequence", "")), 1)
                residue_index = (features[:, -2] * sequence_length).round().long() - 1
                residue_index = residue_index.clamp_min(0)
            atom_residue_index.append(residue_index.long())
            atom_offset += features.shape[0]
        result["atom_features"] = torch.cat(atom_features, dim=0)
        result["atom_edge_index"] = torch.cat(edge_indices, dim=1)
        result["atom_edge_features"] = torch.cat(edge_features, dim=0)
        result["atom_batch"] = torch.cat(atom_batch, dim=0)
        result["atom_residue_index"] = torch.cat(atom_residue_index, dim=0)
    return result

