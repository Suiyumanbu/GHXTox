"""Continuous multiscale interaction network for chemistry-aware pseudo-sites."""

from __future__ import annotations

import torch
from torch import nn

from ghxtox.chemical_sites import CHEMICAL_SITE_TYPE_DIM, CHEMICAL_SITE_TYPE_TO_INDEX
from ghxtox.models.layers import RadialBasis


class ChemicalSiteInteractionBranch(nn.Module):
    """Encode typed chemical-site interactions and return a residue residual.

    Edges use continuous radial bases in both Angstrom and radius-of-gyration
    normalized units.  Same-residue edges are excluded because their geometry
    is largely predetermined by residue identity and would duplicate ESM2.
    """

    def __init__(
        self,
        hidden_dim: int,
        site_type_dim: int = CHEMICAL_SITE_TYPE_DIM,
        site_hidden_dim: int = 64,
        num_layers: int = 2,
        raw_rbf_bins: int = 16,
        normalized_rbf_bins: int = 8,
        max_distance: float = 16.0,
        normalized_max_distance: float = 4.0,
        dropout: float = 0.1,
        use_raw_rbf: bool = True,
        use_normalized_rbf: bool = True,
        use_orientation: bool = True,
        use_interaction_types: bool = True,
        use_plddt: bool = True,
        use_hydrophobic_sites: bool = True,
        max_site_slots: int = 2,
        exclude_same_residue_edges: bool = True,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.max_distance = float(max_distance)
        self.use_raw_rbf = bool(use_raw_rbf)
        self.use_normalized_rbf = bool(use_normalized_rbf)
        self.use_orientation = bool(use_orientation)
        self.use_interaction_types = bool(use_interaction_types)
        self.use_plddt = bool(use_plddt)
        self.use_hydrophobic_sites = bool(use_hydrophobic_sites)
        self.max_site_slots = max(int(max_site_slots), 1)
        self.exclude_same_residue_edges = bool(exclude_same_residue_edges)
        self.raw_rbf = (
            RadialBasis(raw_rbf_bins, self.max_distance) if self.use_raw_rbf else None
        )
        self.normalized_rbf = (
            RadialBasis(normalized_rbf_bins, normalized_max_distance)
            if self.use_normalized_rbf
            else None
        )
        self.site_input = nn.Sequential(
            nn.Linear(site_type_dim + 2, site_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(site_hidden_dim),
        )
        edge_dim = 0
        edge_dim += raw_rbf_bins if self.use_raw_rbf else 0
        edge_dim += normalized_rbf_bins if self.use_normalized_rbf else 0
        edge_dim += 4 if self.use_orientation else 0
        edge_dim += 6 if self.use_interaction_types else 0
        edge_dim += 1 if self.use_plddt else 0
        if edge_dim <= 0:
            raise ValueError("Chemical-site edges require at least one enabled edge feature.")
        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.message_layers.append(
                nn.Sequential(
                    nn.Linear(site_hidden_dim * 2 + edge_dim, site_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(site_hidden_dim),
                    nn.Dropout(dropout),
                    nn.Linear(site_hidden_dim, site_hidden_dim),
                )
            )
            self.update_layers.append(
                nn.Sequential(
                    nn.Linear(site_hidden_dim * 2, site_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(site_hidden_dim),
                    nn.Dropout(dropout),
                )
            )
        self.residual_projection = nn.Linear(site_hidden_dim, hidden_dim)
        if zero_init_residual:
            nn.init.zeros_(self.residual_projection.weight)
            nn.init.zeros_(self.residual_projection.bias)

    @staticmethod
    def _compatibility(types_source: torch.Tensor, types_target: torch.Tensor) -> torch.Tensor:
        def channel(values: torch.Tensor, name: str) -> torch.Tensor:
            return values[:, CHEMICAL_SITE_TYPE_TO_INDEX[name]]

        pos_i, pos_j = channel(types_source, "positive"), channel(types_target, "positive")
        neg_i, neg_j = channel(types_source, "negative"), channel(types_target, "negative")
        donor_i, donor_j = channel(types_source, "donor"), channel(types_target, "donor")
        acceptor_i, acceptor_j = channel(types_source, "acceptor"), channel(types_target, "acceptor")
        aromatic_i, aromatic_j = channel(types_source, "aromatic"), channel(types_target, "aromatic")
        hydrophobic_i, hydrophobic_j = channel(types_source, "hydrophobic"), channel(types_target, "hydrophobic")
        sulfur_i, sulfur_j = channel(types_source, "sulfur"), channel(types_target, "sulfur")
        return torch.stack(
            [
                pos_i * neg_j + neg_i * pos_j,
                donor_i * acceptor_j + acceptor_i * donor_j,
                aromatic_i * aromatic_j,
                pos_i * aromatic_j + aromatic_i * pos_j,
                sulfur_i * sulfur_j,
                hydrophobic_i * hydrophobic_j,
            ],
            dim=-1,
        )

    def _edge_features(
        self,
        coordinates: torch.Tensor,
        orientations: torch.Tensor,
        orientation_mask: torch.Tensor,
        types: torch.Tensor,
        confidence: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        distances: torch.Tensor,
        radius_of_gyration: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vectors = coordinates[target] - coordinates[source]
        unit_vectors = vectors / distances.unsqueeze(-1).clamp_min(1e-6)
        source_valid = orientation_mask[source].float()
        target_valid = orientation_mask[target].float()
        source_alignment = (
            (orientations[source] * unit_vectors).sum(dim=-1).abs() * source_valid
        )
        target_alignment = (
            (orientations[target] * -unit_vectors).sum(dim=-1).abs() * target_valid
        )
        mutual_alignment = (
            (orientations[source] * orientations[target]).sum(dim=-1).abs()
            * source_valid
            * target_valid
        )
        orientation_pair_valid = source_valid * target_valid
        pair_confidence = (confidence[source] * confidence[target]).clamp_min(0.0).sqrt()
        normalized_distances = distances / radius_of_gyration.clamp_min(1e-3)
        parts = []
        if self.raw_rbf is not None:
            parts.append(self.raw_rbf(distances))
        if self.normalized_rbf is not None:
            parts.append(self.normalized_rbf(normalized_distances))
        if self.use_orientation:
            parts.append(
                torch.stack(
                    [
                        source_alignment,
                        target_alignment,
                        mutual_alignment,
                        orientation_pair_valid,
                    ],
                    dim=-1,
                )
            )
        if self.use_interaction_types:
            parts.append(self._compatibility(types[source], types[target]))
        if self.use_plddt:
            parts.append(pair_confidence.unsqueeze(-1))
        edge_features = torch.cat(parts, dim=-1)
        message_weight = pair_confidence if self.use_plddt else torch.ones_like(pair_confidence)
        return edge_features, message_weight

    def forward(
        self,
        site_coords: torch.Tensor,
        site_types: torch.Tensor,
        site_orientations: torch.Tensor,
        site_orientation_mask: torch.Tensor,
        site_mask: torch.Tensor,
        residue_coords: torch.Tensor,
        residue_mask: torch.Tensor,
        plddt: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        site_coords = site_coords[:, :, : self.max_site_slots]
        site_types = site_types[:, :, : self.max_site_slots]
        site_orientations = site_orientations[:, :, : self.max_site_slots]
        site_orientation_mask = site_orientation_mask[:, :, : self.max_site_slots]
        site_mask = site_mask[:, :, : self.max_site_slots]
        if not self.use_hydrophobic_sites:
            site_types = site_types.clone()
            site_types[..., CHEMICAL_SITE_TYPE_TO_INDEX["hydrophobic"]] = 0.0
            site_mask = site_mask & (site_types.sum(dim=-1) > 0.0)
        batch_size, max_length, site_slots, _ = site_coords.shape
        output = site_coords.new_zeros((batch_size, max_length, self.residual_projection.out_features))
        site_counts = site_coords.new_zeros(batch_size)
        edge_counts = site_coords.new_zeros(batch_size)

        for batch_index in range(batch_size):
            flat_mask = site_mask[batch_index].reshape(-1).bool()
            valid_flat = flat_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_flat.numel() == 0:
                continue
            coordinates = site_coords[batch_index].reshape(-1, 3)[valid_flat]
            types = site_types[batch_index].reshape(-1, site_types.shape[-1])[valid_flat]
            orientations = site_orientations[batch_index].reshape(-1, 3)[valid_flat]
            orientation_valid = site_orientation_mask[batch_index].reshape(-1)[valid_flat].bool()
            residue_indices = torch.div(valid_flat, site_slots, rounding_mode="floor")
            confidence = plddt[batch_index, residue_indices].clamp(0.0, 1.0)
            confidence_input = confidence if self.use_plddt else torch.ones_like(confidence)
            orientation_input = (
                orientation_valid.float()
                if self.use_orientation
                else torch.zeros_like(orientation_valid, dtype=coordinates.dtype)
            )
            h = self.site_input(
                torch.cat(
                    [
                        types,
                        confidence_input.unsqueeze(-1),
                        orientation_input.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            )

            pair_distances = torch.cdist(coordinates, coordinates)
            if self.exclude_same_residue_edges:
                allowed_residue_pair = residue_indices[:, None] != residue_indices[None, :]
            else:
                allowed_residue_pair = torch.ones_like(pair_distances, dtype=torch.bool)
            edge_mask = (
                allowed_residue_pair
                & (pair_distances > 1e-6)
                & (pair_distances <= self.max_distance)
            )
            source, target = edge_mask.nonzero(as_tuple=True)
            valid_ca = residue_coords[batch_index, residue_mask[batch_index].bool()]
            if valid_ca.shape[0] > 0:
                ca_center = valid_ca.mean(dim=0, keepdim=True)
                radius_of_gyration = (
                    (valid_ca - ca_center).square().sum(dim=-1).mean().clamp_min(1e-6).sqrt()
                )
            else:
                radius_of_gyration = coordinates.new_tensor(1.0)

            if source.numel() > 0:
                distances = pair_distances[source, target]
                edge_features, pair_confidence = self._edge_features(
                    coordinates,
                    orientations,
                    orientation_valid,
                    types,
                    confidence,
                    source,
                    target,
                    distances,
                    radius_of_gyration,
                )
                for message_layer, update_layer in zip(
                    self.message_layers, self.update_layers, strict=True
                ):
                    messages = message_layer(
                        torch.cat([h[source], h[target], edge_features], dim=-1)
                    )
                    messages = messages * pair_confidence.unsqueeze(-1)
                    aggregate = torch.zeros_like(h)
                    aggregate.index_add_(0, target, messages)
                    degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
                    degree.index_add_(0, target, pair_confidence)
                    aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
                    h = h + update_layer(torch.cat([h, aggregate], dim=-1))
            residue_hidden = h.new_zeros((max_length, h.shape[-1]))
            residue_hidden.index_add_(0, residue_indices, h)
            residue_site_count = h.new_zeros(max_length)
            residue_site_count.index_add_(0, residue_indices, torch.ones_like(residue_indices, dtype=h.dtype))
            residue_hidden = residue_hidden / residue_site_count.clamp_min(1.0).unsqueeze(-1)
            output[batch_index] = self.residual_projection(residue_hidden)
            site_counts[batch_index] = valid_flat.numel()
            edge_counts[batch_index] = source.numel()

        return output, {
            "chemical_site_count": site_counts,
            "chemical_edge_count": edge_counts,
            "chemical_residual_norm": output.square().sum(dim=-1).sqrt().mean(dim=-1),
        }
