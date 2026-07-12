"""Model layers for PLDDT-aware spatial message passing and fusion."""

from __future__ import annotations

import math

import torch
from torch import nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_values = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    pooled = masked_values.max(dim=1).values
    has_values = mask.any(dim=1).unsqueeze(-1)
    return torch.where(has_values, pooled, torch.zeros_like(pooled))


class RadialBasis(nn.Module):
    def __init__(self, bins: int = 16, max_distance: float = 24.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, max_distance, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max((centers[1] - centers[0]).item(), 1e-6) ** 2

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distances.unsqueeze(-1) - self.centers) ** 2)


class SequenceBranch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        group_size: int,
        residue_feature_dim: int,
        plm_embedding_dim: int,
        aa_embedding_dim: int,
        group_embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.plm_embedding_dim = max(int(plm_embedding_dim), 0)
        self.aa_embedding = nn.Embedding(vocab_size, aa_embedding_dim, padding_idx=0)
        self.group_embedding = nn.Embedding(group_size, group_embedding_dim, padding_idx=0)
        self.plm_projection = (
            nn.Sequential(nn.LayerNorm(self.plm_embedding_dim), nn.Linear(self.plm_embedding_dim, hidden_dim), nn.GELU())
            if self.plm_embedding_dim > 0
            else None
        )
        self.position_embedding = nn.Embedding(max_positions, hidden_dim)
        projected_plm_dim = hidden_dim if self.plm_projection is not None else 0
        self.input_projection = nn.Linear(
            aa_embedding_dim + group_embedding_dim + residue_feature_dim + projected_plm_dim,
            hidden_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _project_plm(self, plm_features: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
        if self.plm_projection is None:
            return None
        if plm_features is None:
            plm_features = reference.new_zeros((*reference.shape[:2], self.plm_embedding_dim))
        if plm_features.shape[-1] != self.plm_embedding_dim:
            raise ValueError(
                f"Expected PLM embedding dim {self.plm_embedding_dim}, got {plm_features.shape[-1]}."
            )
        return self.plm_projection(plm_features.to(device=reference.device, dtype=reference.dtype))

    def forward(
        self,
        aa_ids: torch.Tensor,
        group_ids: torch.Tensor,
        residue_features: torch.Tensor,
        mask: torch.Tensor,
        plm_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [
            self.aa_embedding(aa_ids),
            self.group_embedding(group_ids),
            residue_features,
        ]
        projected_plm = self._project_plm(plm_features, residue_features)
        if projected_plm is not None:
            parts.append(projected_plm)
        x = torch.cat(parts, dim=-1)
        h = self.input_projection(x)
        positions = torch.arange(h.shape[1], device=h.device).clamp_max(
            self.position_embedding.num_embeddings - 1
        )
        h = h + self.position_embedding(positions).unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.norm(self.dropout(h))


class PLDDTAwareEGNNLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rbf_bins: int,
        dropout: float,
        graph_mode: str = "hybrid",
        spatial_top_k: int = 12,
        plddt_edge_min: float = 0.25,
        plddt_edge_power: float = 1.0,
        enhanced_edge_features: bool = False,
        local_frame_edge_features: bool = False,
        backbone_geometry_edge_features: bool = False,
    ) -> None:
        super().__init__()
        self.graph_mode = graph_mode
        self.spatial_top_k = spatial_top_k
        self.plddt_edge_min = min(max(plddt_edge_min, 0.0), 1.0)
        self.plddt_edge_power = max(plddt_edge_power, 0.1)
        self.enhanced_edge_features = bool(enhanced_edge_features)
        self.local_frame_edge_features = bool(local_frame_edge_features)
        self.backbone_geometry_edge_features = bool(backbone_geometry_edge_features)
        edge_dim = rbf_bins + 2
        edge_dim += 6 if self.enhanced_edge_features else 0
        edge_dim += 15 if self.local_frame_edge_features else 0
        edge_dim += 25 if self.backbone_geometry_edge_features else 0
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _graph_mask(self, distances: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = distances.shape
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        eye = torch.eye(length, dtype=torch.bool, device=distances.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        if self.graph_mode == "full":
            return pair_mask

        positions = torch.arange(length, device=distances.device)
        seq_adjacent = (positions.view(1, -1, 1) - positions.view(1, 1, -1)).abs() == 1
        seq_adjacent = seq_adjacent.expand(batch, -1, -1) & pair_mask

        k = min(max(self.spatial_top_k, 1), max(length - 1, 1))
        masked_dist = distances.masked_fill(~pair_mask, 1e6)
        knn_idx = masked_dist.topk(k=k, largest=False, dim=-1).indices
        knn = torch.zeros_like(pair_mask)
        knn.scatter_(-1, knn_idx, True)
        knn = (knn | knn.transpose(1, 2)) & pair_mask
        return seq_adjacent | knn

    def _local_tangents(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = coords.shape
        if length <= 1:
            return torch.zeros_like(coords)
        tangents = torch.zeros_like(coords)
        if length == 2:
            direction = coords[:, 1] - coords[:, 0]
            tangents[:, 0] = direction
            tangents[:, 1] = direction
        else:
            tangents[:, 1:-1] = coords[:, 2:] - coords[:, :-2]
            tangents[:, 0] = coords[:, 1] - coords[:, 0]
            tangents[:, -1] = coords[:, -1] - coords[:, -2]
        tangents = tangents / tangents.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tangents * mask.float().unsqueeze(-1)

    def _local_frames(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Construct rotation-equivariant orthonormal frames from the C-alpha trace."""

        tangents = self._local_tangents(coords, mask)
        batch, length, _ = coords.shape
        if length <= 1:
            eye = torch.eye(3, device=coords.device, dtype=coords.dtype)
            return eye.view(1, 1, 3, 3).expand(batch, length, -1, -1)

        backward = torch.zeros_like(coords)
        forward = torch.zeros_like(coords)
        backward[:, 1:] = coords[:, 1:] - coords[:, :-1]
        backward[:, 0] = forward[:, 0] = coords[:, 1] - coords[:, 0]
        forward[:, :-1] = coords[:, 1:] - coords[:, :-1]
        forward[:, -1] = backward[:, -1]
        normal = torch.cross(backward, forward, dim=-1)

        valid_normal = (normal.norm(dim=-1) >= 1e-5) & mask
        positions = torch.arange(length, device=coords.device)
        index_distance = (positions[:, None] - positions[None, :]).abs()
        candidate_distance = index_distance.unsqueeze(0).expand(batch, -1, -1)
        candidate_distance = candidate_distance.masked_fill(~valid_normal.unsqueeze(1), length + 1)
        nearest_index = candidate_distance.argmin(dim=-1)
        nearest_normal = normal.gather(1, nearest_index.unsqueeze(-1).expand(-1, -1, 3))
        normal = torch.where(valid_normal.unsqueeze(-1), normal, nearest_normal)
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        binormal = torch.cross(normal, tangents, dim=-1)
        binormal = binormal / binormal.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        frames = torch.stack([tangents, binormal, normal], dim=-1)
        identity = torch.eye(3, device=coords.device, dtype=coords.dtype).view(1, 1, 3, 3)
        return torch.where(mask[..., None, None], frames, identity)

    def _backbone_frames(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Construct residue frames from true N-CA-C atoms, with a CA-trace fallback."""

        fallback = self._local_frames(coords, mask)
        n, ca, c = backbone_coords[:, :, 0], backbone_coords[:, :, 1], backbone_coords[:, :, 2]
        axis = c - ca
        n_direction = n - ca
        axis_norm = axis.norm(dim=-1, keepdim=True)
        axis = axis / axis_norm.clamp_min(1e-6)
        plane = n_direction - (n_direction * axis).sum(dim=-1, keepdim=True) * axis
        plane_norm = plane.norm(dim=-1, keepdim=True)
        plane = plane / plane_norm.clamp_min(1e-6)
        normal = torch.cross(axis, plane, dim=-1)
        frames = torch.stack([axis, plane, normal], dim=-1)
        valid = mask & backbone_mask[:, :, :3].all(dim=-1)
        valid = valid & (axis_norm.squeeze(-1) >= 1e-5) & (plane_norm.squeeze(-1) >= 1e-5)
        return torch.where(valid[..., None, None], frames, fallback)

    def _backbone_pair_features(
        self,
        ca_distances: torch.Tensor,
        backbone_coords: torch.Tensor,
        backbone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode corresponding-atom distances and availability for each residue pair."""

        slot_distances = []
        slot_validity = []
        for atom_index in range(backbone_coords.shape[2]):
            atom_coords = backbone_coords[:, :, atom_index]
            atom_distances = torch.cdist(atom_coords, atom_coords)
            atom_valid = (
                backbone_mask[:, :, atom_index].unsqueeze(1)
                & backbone_mask[:, :, atom_index].unsqueeze(2)
            )
            atom_distances = torch.where(atom_valid, atom_distances, ca_distances)
            slot_distances.append((atom_distances - ca_distances).clamp(-8.0, 8.0) / 8.0)
            slot_validity.append(atom_valid.to(dtype=ca_distances.dtype))
        return torch.cat(
            [torch.stack(slot_distances, dim=-1), torch.stack(slot_validity, dim=-1)],
            dim=-1,
        )

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        rbf: torch.Tensor,
        mask: torch.Tensor,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)
        distances = rel.norm(dim=-1).clamp_min(1e-6)
        graph_mask = self._graph_mask(distances, mask)

        pair_conf = (plddt.unsqueeze(1) * plddt.unsqueeze(2)).clamp_min(0.0).sqrt()
        positions = torch.arange(h.shape[1], device=h.device).float()
        seq_sep = (positions.view(1, -1, 1) - positions.view(1, 1, -1)).abs()
        seq_sep = torch.log1p(seq_sep) / torch.log(torch.tensor(float(h.shape[1] + 1), device=h.device))
        seq_sep = seq_sep.expand(h.shape[0], -1, -1)

        hi = h.unsqueeze(2).expand(-1, -1, h.shape[1], -1)
        hj = h.unsqueeze(1).expand(-1, h.shape[1], -1, -1)
        base_edge_features = [rbf(distances), pair_conf.unsqueeze(-1), seq_sep.unsqueeze(-1)]
        if self.enhanced_edge_features:
            unit_rel = rel / distances.unsqueeze(-1).clamp_min(1e-6)
            tangents = self._local_tangents(coords, mask)
            tangent_i = tangents.unsqueeze(2)
            tangent_j = tangents.unsqueeze(1)
            tangent_alignment = (tangent_i * tangent_j).sum(dim=-1).clamp(-1.0, 1.0)
            direction_i = (unit_rel * tangent_i).sum(dim=-1).clamp(-1.0, 1.0)
            direction_j = (unit_rel * tangent_j).sum(dim=-1).clamp(-1.0, 1.0)
            centroid = masked_mean(coords, mask).unsqueeze(1)
            radial = (coords - centroid).norm(dim=-1)
            radial_delta = (radial.unsqueeze(2) - radial.unsqueeze(1)) / radial.max(dim=1).values.view(-1, 1, 1).clamp_min(1.0)
            seq_adjacent_float = (seq_sep <= (1.0 / torch.log(torch.tensor(float(h.shape[1] + 1), device=h.device)) + 1e-6)).float()
            nonlocal_contact = ((seq_sep > 0.4) & (distances < 8.0)).float()
            enhanced = torch.stack(
                [
                    tangent_alignment,
                    direction_i,
                    direction_j,
                    radial_delta.clamp(-1.0, 1.0),
                    seq_adjacent_float,
                    nonlocal_contact,
                ],
                dim=-1,
            )
            base_edge_features.append(enhanced)
        if self.local_frame_edge_features:
            unit_rel = rel / distances.unsqueeze(-1).clamp_min(1e-6)
            frames = self._local_frames(coords, mask)
            frame_i = frames.unsqueeze(2)
            frame_j = frames.unsqueeze(1)
            direction_i = torch.matmul(frame_i.transpose(-1, -2), unit_rel.unsqueeze(-1)).squeeze(-1)
            direction_j = torch.matmul(frame_j.transpose(-1, -2), unit_rel.unsqueeze(-1)).squeeze(-1)
            relative_rotation = torch.matmul(frame_i.transpose(-1, -2), frame_j).flatten(start_dim=-2)
            base_edge_features.append(torch.cat([direction_i, direction_j, relative_rotation], dim=-1))
        if self.backbone_geometry_edge_features:
            if backbone_coords is None or backbone_mask is None:
                raise ValueError("Full-backbone geometry requires backbone_coords and backbone_mask.")
            unit_rel = rel / distances.unsqueeze(-1).clamp_min(1e-6)
            frames = self._backbone_frames(coords, mask, backbone_coords, backbone_mask)
            frame_i = frames.unsqueeze(2)
            frame_j = frames.unsqueeze(1)
            direction_i = torch.matmul(frame_i.transpose(-1, -2), unit_rel.unsqueeze(-1)).squeeze(-1)
            direction_j = torch.matmul(frame_j.transpose(-1, -2), unit_rel.unsqueeze(-1)).squeeze(-1)
            relative_rotation = torch.matmul(frame_i.transpose(-1, -2), frame_j).flatten(start_dim=-2)
            pair_geometry = self._backbone_pair_features(distances, backbone_coords, backbone_mask)
            base_edge_features.append(
                torch.cat([direction_i, direction_j, relative_rotation, pair_geometry], dim=-1)
            )
        edge_features = torch.cat(base_edge_features, dim=-1)
        messages = self.message_mlp(torch.cat([hi, hj, edge_features], dim=-1))

        distance_weight = torch.exp(-distances / 12.0)
        confidence_weight = self.plddt_edge_min + (1.0 - self.plddt_edge_min) * pair_conf.pow(self.plddt_edge_power)
        edge_weight = (graph_mask.float() * distance_weight * confidence_weight).unsqueeze(-1)
        degree = edge_weight.sum(dim=2).clamp_min(1.0)
        aggregated = (messages * edge_weight).sum(dim=2) / degree

        update = self.update_mlp(torch.cat([h, aggregated], dim=-1))
        h = self.norm(h + update)

        coord_coeff = torch.tanh(self.coord_mlp(messages)) * edge_weight
        coord_delta = (rel * coord_coeff).sum(dim=2) / degree
        coords = coords + 0.05 * coord_delta * mask.float().unsqueeze(-1)
        coords = coords - masked_mean(coords, mask).unsqueeze(1)
        return h, coords


class SpatialBranch(nn.Module):
    def __init__(
        self,
        residue_feature_dim: int,
        structure_feature_dim: int,
        plm_embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        rbf_bins: int,
        rbf_max_distance: float,
        dropout: float,
        graph_mode: str,
        spatial_top_k: int,
        plddt_edge_min: float = 0.25,
        plddt_edge_power: float = 1.0,
        enhanced_edge_features: bool = False,
        local_frame_edge_features: bool = False,
        backbone_geometry_edge_features: bool = False,
    ) -> None:
        super().__init__()
        self.structure_feature_dim = max(int(structure_feature_dim), 0)
        self.plm_embedding_dim = max(int(plm_embedding_dim), 0)
        self.plm_projection = (
            nn.Sequential(nn.LayerNorm(self.plm_embedding_dim), nn.Linear(self.plm_embedding_dim, hidden_dim), nn.GELU())
            if self.plm_embedding_dim > 0
            else None
        )
        projected_plm_dim = hidden_dim if self.plm_projection is not None else 0
        self.input_projection = nn.Linear(
            residue_feature_dim + self.structure_feature_dim + 1 + projected_plm_dim,
            hidden_dim,
        )
        self.rbf = RadialBasis(rbf_bins, rbf_max_distance)
        self.layers = nn.ModuleList(
            [
                PLDDTAwareEGNNLayer(
                    hidden_dim=hidden_dim,
                    rbf_bins=rbf_bins,
                    dropout=dropout,
                    graph_mode=graph_mode,
                    spatial_top_k=spatial_top_k,
                    plddt_edge_min=plddt_edge_min,
                    plddt_edge_power=plddt_edge_power,
                    enhanced_edge_features=enhanced_edge_features,
                    local_frame_edge_features=local_frame_edge_features,
                    backbone_geometry_edge_features=backbone_geometry_edge_features,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _project_plm(self, plm_features: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor | None:
        if self.plm_projection is None:
            return None
        if plm_features is None:
            plm_features = reference.new_zeros((*reference.shape[:2], self.plm_embedding_dim))
        if plm_features.shape[-1] != self.plm_embedding_dim:
            raise ValueError(
                f"Expected PLM embedding dim {self.plm_embedding_dim}, got {plm_features.shape[-1]}."
            )
        return self.plm_projection(plm_features.to(device=reference.device, dtype=reference.dtype))

    def forward(
        self,
        residue_features: torch.Tensor,
        structure_features: torch.Tensor | None,
        coords: torch.Tensor,
        plddt: torch.Tensor,
        mask: torch.Tensor,
        plm_features: torch.Tensor | None = None,
        backbone_coords: torch.Tensor | None = None,
        backbone_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [residue_features]
        if self.structure_feature_dim > 0:
            if structure_features is None:
                structure_features = residue_features.new_zeros(
                    (*residue_features.shape[:2], self.structure_feature_dim)
                )
            if structure_features.shape[-1] != self.structure_feature_dim:
                raise ValueError(
                    f"Expected structure feature dim {self.structure_feature_dim}, "
                    f"got {structure_features.shape[-1]}."
                )
            parts.append(structure_features.to(device=residue_features.device, dtype=residue_features.dtype))
        parts.append(plddt.unsqueeze(-1))
        projected_plm = self._project_plm(plm_features, residue_features)
        if projected_plm is not None:
            parts.append(projected_plm)
        h = self.input_projection(torch.cat(parts, dim=-1))
        coords = coords - masked_mean(coords, mask).unsqueeze(1)
        for layer in self.layers:
            h, coords = layer(
                h,
                coords,
                plddt,
                self.rbf,
                mask,
                backbone_coords=backbone_coords,
                backbone_mask=backbone_mask,
            )
        return self.norm(h)


class PLDDTAwareFusion(nn.Module):
    def __init__(
        self,
        residue_feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        gate_mode: str = "plddt",
        initial_3d_weight: float = 0.5,
        gate_center: float = 0.7,
        gate_temperature: float = 0.12,
        gate_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.gate_mode = gate_mode.lower()
        self.gate_center = gate_center
        self.gate_temperature = max(gate_temperature, 1e-3)
        self.gate_feature_dim = max(int(gate_feature_dim), 0)
        prior = min(max(initial_3d_weight, 1e-4), 1.0 - 1e-4)
        prior_logit = math.log(prior / (1.0 - prior))
        self.initial_gate_logit = prior_logit
        self.node_gate = nn.Sequential(
            nn.Linear(residue_feature_dim + self.gate_feature_dim + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(3 + self.gate_feature_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.node_confidence_weights = nn.Parameter(torch.tensor([2.0, -0.5, -1.0]))
        self.node_confidence_bias = nn.Parameter(torch.tensor(prior_logit))
        self.global_confidence_weights = nn.Parameter(torch.tensor([2.0, 0.75, -1.0]))
        self.global_confidence_bias = nn.Parameter(torch.tensor(prior_logit))
        self.residual_node_confidence_weights = nn.Parameter(torch.zeros(3))
        self.residual_node_confidence_bias = nn.Parameter(torch.tensor(0.0))
        self.residual_global_confidence_weights = nn.Parameter(torch.zeros(3))
        self.residual_global_confidence_bias = nn.Parameter(torch.tensor(0.0))
        self._initialize_gate_prior(prior_logit)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)

    def _initialize_gate_prior(self, prior_logit: float) -> None:
        for gate in (self.node_gate, self.global_gate):
            final = gate[-2]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.constant_(final.bias, prior_logit)

    def forward(
        self,
        sequence_h: torch.Tensor,
        spatial_h: torch.Tensor,
        residue_features: torch.Tensor,
        plddt: torch.Tensor,
        mask: torch.Tensor,
        gate_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.gate_feature_dim > 0:
            if gate_features is None:
                gate_features = residue_features.new_zeros(
                    (*residue_features.shape[:2], self.gate_feature_dim)
                )
            if gate_features.shape[-1] != self.gate_feature_dim:
                raise ValueError(
                    f"Expected gate feature dim {self.gate_feature_dim}, got {gate_features.shape[-1]}."
                )
            gate_features = gate_features.to(device=residue_features.device, dtype=residue_features.dtype)
        else:
            gate_features = None

        node_gate_inputs = [residue_features]
        if gate_features is not None:
            node_gate_inputs.append(gate_features)
        node_gate_inputs.append(plddt.unsqueeze(-1))
        node_gate_input = torch.cat(node_gate_inputs, dim=-1)

        confidence_centered = plddt - self.gate_center
        node_confidence_inputs = torch.stack(
            [
                confidence_centered / self.gate_temperature,
                confidence_centered.pow(2) / (self.gate_temperature**2),
                (plddt < 0.55).float(),
            ],
            dim=-1,
        )

        if self.gate_mode in {"none", "off", "disabled"}:
            node_gate = torch.ones_like(plddt).unsqueeze(-1)
        elif self.gate_mode in {"direct", "direct_plddt", "normalized"}:
            node_gate = torch.sigmoid(
                (plddt.unsqueeze(-1) - self.gate_center) / self.gate_temperature
                + self.initial_gate_logit
            )
        elif self.gate_mode in {"confidence", "confidence_parametric", "adaptive_confidence"}:
            node_logit = node_confidence_inputs @ self.node_confidence_weights.to(dtype=node_confidence_inputs.dtype)
            node_gate = torch.sigmoid(node_logit.unsqueeze(-1) + self.node_confidence_bias)
        elif self.gate_mode in {"confidence_residual", "residual_confidence", "learned_confidence"}:
            base_gate = self.node_gate(node_gate_input)
            scale_logit = (
                node_confidence_inputs
                @ self.residual_node_confidence_weights.to(dtype=node_confidence_inputs.dtype)
            )
            scale = 0.5 + torch.sigmoid(scale_logit.unsqueeze(-1) + self.residual_node_confidence_bias)
            node_gate = (base_gate * scale).clamp(0.0, 1.0)
        else:
            node_gate = self.node_gate(node_gate_input)
        gated_spatial = spatial_h * node_gate
        cross, attention_weights = self.attention(
            query=sequence_h,
            key=gated_spatial,
            value=gated_spatial,
            key_padding_mask=~mask,
            need_weights=True,
        )

        valid_plddt = plddt.masked_fill(~mask, 0.0)
        lengths = mask.float().sum(dim=1).clamp_min(1.0)
        mean_conf = valid_plddt.sum(dim=1) / lengths
        min_conf = valid_plddt.masked_fill(~mask, 1.0).min(dim=1).values
        low_conf_frac = ((plddt < 0.55) & mask).float().sum(dim=1) / lengths
        global_gate_inputs = [mean_conf.unsqueeze(-1), min_conf.unsqueeze(-1), low_conf_frac.unsqueeze(-1)]
        if gate_features is not None:
            global_gate_inputs.append(masked_mean(gate_features, mask))
        global_gate_input = torch.cat(global_gate_inputs, dim=-1)

        if self.gate_mode in {"none", "off", "disabled"}:
            graph_gate = torch.ones((sequence_h.shape[0], 1), device=sequence_h.device, dtype=sequence_h.dtype)
        elif self.gate_mode in {"direct", "direct_plddt", "normalized"}:
            graph_gate = torch.sigmoid(
                ((mean_conf - self.gate_center) / self.gate_temperature)
                - low_conf_frac
                + 0.5 * (min_conf - self.gate_center)
                + self.initial_gate_logit
            ).unsqueeze(-1)
        elif self.gate_mode in {"confidence", "confidence_parametric", "adaptive_confidence"}:
            global_inputs = torch.stack(
                [
                    (mean_conf - self.gate_center) / self.gate_temperature,
                    (min_conf - self.gate_center) / self.gate_temperature,
                    low_conf_frac,
                ],
                dim=-1,
            )
            graph_logit = global_inputs @ self.global_confidence_weights.to(dtype=global_inputs.dtype)
            graph_gate = torch.sigmoid(graph_logit + self.global_confidence_bias).unsqueeze(-1)
        elif self.gate_mode in {"confidence_residual", "residual_confidence", "learned_confidence"}:
            global_inputs = torch.stack(
                [
                    (mean_conf - self.gate_center) / self.gate_temperature,
                    (min_conf - self.gate_center) / self.gate_temperature,
                    low_conf_frac,
                ],
                dim=-1,
            )
            base_gate = self.global_gate(global_gate_input)
            scale_logit = (
                global_inputs
                @ self.residual_global_confidence_weights.to(dtype=global_inputs.dtype)
            )
            scale = 0.5 + torch.sigmoid(scale_logit + self.residual_global_confidence_bias)
            graph_gate = (base_gate * scale.unsqueeze(-1)).clamp(0.0, 1.0)
        else:
            graph_gate = self.global_gate(global_gate_input)

        fused_nodes = self.norm(torch.cat([sequence_h, cross * graph_gate.unsqueeze(1)], dim=-1))
        diagnostics = {
            "node_gate": node_gate.squeeze(-1).detach(),
            "global_gate": graph_gate.squeeze(-1).detach(),
            "attention": attention_weights.detach(),
        }
        return fused_nodes, diagnostics

