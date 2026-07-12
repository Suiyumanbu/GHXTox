"""Lightweight molecular message-passing branch for peptide atom graphs."""

from __future__ import annotations

import torch
from torch import nn


class AtomMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        messages = self.message(torch.cat([h[target], h[source], edge_features], dim=-1))
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, target, messages)
        degree = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=h.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        return self.norm(h + self.update(torch.cat([h, aggregated], dim=-1)))


class AtomGraphBranch(nn.Module):
    def __init__(
        self,
        atom_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        multiscale: bool = False,
    ) -> None:
        super().__init__()
        self.multiscale = bool(multiscale)
        self.input_projection = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList(
            [AtomMessageLayer(hidden_dim, edge_feature_dim, dropout) for _ in range(num_layers)]
        )
        self.scale_fusion = (
            nn.Sequential(
                nn.Linear(hidden_dim * num_layers, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            if self.multiscale
            else None
        )
        self.attention = nn.Linear(hidden_dim, 1)

    def encode_atoms(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        h = self.input_projection(atom_features)
        scales = []
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
            scales.append(h)
        if self.scale_fusion is not None:
            h = self.scale_fusion(torch.cat(scales, dim=-1))
        return h

    def residue_embeddings(
        self,
        atom_h: torch.Tensor,
        atom_batch: torch.Tensor,
        atom_residue_index: torch.Tensor,
        batch_size: int,
        max_length: int,
    ) -> torch.Tensor:
        """Attention-pool atoms within each residue while preserving sequence position."""

        flat_index = atom_batch * max_length + atom_residue_index.clamp(0, max_length - 1)
        attention = torch.exp(self.attention(atom_h).squeeze(-1).clamp(-12.0, 12.0))
        weighted_sum = atom_h.new_zeros((batch_size * max_length, atom_h.shape[-1]))
        weight_sum = atom_h.new_zeros(batch_size * max_length)
        weighted_sum.index_add_(0, flat_index, atom_h * attention.unsqueeze(-1))
        weight_sum.index_add_(0, flat_index, attention)
        pooled = weighted_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)
        return pooled.view(batch_size, max_length, atom_h.shape[-1])

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        atom_batch: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        h = self.encode_atoms(atom_features, edge_index, edge_features)

        attention = torch.exp(self.attention(h).squeeze(-1).clamp(-12.0, 12.0))
        weighted_sum = h.new_zeros((batch_size, h.shape[-1]))
        weight_sum = h.new_zeros(batch_size)
        weighted_sum.index_add_(0, atom_batch, h * attention.unsqueeze(-1))
        weight_sum.index_add_(0, atom_batch, attention)
        attention_pool = weighted_sum / weight_sum.clamp_min(1e-6).unsqueeze(-1)

        max_pool = torch.stack([h[atom_batch == index].max(dim=0).values for index in range(batch_size)])
        return torch.cat([attention_pool, max_pool], dim=-1)
