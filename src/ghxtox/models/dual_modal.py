"""Dual-modal GHXTox architecture."""

from __future__ import annotations

import torch
from torch import nn

from ghxtox.constants import AA_TO_IDX, FUNCTIONAL_GROUPS
from ghxtox.data import GLOBAL_FEATURE_DIM
from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.geometry_features import STRUCTURE_FEATURE_DIM
from ghxtox.atom_graph import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM
from ghxtox.models.atom_graph import AtomGraphBranch
from ghxtox.models.layers import (
    PLDDTAwareFusion,
    SequenceBranch,
    SequenceMultiscaleBranch,
    SpatialBranch,
    masked_max,
    masked_mean,
)


class GHXToxModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        hidden_dim = int(model_cfg.get("hidden_dim", 96))
        dropout = float(model_cfg.get("dropout", 0.15))
        heads = int(model_cfg.get("num_attention_heads", 4))
        plm_embedding_dim = int(model_cfg.get("plm_embedding_dim", 0))
        spatial_plm_embedding_dim = int(
            model_cfg.get("spatial_plm_embedding_dim", plm_embedding_dim)
        )
        structure_feature_dim = int(model_cfg.get("structure_feature_dim", 0))
        if structure_feature_dim < 0:
            structure_feature_dim = STRUCTURE_FEATURE_DIM
        spatial_structure_feature_dim = int(model_cfg.get("spatial_structure_feature_dim", structure_feature_dim))
        self.spatial_structure_feature_dim = spatial_structure_feature_dim
        self.gate_feature_start = int(model_cfg.get("gate_feature_start", spatial_structure_feature_dim))
        self.gate_feature_dim = int(model_cfg.get("gate_feature_dim", 0))
        self.modality = str(model_cfg.get("modality", "fusion")).lower()
        self.use_global_features = bool(model_cfg.get("global_features", False))
        classifier_input_dim = hidden_dim * 2
        classifier_input_dim += GLOBAL_FEATURE_DIM if self.use_global_features else 0

        sequence_architecture = str(model_cfg.get("sequence_architecture", "transformer")).lower()
        sequence_class = SequenceMultiscaleBranch if sequence_architecture == "multiscale_1d" else SequenceBranch
        sequence_kwargs = dict(
            vocab_size=len(AA_TO_IDX),
            group_size=len(FUNCTIONAL_GROUPS),
            residue_feature_dim=RESIDUE_FEATURE_DIM,
            plm_embedding_dim=plm_embedding_dim,
            aa_embedding_dim=int(model_cfg.get("aa_embedding_dim", 32)),
            group_embedding_dim=int(model_cfg.get("group_embedding_dim", 12)),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if sequence_class is SequenceMultiscaleBranch:
            sequence_kwargs.update(
                kernels=tuple(int(value) for value in model_cfg.get("sequence_kernels", [3, 5, 7])),
                use_multiscale=bool(model_cfg.get("sequence_use_multiscale", True)),
                use_bilstm=bool(model_cfg.get("sequence_use_bilstm", True)),
                use_residual=bool(model_cfg.get("sequence_use_residual", True)),
            )
        else:
            sequence_kwargs.update(
                num_layers=int(model_cfg.get("num_sequence_layers", 2)),
                num_heads=heads,
            )
        self.sequence_branch = sequence_class(**sequence_kwargs)
        self.atom_branch = None
        if self.modality in {"atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"}:
            self.atom_branch = AtomGraphBranch(
                atom_feature_dim=int(model_cfg.get("atom_feature_dim", ATOM_FEATURE_DIM)),
                edge_feature_dim=int(model_cfg.get("atom_edge_feature_dim", EDGE_FEATURE_DIM)),
                hidden_dim=hidden_dim,
                num_layers=int(model_cfg.get("num_atom_layers", 3)),
                dropout=dropout,
                multiscale=bool(model_cfg.get("atom_multiscale", False)),
            )
        if self.modality == "sequence_atom":
            self.sequence_to_atom_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True
            )
            self.atom_to_sequence_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True
            )
            self.sequence_atom_norm = nn.LayerNorm(hidden_dim)
            self.atom_sequence_norm = nn.LayerNorm(hidden_dim)
        if self.modality == "fusion_atom_residual":
            initial_atom_weight = min(max(float(model_cfg.get("atom_residual_initial_weight", 0.05)), 1e-4), 1.0 - 1e-4)
            self.atom_residual_logit = nn.Parameter(
                torch.logit(torch.tensor(initial_atom_weight, dtype=torch.float32))
            )
            self.atom_residual_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.LayerNorm(hidden_dim * 2),
            )
        if self.modality == "residual_experts":
            self.residual_atom_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True
            )
            self.residual_spatial_attention = nn.MultiheadAttention(
                hidden_dim, heads, dropout=dropout, batch_first=True
            )
            self.residual_atom_norm = nn.LayerNorm(hidden_dim)
            self.residual_spatial_norm = nn.LayerNorm(hidden_dim)
            self.atom_delta_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.spatial_delta_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            nn.init.zeros_(self.atom_delta_head[-1].weight)
            nn.init.zeros_(self.atom_delta_head[-1].bias)
            nn.init.zeros_(self.spatial_delta_head[-1].weight)
            nn.init.zeros_(self.spatial_delta_head[-1].bias)
            initial_atom = min(max(float(model_cfg.get("residual_atom_initial_weight", 0.10)), 1e-4), 1 - 1e-4)
            initial_spatial = min(max(float(model_cfg.get("residual_spatial_initial_weight", 0.10)), 1e-4), 1 - 1e-4)
            self.residual_atom_logit = nn.Parameter(torch.logit(torch.tensor(initial_atom)))
            self.residual_spatial_logit = nn.Parameter(torch.logit(torch.tensor(initial_spatial)))
        self.spatial_branch = SpatialBranch(
            residue_feature_dim=RESIDUE_FEATURE_DIM,
            structure_feature_dim=spatial_structure_feature_dim,
            plm_embedding_dim=spatial_plm_embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg.get("num_egnn_layers", 3)),
            rbf_bins=int(model_cfg.get("rbf_bins", 16)),
            rbf_max_distance=float(model_cfg.get("rbf_max_distance", 24.0)),
            dropout=dropout,
            graph_mode=str(model_cfg.get("graph_mode", "full")),
            spatial_top_k=int(model_cfg.get("spatial_top_k", 12)),
            plddt_edge_min=float(model_cfg.get("plddt_edge_min", 0.25)),
            plddt_edge_power=float(model_cfg.get("plddt_edge_power", 1.0)),
            enhanced_edge_features=bool(model_cfg.get("enhanced_edge_features", False)),
            local_frame_edge_features=bool(model_cfg.get("local_frame_edge_features", False)),
            backbone_geometry_edge_features=bool(model_cfg.get("backbone_geometry_edge_features", False)),
        )
        self.fusion = PLDDTAwareFusion(
            residue_feature_dim=RESIDUE_FEATURE_DIM,
            hidden_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            gate_mode=str(model_cfg.get("fusion_gate", "plddt")),
            initial_3d_weight=float(model_cfg.get("ahp_initial_3d_weight", 0.5)),
            gate_center=float(model_cfg.get("plddt_gate_center", 0.7)),
            gate_temperature=float(model_cfg.get("plddt_gate_temperature", 0.12)),
            gate_feature_dim=self.gate_feature_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _classifier_input(self, pooled: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [pooled]
        if self.use_global_features:
            global_features = batch.get("global_features")
            if global_features is None:
                global_features = pooled.new_zeros((pooled.shape[0], GLOBAL_FEATURE_DIM))
            parts.append(global_features.to(device=pooled.device, dtype=pooled.dtype))
        return torch.cat(parts, dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["mask"].bool()
        if self.modality == "atom_only":
            required = ("atom_features", "atom_edge_index", "atom_edge_features", "atom_batch")
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"atom_only model requires atom graph fields; missing {missing}.")
            pooled = self.atom_branch(
                atom_features=batch["atom_features"],
                edge_index=batch["atom_edge_index"],
                edge_features=batch["atom_edge_features"],
                atom_batch=batch["atom_batch"],
                batch_size=mask.shape[0],
            )
            logits = self.classifier(self._classifier_input(pooled, batch)).squeeze(-1)
            return {
                "logits": logits,
                "embedding": pooled,
                "node_gate": torch.zeros_like(batch["plddt"]),
                "global_gate": pooled.new_zeros(mask.shape[0]),
            }
        plm_features = batch.get("plm_features")
        sequence_h = self.sequence_branch(
            aa_ids=batch["aa_ids"],
            group_ids=batch["group_ids"],
            residue_features=batch["residue_features"],
            mask=mask,
            plm_features=plm_features,
        )

        if self.modality == "sequence_atom":
            required = (
                "atom_features",
                "atom_edge_index",
                "atom_edge_features",
                "atom_batch",
                "atom_residue_index",
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"sequence_atom model requires atom graph fields; missing {missing}.")
            atom_h = self.atom_branch.encode_atoms(
                batch["atom_features"], batch["atom_edge_index"], batch["atom_edge_features"]
            )
            residue_atom_h = self.atom_branch.residue_embeddings(
                atom_h=atom_h,
                atom_batch=batch["atom_batch"],
                atom_residue_index=batch["atom_residue_index"],
                batch_size=mask.shape[0],
                max_length=mask.shape[1],
            )
            sequence_context, _ = self.sequence_to_atom_attention(
                query=sequence_h,
                key=residue_atom_h,
                value=residue_atom_h,
                key_padding_mask=~mask,
                need_weights=False,
            )
            atom_context, _ = self.atom_to_sequence_attention(
                query=residue_atom_h,
                key=sequence_h,
                value=sequence_h,
                key_padding_mask=~mask,
                need_weights=False,
            )
            sequence_fused = self.sequence_atom_norm(sequence_h + sequence_context)
            atom_fused = self.atom_sequence_norm(residue_atom_h + atom_context)
            fused_nodes = torch.cat([sequence_fused, atom_fused], dim=-1)
            pooled = masked_mean(fused_nodes, mask)
            logits = self.classifier(self._classifier_input(pooled, batch)).squeeze(-1)
            return {
                "logits": logits,
                "embedding": pooled,
                "node_gate": torch.zeros_like(batch["plddt"]),
                "global_gate": pooled.new_zeros(mask.shape[0]),
            }

        if self.modality == "sequence_only":
            pooled = torch.cat([masked_mean(sequence_h, mask), masked_max(sequence_h, mask)], dim=-1)
            logits = self.classifier(self._classifier_input(pooled, batch)).squeeze(-1)
            batch_size = sequence_h.shape[0]
            return {
                "logits": logits,
                "embedding": pooled,
                "node_gate": torch.zeros_like(batch["plddt"]),
                "global_gate": torch.zeros(batch_size, device=sequence_h.device, dtype=sequence_h.dtype),
            }

        structure_features = batch.get("structure_features")
        spatial_structure_features = structure_features
        gate_features = None
        if structure_features is not None:
            if self.spatial_structure_feature_dim > 0:
                spatial_structure_features = structure_features[..., : self.spatial_structure_feature_dim]
            if self.gate_feature_dim > 0:
                end = self.gate_feature_start + self.gate_feature_dim
                gate_features = structure_features[..., self.gate_feature_start:end]

        spatial_h = self.spatial_branch(
            residue_features=batch["residue_features"],
            structure_features=spatial_structure_features,
            coords=batch["coords"],
            plddt=batch["plddt"],
            mask=mask,
            plm_features=plm_features,
            backbone_coords=batch.get("backbone_coords"),
            backbone_mask=batch.get("backbone_mask"),
        )

        if self.modality == "residual_experts":
            required = (
                "atom_features", "atom_edge_index", "atom_edge_features", "atom_batch", "atom_residue_index"
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"residual_experts model requires atom graph fields; missing {missing}.")
            atom_h = self.atom_branch.encode_atoms(
                batch["atom_features"], batch["atom_edge_index"], batch["atom_edge_features"]
            )
            residue_atom_h = self.atom_branch.residue_embeddings(
                atom_h=atom_h,
                atom_batch=batch["atom_batch"],
                atom_residue_index=batch["atom_residue_index"],
                batch_size=mask.shape[0],
                max_length=mask.shape[1],
            )
            atom_context, _ = self.residual_atom_attention(
                query=sequence_h,
                key=residue_atom_h,
                value=residue_atom_h,
                key_padding_mask=~mask,
                need_weights=False,
            )
            spatial_values = spatial_h * batch["plddt"].clamp(0.0, 1.0).unsqueeze(-1)
            spatial_context, _ = self.residual_spatial_attention(
                query=sequence_h,
                key=spatial_values,
                value=spatial_values,
                key_padding_mask=~mask,
                need_weights=False,
            )
            base_pooled = torch.cat([masked_mean(sequence_h, mask), masked_max(sequence_h, mask)], dim=-1)
            base_logits = self.classifier(self._classifier_input(base_pooled, batch)).squeeze(-1)
            atom_delta = self.atom_delta_head(masked_mean(self.residual_atom_norm(atom_context), mask)).squeeze(-1)
            spatial_delta = self.spatial_delta_head(
                masked_mean(self.residual_spatial_norm(spatial_context), mask)
            ).squeeze(-1)
            atom_weight = torch.sigmoid(self.residual_atom_logit)
            spatial_weight = torch.sigmoid(self.residual_spatial_logit)
            logits = base_logits + atom_weight * atom_delta + spatial_weight * spatial_delta
            return {
                "logits": logits,
                "base_logits": base_logits,
                "atom_delta": atom_weight * atom_delta,
                "spatial_delta": spatial_weight * spatial_delta,
                "atom_expert_weight": atom_weight,
                "spatial_expert_weight": spatial_weight,
                "embedding": base_pooled,
                "node_gate": batch["plddt"] * spatial_weight,
                "global_gate": spatial_weight.expand(mask.shape[0]),
            }

        if self.modality == "spatial_only":
            zero = torch.zeros_like(spatial_h)
            fused_nodes = torch.cat([zero, spatial_h], dim=-1)
            pooled = masked_mean(fused_nodes, mask)
            logits = self.classifier(self._classifier_input(pooled, batch)).squeeze(-1)
            batch_size = spatial_h.shape[0]
            return {
                "logits": logits,
                "embedding": pooled,
                "node_gate": torch.ones_like(batch["plddt"]),
                "global_gate": torch.ones(batch_size, device=spatial_h.device, dtype=spatial_h.dtype),
            }

        fused_nodes, diagnostics = self.fusion(
            sequence_h=sequence_h,
            spatial_h=spatial_h,
            residue_features=batch["residue_features"],
            plddt=batch["plddt"],
            mask=mask,
            gate_features=gate_features,
        )
        if self.modality == "fusion_atom_residual":
            required = (
                "atom_features",
                "atom_edge_index",
                "atom_edge_features",
                "atom_batch",
                "atom_residue_index",
            )
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"fusion_atom_residual model requires atom graph fields; missing {missing}.")
            atom_h = self.atom_branch.encode_atoms(
                batch["atom_features"], batch["atom_edge_index"], batch["atom_edge_features"]
            )
            residue_atom_h = self.atom_branch.residue_embeddings(
                atom_h=atom_h,
                atom_batch=batch["atom_batch"],
                atom_residue_index=batch["atom_residue_index"],
                batch_size=mask.shape[0],
                max_length=mask.shape[1],
            )
            atom_weight = torch.sigmoid(self.atom_residual_logit)
            fused_nodes = fused_nodes + atom_weight * self.atom_residual_projection(residue_atom_h)
            diagnostics["atom_residual_weight"] = atom_weight
        pooled = masked_mean(fused_nodes, mask)
        logits = self.classifier(self._classifier_input(pooled, batch)).squeeze(-1)
        return {"logits": logits, "embedding": pooled, **diagnostics}

