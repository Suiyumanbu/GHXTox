"""Stage 2: train the PLDDT-aware dual-modal toxicity model."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.folds import load_fold_indices
from ghxtox.nested_folds import load_nested_indices
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import (
    DEFAULT_DEVICE,
    DEFAULT_TRAIN_PROCESSED,
    load_json,
    move_batch_to_device,
    resolve_device,
    save_json,
    set_seed,
)


def _make_loaders(
    train_path: str | Path,
    val_path: str | Path | None,
    batch_size: int,
    val_fraction: float,
    seed: int,
    fold_manifest: str | Path | None = None,
    validation_fold: int = 0,
    nested_manifest: str | Path | None = None,
    outer_fold: int = 0,
    modality: str = "fusion",
) -> tuple[DataLoader, DataLoader]:
    train_dataset = PeptideTensorDataset(train_path, require_labels=True)
    if fold_manifest and nested_manifest:
        raise ValueError("--fold-manifest and --nested-manifest are mutually exclusive.")
    if nested_manifest:
        if val_path:
            raise ValueError("--val and --nested-manifest are mutually exclusive.")
        roles = load_nested_indices(nested_manifest, outer_fold, len(train_dataset))
        source_dataset = train_dataset
        train_dataset = Subset(source_dataset, roles["train"])
        val_dataset = Subset(source_dataset, roles["validation"])
    elif fold_manifest:
        if val_path:
            raise ValueError("--val and --fold-manifest are mutually exclusive.")
        train_indices, val_indices = load_fold_indices(
            fold_manifest, validation_fold, len(train_dataset)
        )
        source_dataset = train_dataset
        train_dataset = Subset(source_dataset, train_indices)
        val_dataset = Subset(source_dataset, val_indices)
    elif val_path:
        val_dataset = PeptideTensorDataset(val_path, require_labels=True)
    else:
        val_size = max(1, int(round(len(train_dataset) * val_fraction)))
        train_size = max(1, len(train_dataset) - val_size)
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

    modality = modality.lower()
    include_structure = modality not in {"sequence_only", "atom_only", "sequence_atom"}
    include_atom = modality in {
        "atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"
    }
    collate_fn = partial(
        collate_peptides,
        include_structure=include_structure,
        include_atom=include_atom,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def _labels_from_loader(loader: DataLoader) -> torch.Tensor:
    labels = []
    records = _records_from_dataset(loader.dataset)
    for item in records:
        labels.append(float(item["label"]))
    return torch.tensor(labels, dtype=torch.float32)


def _records_from_dataset(dataset: PeptideTensorDataset | Subset) -> list[dict[str, Any]]:
    if isinstance(dataset, Subset):
        return [dataset.dataset.records[index] for index in dataset.indices]
    return dataset.records


def _validate_loader_plm_features(loader: DataLoader, required_dim: int, source: str) -> None:
    validate_plm_feature_dim(_records_from_dataset(loader.dataset), required_dim, source)


def _pos_weight(train_loader: DataLoader, mode: str | float) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "auto":
        return torch.tensor(float(mode), dtype=torch.float32)
    labels = _labels_from_loader(train_loader)
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return negatives / positives


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        p_t = torch.where(labels > 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - p_t).clamp_min(1e-6).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = torch.where(
                labels > 0.5,
                torch.full_like(labels, self.alpha),
                torch.full_like(labels, 1.0 - self.alpha),
            )
            focal_factor = focal_factor * alpha_t
        return (bce * focal_factor).mean()


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.smoothing = min(max(float(smoothing), 0.0), 0.49)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().clone())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = labels * (1.0 - 2.0 * self.smoothing) + self.smoothing
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
        )


def _make_criterion(train_cfg: dict[str, Any], pos_weight: torch.Tensor | None) -> nn.Module:
    loss_name = str(train_cfg.get("loss", "bce")).lower()
    if loss_name == "bce":
        smoothing = float(train_cfg.get("label_smoothing", 0.0))
        if smoothing > 0.0:
            return SmoothedBCEWithLogitsLoss(smoothing=smoothing, pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "focal_bce":
        alpha = train_cfg.get("focal_alpha", None)
        return FocalBCEWithLogitsLoss(
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
            alpha=None if alpha is None else float(alpha),
            pos_weight=pos_weight,
        )
    raise ValueError(f"Unsupported train.loss={loss_name!r}. Expected 'bce' or 'focal_bce'.")


def _apply_coordinate_noise(
    batch: dict[str, Any],
    base_std: float,
    plddt_scale: float,
) -> dict[str, Any]:
    if base_std <= 0.0:
        return batch
    mask = batch["mask"].bool()
    confidence = batch["plddt"].clamp(0.0, 1.0)
    residue_std = base_std * (1.0 + plddt_scale * (1.0 - confidence))
    noise = torch.randn_like(batch["coords"]) * residue_std.unsqueeze(-1)
    noise = noise * mask.float().unsqueeze(-1)
    augmented = dict(batch)
    augmented["coords"] = batch["coords"] + noise
    if "backbone_coords" in batch:
        augmented["backbone_coords"] = batch["backbone_coords"] + noise.unsqueeze(2)
        augmented["coords"] = augmented["backbone_coords"][:, :, 1]
    if "functional_group_coords" in batch:
        augmented["functional_group_coords"] = batch["functional_group_coords"] + noise
    if "chemical_site_coords" in batch:
        augmented["chemical_site_coords"] = batch["chemical_site_coords"] + noise.unsqueeze(2)
    return augmented


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    labels = labels.view(-1).long()
    if labels.numel() <= 1 or labels.unique().numel() < 2:
        return embeddings.new_zeros(())

    features = torch.nn.functional.normalize(embeddings, dim=-1)
    logits = features @ features.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return embeddings.new_zeros(())

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    mean_log_prob_pos = (log_prob * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def _run_epoch(
    model: GHXToxModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    coord_noise_std: float = 0.0,
    coord_noise_plddt_scale: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.2,
    atom_residual_l1: float = 0.0,
    residual_base_aux_weight: float = 0.0,
    residual_delta_l1: float = 0.0,
    conformer_delta_l1: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    logits_all = []
    labels_all = []
    gates = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        if labels is None:
            continue
        if training:
            batch = _apply_coordinate_noise(batch, coord_noise_std, coord_noise_plddt_scale)
            optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        logits = output["logits"]
        loss = criterion(logits, labels)
        if residual_base_aux_weight > 0.0 and "base_logits" in output:
            loss = loss + residual_base_aux_weight * criterion(output["base_logits"], labels)
        if residual_delta_l1 > 0.0 and "atom_delta" in output and "spatial_delta" in output:
            residual_size = output["atom_delta"].abs().mean() + output["spatial_delta"].abs().mean()
            loss = loss + residual_delta_l1 * residual_size
        if conformer_delta_l1 > 0.0 and "conformer_delta" in output:
            available = output.get("conformer_available")
            if available is not None and available.sum() > 0:
                residual_size = (output["conformer_delta"].abs() * available).sum() / available.sum()
                loss = loss + conformer_delta_l1 * residual_size
        if training and atom_residual_l1 > 0.0 and "atom_residual_weight" in output:
            loss = loss + atom_residual_l1 * output["atom_residual_weight"].abs()
        if training and contrastive_weight > 0.0:
            contrastive = _supervised_contrastive_loss(
                output["embedding"],
                labels,
                temperature=contrastive_temperature,
            )
            loss = loss + contrastive_weight * contrastive
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
        gates.append(output["global_gate"].detach().cpu())

    if total_items == 0:
        raise RuntimeError("No labeled samples were available in the data loader.")
    logits_cat = torch.cat(logits_all)
    labels_cat = torch.cat(labels_all)
    result = binary_metrics(logits_cat, labels_cat)
    result["loss"] = total_loss / total_items
    result["mean_global_gate"] = float(torch.cat(gates).mean()) if gates else float("nan")
    return result


def train(config: dict, args: argparse.Namespace) -> Path:
    set_seed(int(config.get("seed", 42)))
    train_cfg = config["train"]
    device = resolve_device(args.device)
    train_loader, val_loader = _make_loaders(
        train_path=args.train,
        val_path=args.val,
        batch_size=int(args.batch_size or train_cfg["batch_size"]),
        val_fraction=float(train_cfg.get("val_fraction", 0.15)),
        seed=int(config.get("seed", 42)),
        fold_manifest=args.fold_manifest,
        validation_fold=args.fold,
        nested_manifest=args.nested_manifest,
        outer_fold=args.outer_fold,
        modality=str(config.get("model", {}).get("modality", "fusion")),
    )
    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    _validate_loader_plm_features(train_loader, required_plm_dim, args.train)
    _validate_loader_plm_features(val_loader, required_plm_dim, args.val or args.train)

    model = GHXToxModel(config).to(device)
    initial_checkpoint = args.initial_checkpoint or train_cfg.get("initial_checkpoint")
    initial_checkpoint_metrics: dict[str, Any] = {}
    if initial_checkpoint:
        initial_checkpoint = str(initial_checkpoint).format(
            fold=args.fold,
            outer_fold=args.outer_fold,
        )
        checkpoint = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        initial_checkpoint_metrics = dict(checkpoint.get("val_metrics", {}))
        state = checkpoint.get("model_state", checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected:
            raise ValueError(f"Unexpected keys in initial checkpoint: {unexpected[:10]}")
        allowed_missing_prefixes = tuple(train_cfg.get("allowed_missing_prefixes", []))
        disallowed_missing = [
            key for key in missing if not key.startswith(allowed_missing_prefixes)
        ] if allowed_missing_prefixes else missing
        if disallowed_missing:
            raise ValueError(f"Missing keys in initial checkpoint: {disallowed_missing[:10]}")
        print(
            f"Initialized from {initial_checkpoint}; "
            f"new_parameters={len(missing)} unexpected_parameters={len(unexpected)}"
        )
    trainable_prefixes = tuple(train_cfg.get("trainable_prefixes", []))
    if trainable_prefixes:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(trainable_prefixes)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names:
            raise ValueError(f"No parameters matched trainable_prefixes={trainable_prefixes}.")
        print(
            f"Frozen-base training: trainable_tensors={len(trainable_names)} "
            f"prefixes={trainable_prefixes}"
        )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    pos_weight = _pos_weight(train_loader, args.pos_weight or train_cfg.get("pos_weight", "auto"))
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = _make_criterion(train_cfg, pos_weight)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate or train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    if args.full_data_epochs is not None:
        if args.val or args.fold_manifest or args.nested_manifest:
            raise ValueError("--full-data-epochs cannot be combined with validation manifests.")
        full_modality = str(config.get("model", {}).get("modality", "fusion")).lower()
        sequence_only = full_modality == "sequence_only"
        include_atom = full_modality in {
            "atom_only", "sequence_atom", "fusion_atom_residual", "residual_experts"
        }
        full_dataset = PeptideTensorDataset(args.train, require_labels=True)
        full_loader = DataLoader(
            full_dataset,
            batch_size=int(args.batch_size or train_cfg["batch_size"]),
            shuffle=True,
            collate_fn=partial(
                collate_peptides,
                include_structure=not sequence_only,
                include_atom=include_atom,
            ),
        )
        _validate_loader_plm_features(full_loader, required_plm_dim, args.train)
        full_pos_weight = _pos_weight(full_loader, args.pos_weight or train_cfg.get("pos_weight", "auto"))
        if full_pos_weight is not None:
            full_pos_weight = full_pos_weight.to(device)
        criterion = _make_criterion(train_cfg, full_pos_weight)
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(args.learning_rate or train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        fixed_epochs = int(args.full_data_epochs)
        if fixed_epochs <= 0:
            raise ValueError("--full-data-epochs must be positive.")
        history = []
        coord_noise_std = float(train_cfg.get("coord_noise_std", 0.0))
        coord_noise_plddt_scale = float(train_cfg.get("coord_noise_plddt_scale", 0.0))
        contrastive_weight = float(train_cfg.get("contrastive_weight", 0.0))
        contrastive_temperature = float(train_cfg.get("contrastive_temperature", 0.2))
        atom_residual_l1 = float(train_cfg.get("atom_residual_l1", 0.0))
        for epoch in range(1, fixed_epochs + 1):
            train_metrics = _run_epoch(
                model,
                full_loader,
                criterion,
                device,
                optimizer,
                coord_noise_std=coord_noise_std,
                coord_noise_plddt_scale=coord_noise_plddt_scale,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                atom_residual_l1=atom_residual_l1,
            )
            history.append({"epoch": epoch, "train": train_metrics})
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
                f"train_auprc={train_metrics['auprc']:.4f} train_mcc={train_metrics['mcc']:.4f}"
            )
        best_path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": fixed_epochs,
                "train_metrics": history[-1]["train"],
                "monitor": "fixed_epoch_full_data_refit",
                "refit_protocol": {
                    "full_training_data": True,
                    "fixed_epochs": fixed_epochs,
                    "epoch_source": "median best epoch from fixed group-aware cross-validation",
                },
            },
            best_path,
        )
        save_json({"history": history}, output_dir / "history.json")
        print(f"Full-data fixed-epoch checkpoint saved to {best_path}")
        return best_path

    monitor = args.monitor or train_cfg.get("monitor", "loss")
    monitor_modes = {
        "loss": "min",
        "accuracy": "max",
        "balanced_accuracy": "max",
        "precision": "max",
        "recall": "max",
        "f1": "max",
        "mcc": "max",
        "auroc": "max",
        "auprc": "max",
    }
    monitor_mode = monitor_modes[monitor]
    best_score = float("inf") if monitor_mode == "min" else -float("inf")
    best_path = output_dir / "best_model.pt"
    patience = int(train_cfg.get("early_stop_patience", 8))
    stale_epochs = 0
    history = []
    epochs = int(args.epochs or train_cfg["epochs"])
    coord_noise_std = float(train_cfg.get("coord_noise_std", 0.0))
    coord_noise_plddt_scale = float(train_cfg.get("coord_noise_plddt_scale", 0.0))
    contrastive_weight = float(train_cfg.get("contrastive_weight", 0.0))
    contrastive_temperature = float(train_cfg.get("contrastive_temperature", 0.2))
    atom_residual_l1 = float(train_cfg.get("atom_residual_l1", 0.0))
    residual_base_aux_weight = float(train_cfg.get("residual_base_aux_weight", 0.0))
    residual_delta_l1 = float(train_cfg.get("residual_delta_l1", 0.0))
    conformer_delta_l1 = float(train_cfg.get("conformer_delta_l1", 0.0))
    selection_mode = str(train_cfg.get("selection_mode", "monitor")).lower()
    if selection_mode not in {"monitor", "pareto"}:
        raise ValueError("train.selection_mode must be 'monitor' or 'pareto'.")
    pareto_metric = str(train_cfg.get("pareto_metric", "auprc"))
    pareto_constraint_metric = str(train_cfg.get("pareto_constraint_metric", "mcc"))
    pareto_constraint_margin = float(train_cfg.get("pareto_constraint_margin", 0.0))
    pareto_constraint_value = train_cfg.get("pareto_constraint_value")
    if pareto_constraint_value is None:
        pareto_constraint_value = initial_checkpoint_metrics.get(pareto_constraint_metric)
    if selection_mode == "pareto" and pareto_constraint_value is None:
        raise ValueError(
            "Pareto selection requires a constraint value from the initial checkpoint "
            "or train.pareto_constraint_value."
        )
    if pareto_metric not in monitor_modes or pareto_constraint_metric not in monitor_modes:
        raise ValueError("Pareto metric names must be supported validation metrics.")
    if selection_mode == "pareto":
        if monitor_modes[pareto_metric] != "max":
            raise ValueError("Pareto target metric must be maximized.")
        best_score = -float("inf")
    auxiliary_best: dict[str, float] = {"mcc": -float("inf"), "auprc": -float("inf")}
    save_auxiliary = bool(train_cfg.get("save_auxiliary_checkpoints", False))

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            coord_noise_std=coord_noise_std,
            coord_noise_plddt_scale=coord_noise_plddt_scale,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
            atom_residual_l1=atom_residual_l1,
            residual_base_aux_weight=residual_base_aux_weight,
            residual_delta_l1=residual_delta_l1,
            conformer_delta_l1=conformer_delta_l1,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"val_mcc={val_metrics['mcc']:.4f} "
            f"gate={val_metrics['mean_global_gate']:.3f}"
        )

        if save_auxiliary:
            for metric_name in ("mcc", "auprc"):
                metric_value = float(val_metrics[metric_name])
                if metric_value > auxiliary_best[metric_name]:
                    auxiliary_best[metric_name] = metric_value
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "config": config,
                            "epoch": epoch,
                            "val_metrics": val_metrics,
                            "monitor": metric_name,
                            "monitor_mode": "max",
                            "monitor_value": metric_value,
                        },
                        output_dir / f"best_{metric_name}_model.pt",
                    )
        if selection_mode == "pareto":
            constraint_threshold = float(pareto_constraint_value) + pareto_constraint_margin
            feasible = float(val_metrics[pareto_constraint_metric]) >= constraint_threshold
            current_score = float(val_metrics[pareto_metric])
            improved = feasible and current_score > best_score
            row["selection"] = {
                "mode": "pareto",
                "target_metric": pareto_metric,
                "target_value": current_score,
                "constraint_metric": pareto_constraint_metric,
                "constraint_threshold": constraint_threshold,
                "constraint_value": float(val_metrics[pareto_constraint_metric]),
                "feasible": feasible,
            }
        else:
            current_score = float(val_metrics[monitor])
            improved = (
                current_score < best_score
                if monitor_mode == "min"
                else current_score > best_score
            )
            row["selection"] = {
                "mode": "monitor",
                "metric": monitor,
                "value": current_score,
            }
        if improved:
            best_score = current_score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "monitor": pareto_metric if selection_mode == "pareto" else monitor,
                    "monitor_mode": (
                        monitor_modes[pareto_metric]
                        if selection_mode == "pareto"
                        else monitor_mode
                    ),
                    "monitor_value": current_score,
                    "selection_mode": selection_mode,
                    "pareto_constraint": (
                        {
                            "metric": pareto_constraint_metric,
                            "threshold": float(pareto_constraint_value)
                            + pareto_constraint_margin,
                            "observed": float(val_metrics[pareto_constraint_metric]),
                        }
                        if selection_mode == "pareto"
                        else None
                    ),
                },
                best_path,
            )
        else:
            # Do not stop a Pareto run before it has produced at least one
            # checkpoint satisfying the control-MCC constraint.
            if selection_mode != "pareto" or best_score > -float("inf"):
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

    save_json({"history": history}, output_dir / "history.json")
    if selection_mode == "pareto" and best_score == -float("inf"):
        raise RuntimeError(
            "No validation epoch satisfied the Pareto constraint; "
            "auxiliary checkpoints and history were retained for diagnosis."
        )
    selected_metric = pareto_metric if selection_mode == "pareto" else monitor
    print(f"Best checkpoint saved to {best_path} ({selected_metric}={best_score:.6f})")
    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GHXTox.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PROCESSED, help="Preprocessed training .pt file.")
    parser.add_argument("--val", default=None, help="Optional preprocessed validation .pt file.")
    parser.add_argument("--fold-manifest", default=None, help="Fixed fold CSV created by ghxtox.folds.")
    parser.add_argument("--fold", type=int, default=0, help="Validation fold used with --fold-manifest.")
    parser.add_argument("--nested-manifest", default=None, help="Nested role manifest created by ghxtox.nested_folds.")
    parser.add_argument("--outer-fold", type=int, default=0, help="Outer fold used with --nested-manifest.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--initial-checkpoint",
        default=None,
        help="Optional warm-start checkpoint overriding train.initial_checkpoint.",
    )
    parser.add_argument("--output-dir", default="runs/plm_sequence_only_esm2_mcc")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--pos-weight", default=None, help="'auto', 'none', or a numeric weight.")
    parser.add_argument(
        "--full-data-epochs",
        type=int,
        default=None,
        help="Refit on all training records for a fixed CV-selected number of epochs.",
    )
    parser.add_argument(
        "--monitor",
        choices=["loss", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"],
        default=None,
        help="Validation metric used for checkpoint selection and early stopping.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_json(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
