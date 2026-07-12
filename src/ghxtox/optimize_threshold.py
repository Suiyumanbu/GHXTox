"""Optimize a probability threshold on a validation split for a trained GHXTox checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import DEFAULT_DEVICE, DEFAULT_TRAIN_PROCESSED, move_batch_to_device, resolve_device, save_json


def _validation_subset(dataset: PeptideTensorDataset, val_fraction: float, seed: int) -> Subset:
    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = max(1, len(dataset) - val_size)
    generator = torch.Generator().manual_seed(seed)
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return val_dataset


def _collect_logits(
    checkpoint_path: str | Path,
    processed_path: str | Path,
    device_name: str,
    batch_size: int,
    val_fraction: float | None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_plm_dim, processed_path)
    if val_fraction is None:
        train_cfg = config.get("train", {})
        val_fraction = float(train_cfg.get("val_fraction", 0.15))
    seed = int(config.get("seed", 42))
    eval_dataset = _validation_subset(dataset, val_fraction, seed)
    loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_peptides)

    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            logits_all.append(output["logits"].detach().cpu())
            labels_all.append(batch["labels"].detach().cpu())
    return torch.cat(logits_all), torch.cat(labels_all), checkpoint


def optimize_threshold(
    checkpoint: str | Path,
    processed: str | Path,
    output: str | Path,
    device: str,
    batch_size: int,
    metric: str,
    val_fraction: float | None = None,
) -> dict[str, float]:
    logits, labels, checkpoint_payload = _collect_logits(checkpoint, processed, device, batch_size, val_fraction)
    candidates = torch.linspace(0.01, 0.99, 99).tolist()
    probs = torch.sigmoid(logits)
    candidates.extend(float(x) for x in probs.unique().tolist())
    candidates = sorted(set(round(float(x), 6) for x in candidates if 0.0 < float(x) < 1.0))

    best_threshold = 0.5
    best_metrics = binary_metrics(logits, labels, threshold=best_threshold)
    best_score = float(best_metrics[metric])
    for threshold in candidates:
        metrics = binary_metrics(logits, labels, threshold=threshold)
        score = float(metrics[metric])
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    result = {
        "threshold": float(best_threshold),
        "metric": metric,
        "score": float(best_score),
        "num_validation_samples": float(labels.numel()),
        "checkpoint_epoch": float(checkpoint_payload.get("epoch", -1)),
        **{f"val_{key}": float(value) for key, value in best_metrics.items()},
    }
    save_json(result, output)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize a GHXTox probability threshold on the validation split.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed", default=DEFAULT_TRAIN_PROCESSED)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc"], default="mcc")
    parser.add_argument("--val-fraction", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = optimize_threshold(
        checkpoint=args.checkpoint,
        processed=args.processed,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        metric=args.metric,
        val_fraction=args.val_fraction,
    )
    print(f"threshold: {result['threshold']:.6f}")
    print(f"{result['metric']}: {result['score']:.6f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
