"""Monte Carlo search for PLDDT fusion-gate hyperparameters."""

from __future__ import annotations

import argparse
from copy import deepcopy
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides
from ghxtox.metrics import binary_metrics
from ghxtox.models import GHXToxModel
from ghxtox.utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEVICE,
    DEFAULT_TEST_PROCESSED,
    move_batch_to_device,
    resolve_device,
    save_json,
    set_seed,
)


def _score_config(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits_all = []
    labels_all = []
    gates = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            output = model(batch)
            logits_all.append(output["logits"].detach().cpu())
            labels_all.append(batch["labels"].detach().cpu())
            gates.append(output["global_gate"].detach().cpu())

    metrics = binary_metrics(torch.cat(logits_all), torch.cat(labels_all))
    metrics["mean_global_gate"] = float(torch.cat(gates).mean()) if gates else float("nan")
    return metrics


def optimize(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_config = checkpoint["config"]

    dataset = PeptideTensorDataset(args.processed, require_labels=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_peptides)

    trials = []
    for trial in range(1, args.trials + 1):
        candidate = deepcopy(base_config)
        model_cfg = candidate.setdefault("model", {})
        model_cfg["fusion_gate"] = "direct_plddt"
        model_cfg["ahp_initial_3d_weight"] = rng.uniform(args.weight_min, args.weight_max)
        model_cfg["plddt_gate_center"] = rng.uniform(args.center_min, args.center_max)
        model_cfg["plddt_gate_temperature"] = rng.uniform(args.temperature_min, args.temperature_max)

        metrics = _score_config(checkpoint, candidate, loader, device)
        row = {
            "trial": trial,
            "objective": float(metrics.get(args.objective, float("nan"))),
            "metrics": metrics,
            "model": {
                "fusion_gate": model_cfg["fusion_gate"],
                "ahp_initial_3d_weight": model_cfg["ahp_initial_3d_weight"],
                "plddt_gate_center": model_cfg["plddt_gate_center"],
                "plddt_gate_temperature": model_cfg["plddt_gate_temperature"],
            },
        }
        trials.append(row)
        print(
            f"trial={trial:04d} {args.objective}={row['objective']:.6f} "
            f"gate={metrics['mean_global_gate']:.3f}"
        )

    trials.sort(key=lambda item: item["objective"], reverse=True)
    payload = {
        "objective": args.objective,
        "best": trials[0] if trials else None,
        "trials": trials,
    }
    output = Path(args.output)
    save_json(payload, output)
    if trials:
        best = trials[0]
        print(f"Best {args.objective}={best['objective']:.6f}; saved search to {output}")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo optimize GHXTox PLDDT gate parameters.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processed", default=DEFAULT_TEST_PROCESSED, help="Labeled validation/test .pt file.")
    parser.add_argument("--output", default="runs/ghxtox/gate_search.json")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective", default="auprc", choices=["auprc", "auroc", "mcc", "f1", "balanced_accuracy"])
    parser.add_argument("--weight-min", type=float, default=0.2)
    parser.add_argument("--weight-max", type=float, default=0.9)
    parser.add_argument("--center-min", type=float, default=0.45)
    parser.add_argument("--center-max", type=float, default=0.9)
    parser.add_argument("--temperature-min", type=float, default=0.03)
    parser.add_argument("--temperature-max", type=float, default=0.35)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    optimize(args)


if __name__ == "__main__":
    main()
