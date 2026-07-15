"""Export per-sample prediction and uncertainty diagnostics for one expert."""

from __future__ import annotations

import argparse
import csv
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.oof import _confidence_diagnostics, _enable_mc_dropout
from ghxtox.utils import move_batch_to_device, resolve_device


def export_confidence_predictions(
    checkpoint_path: str | Path,
    processed_path: str | Path,
    output_csv: str | Path,
    device_name: str = "cuda",
    batch_size: int = 32,
    mc_samples: int = 5,
    seed: int = 24000,
) -> None:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = PeptideTensorDataset(processed_path, require_labels=True)
    required_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_dim, processed_path)
    modality = str(config.get("model", {}).get("modality", "fusion")).lower()
    include_structure = modality not in {"sequence_only", "atom_only", "sequence_atom"}
    include_atom = modality in {"atom_only", "sequence_atom", "fusion_atom_residual"}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(
            collate_peptides,
            include_structure=include_structure,
            include_atom=include_atom,
        ),
    )
    rows = []
    position = 0
    with torch.no_grad():
        for batch in loader:
            batch_on_device = move_batch_to_device(batch, device)
            model.eval()
            output = model(batch_on_device)
            logits = output["logits"].detach().cpu()
            probabilities = torch.sigmoid(logits)
            mc_probabilities = None
            if mc_samples > 0:
                torch.manual_seed(seed + position)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed + position)
                _enable_mc_dropout(model)
                mc_probabilities = torch.stack(
                    [torch.sigmoid(model(batch_on_device)["logits"]).detach().cpu() for _ in range(mc_samples)]
                )
                model.eval()
            diagnostics = _confidence_diagnostics(
                batch_on_device,
                {
                    "global_gate": output["global_gate"].detach().cpu()
                    if torch.is_tensor(output.get("global_gate"))
                    else None
                },
                probabilities,
                mc_probabilities,
            )
            for offset, (sample_id, sequence, label, logit, probability) in enumerate(
                zip(
                    batch["sample_id"],
                    batch["sequence"],
                    batch["labels"].tolist(),
                    logits.tolist(),
                    probabilities.tolist(),
                )
            ):
                row = {
                    "source_index": position + offset,
                    "sample_id": sample_id,
                    "sequence": sequence,
                    "label": int(label),
                    "logit": f"{float(logit):.9g}",
                    "toxicity_probability": f"{float(probability):.9g}",
                }
                for key, values in diagnostics.items():
                    row[key] = f"{float(values[offset]):.9g}"
                rows.append(row)
            position += len(logits)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export confidence diagnostics for a frozen expert.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=24000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_confidence_predictions(
        args.checkpoint,
        args.processed,
        args.output_csv,
        device_name=args.device,
        batch_size=args.batch_size,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
