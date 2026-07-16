"""Predict toxicity probabilities from preprocessed files or FASTA input."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ghxtox.data import PeptideTensorDataset, collate_peptides, validate_plm_feature_dim
from ghxtox.models import GHXToxModel
from ghxtox.preprocess import preprocess_fasta
from ghxtox.utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DEVICE,
    DEFAULT_STRUCTURE_CACHE_DIR,
    DEFAULT_TEST_FASTA,
    DEFAULT_THRESHOLD,
    move_batch_to_device,
    resolve_inference_checkpoint,
    resolve_device,
)


def _prepare_input(args: argparse.Namespace) -> Path:
    if args.processed:
        return Path(args.processed)
    if not args.input_fasta:
        raise ValueError("Either --processed or --input-fasta is required.")
    output = Path(args.temp_processed)
    preprocess_fasta(
        input_path=args.input_fasta,
        output_path=output,
        structure_mode=args.structure_mode,
        structure_cache_dir=args.structure_cache_dir,
        max_length=args.max_length,
    )
    return output


def predict(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    processed_path = _prepare_input(args)
    dataset = PeptideTensorDataset(processed_path, require_labels=False)
    checkpoint, selected_checkpoint, threshold, fallback_used = resolve_inference_checkpoint(
        args.checkpoint,
        dataset.records,
        device,
        requested_threshold=args.threshold,
    )
    config = checkpoint["config"]
    model = GHXToxModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    required_plm_dim = int(config.get("model", {}).get("plm_embedding_dim", 0))
    validate_plm_feature_dim(dataset.records, required_plm_dim, processed_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_peptides)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "sequence",
                "toxicity_probability",
                "prediction",
                "global_3d_gate",
                "model_checkpoint",
                "fallback_used",
                "decision_threshold",
            ],
        )
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                output = model(batch)
                probs = torch.sigmoid(output["logits"]).detach().cpu()
                gates = output["global_gate"].detach().cpu()
                for sample_id, sequence, prob, gate in zip(
                    batch["sample_id"],
                    batch["sequence"],
                    probs.tolist(),
                    gates.tolist(),
                ):
                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "sequence": sequence,
                            "toxicity_probability": f"{prob:.9g}",
                            "prediction": int(prob >= threshold),
                            "global_3d_gate": f"{gate:.9g}",
                            "model_checkpoint": selected_checkpoint,
                            "fallback_used": int(fallback_used),
                            "decision_threshold": f"{threshold:.9g}",
                        }
                    )
    if fallback_used:
        print(
            "Chemical-site tensors were unavailable; "
            f"used fallback checkpoint {selected_checkpoint}."
        )
    print(f"Predictions saved to {output_path}")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict with GHXTox.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--processed", default=None, help="Preprocessed .pt file.")
    parser.add_argument("--input-fasta", default=DEFAULT_TEST_FASTA, help="Raw FASTA file to preprocess before prediction.")
    parser.add_argument("--output", default="runs/ghxtox/predictions.csv")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Decision threshold. Defaults to checkpoint metadata ({DEFAULT_THRESHOLD} for 3D-v2).",
    )
    parser.add_argument("--temp-processed", default="runs/ghxtox/predict_input.pt")
    parser.add_argument("--structure-mode", default="heuristic", choices=["heuristic", "cached"])
    parser.add_argument("--structure-cache-dir", default=DEFAULT_STRUCTURE_CACHE_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    predict(args)


if __name__ == "__main__":
    main()
