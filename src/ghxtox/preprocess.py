"""Stage 1: convert FASTA peptides into tensorized GHXTox records."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ghxtox.fasta import read_fasta
from ghxtox.features import (
    encode_amino_acids,
    encode_functional_groups,
    residue_feature_matrix,
    sequence_global_features,
)
from ghxtox.geometry_features import chemical_structure_feature_matrix, structure_feature_matrix
from ghxtox.structure import make_structure_provider
from ghxtox.utils import DEFAULT_STRUCTURE_CACHE_DIR, DEFAULT_TRAIN_FASTA, DEFAULT_TRAIN_PROCESSED


def resolve_structure_cache_dir(input_path: str | Path, structure_cache_dir: str | Path) -> Path:
    """Choose split-specific structure caches when they exist."""

    base = Path(structure_cache_dir)
    name = Path(input_path).stem.lower()
    if "test1" in name and (base / "test1").exists():
        return base / "test1"
    if "test2" in name and (base / "test2").exists():
        return base / "test2"
    return base


def preprocess_fasta(
    input_path: str | Path,
    output_path: str | Path,
    structure_mode: str = "heuristic",
    structure_cache_dir: str | Path | None = None,
    max_length: int | None = None,
    missing_structure: str = "error",
) -> list[dict]:
    records = read_fasta(input_path)
    resolved_cache_dir = resolve_structure_cache_dir(
        input_path,
        structure_cache_dir or DEFAULT_STRUCTURE_CACHE_DIR,
    )
    provider = make_structure_provider(
        structure_mode,
        resolved_cache_dir,
        allow_cached_fallback=missing_structure == "heuristic",
    )
    examples: list[dict] = []

    for index, record in enumerate(records, start=1):
        sample_key = f"{record.sample_id}_{index}"
        sequence = record.sequence
        if max_length is not None and len(sequence) > max_length:
            sequence = sequence[:max_length]
        structure = provider.get(sample_key, sequence)
        length = len(sequence)
        coords = structure.coords[:length]
        plddt = structure.plddt[:length]
        backbone_coords = structure.backbone_coords
        backbone_mask = structure.backbone_mask
        functional_group_coords = structure.functional_group_coords
        functional_group_mask = structure.functional_group_mask
        if torch.is_tensor(backbone_coords):
            backbone_coords = backbone_coords[:length]
        if torch.is_tensor(backbone_mask):
            backbone_mask = backbone_mask[:length]
        if not torch.is_tensor(backbone_coords) or not torch.is_tensor(backbone_mask):
            backbone_coords = coords.unsqueeze(1).expand(-1, 5, -1).clone()
            backbone_mask = torch.zeros(length, 5, dtype=torch.bool)
            if length > 0:
                backbone_mask[:, 1] = True
        if torch.is_tensor(functional_group_coords):
            functional_group_coords = functional_group_coords[:length]
        if torch.is_tensor(functional_group_mask):
            functional_group_mask = functional_group_mask[:length]
        if not torch.is_tensor(functional_group_coords) or not torch.is_tensor(functional_group_mask):
            functional_group_coords = coords.clone()
            functional_group_mask = torch.zeros(length, dtype=torch.bool)
        if coords.shape[0] != length or plddt.shape[0] != length:
            raise ValueError(
                f"Structure length mismatch for {record.sample_id}: "
                f"sequence={length}, coords={coords.shape[0]}, plddt={plddt.shape[0]}"
            )
        if backbone_coords.shape[0] != length or backbone_mask.shape[0] != length:
            raise ValueError(
                f"Backbone length mismatch for {record.sample_id}: "
                f"sequence={length}, backbone_coords={backbone_coords.shape[0]}, "
                f"backbone_mask={backbone_mask.shape[0]}"
            )
        if functional_group_coords.shape[0] != length or functional_group_mask.shape[0] != length:
            raise ValueError(
                f"Functional-group length mismatch for {record.sample_id}: "
                f"sequence={length}, functional_group_coords={functional_group_coords.shape[0]}, "
                f"functional_group_mask={functional_group_mask.shape[0]}"
            )

        examples.append(
            {
                "sample_id": sample_key,
                "original_id": record.sample_id,
                "header": record.header,
                "sequence": sequence,
                "label": record.label,
                "aa_ids": encode_amino_acids(sequence),
                "group_ids": encode_functional_groups(sequence),
                "residue_features": residue_feature_matrix(sequence),
                "coords": coords,
                "plddt": plddt,
                "backbone_coords": backbone_coords,
                "backbone_mask": backbone_mask,
                "functional_group_coords": functional_group_coords,
                "functional_group_mask": functional_group_mask,
                "structure_features": structure_feature_matrix(coords, plddt),
                "global_features": sequence_global_features(sequence),
                "structure_source": structure.source,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": examples, "source": str(input_path)}, output_path)
    return examples


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess FASTA files for GHXTox.")
    parser.add_argument("--input", default=DEFAULT_TRAIN_FASTA, help="Input FASTA path.")
    parser.add_argument("--output", default=DEFAULT_TRAIN_PROCESSED, help="Output .pt tensor file.")
    parser.add_argument(
        "--structure-mode",
        default="cached",
        choices=["heuristic", "cached"],
        help="Use deterministic fallback coordinates or cached ESMFold outputs.",
    )
    parser.add_argument("--structure-cache-dir", default=DEFAULT_STRUCTURE_CACHE_DIR)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--missing-structure", choices=["error", "heuristic"], default="error")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    examples = preprocess_fasta(
        input_path=args.input,
        output_path=args.output,
        structure_mode=args.structure_mode,
        structure_cache_dir=args.structure_cache_dir,
        max_length=args.max_length,
        missing_structure=args.missing_structure,
    )
    labeled = sum(item["label"] is not None for item in examples)
    positives = sum(item["label"] == 1 for item in examples)
    negatives = sum(item["label"] == 0 for item in examples)
    print(
        f"Saved {len(examples)} records to {args.output} "
        f"(labeled={labeled}, pos={positives}, neg={negatives})."
    )


if __name__ == "__main__":
    main()

