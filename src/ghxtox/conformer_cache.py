"""Convert multi-model peptide PDB outputs into GHXTox ensemble caches."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from ghxtox.chemical_sites import parse_chemical_sites
from ghxtox.esmfold_cache import parse_esmfold_pdb
from ghxtox.fasta import read_fasta
from ghxtox.features import clean_sequence


def pdb_model_blocks(path: str | Path) -> list[list[str]]:
    """Return one ATOM-containing block per PDB MODEL (or one whole-file block)."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    has_models = any(line.startswith("MODEL") for line in lines)
    if not has_models:
        return [lines] if any(line.startswith("ATOM") for line in lines) else []

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("MODEL"):
            if current and any(item.startswith("ATOM") for item in current):
                blocks.append(current)
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current is not None and any(item.startswith("ATOM") for item in current):
                blocks.append(current)
            current = None
            continue
        if current is not None:
            current.append(line)
    if current and any(item.startswith("ATOM") for item in current):
        blocks.append(current)
    return blocks


def _parse_block(lines: list[str]):
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as handle:
            handle.write("\n".join(lines))
            handle.write("\nEND\n")
            temporary_path = handle.name
        parsed = parse_esmfold_pdb(temporary_path)
        sites = parse_chemical_sites(temporary_path)
        return parsed, sites
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _kabsch_align(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    target = reference - reference.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(centered.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return centered @ rotation


def ensemble_descriptors(coords: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate alignment-aware flexibility and contact-occupancy descriptors."""

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected conformer coordinates [K,L,3], got {coords.shape}.")
    reference = coords[0]
    aligned = np.stack([_kabsch_align(item, reference) for item in coords]).astype(np.float32)
    ensemble_mean = aligned.mean(axis=0)
    residue_rmsf = np.sqrt(np.mean(np.sum((aligned - ensemble_mean) ** 2, axis=-1), axis=0))
    conformer_rmsd = np.sqrt(np.mean(np.sum((aligned - ensemble_mean) ** 2, axis=-1), axis=1))
    centered = coords - coords.mean(axis=1, keepdims=True)
    radius_gyration = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))
    pair_distances = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)
    result: dict[str, np.ndarray] = {
        "aligned_coords": aligned,
        "ensemble_mean_coords": ensemble_mean.astype(np.float32),
        "residue_rmsf": residue_rmsf.astype(np.float32),
        "conformer_rmsd": conformer_rmsd.astype(np.float32),
        "radius_gyration": radius_gyration.astype(np.float32),
        "pair_distance_mean": pair_distances.mean(axis=0).astype(np.float32),
        "pair_distance_std": pair_distances.std(axis=0).astype(np.float32),
    }
    eye = np.eye(coords.shape[1], dtype=bool)
    for cutoff in (6.0, 8.0, 10.0):
        occupancy = (pair_distances < cutoff).mean(axis=0)
        occupancy[eye] = 0.0
        result[f"contact_occupancy_{int(cutoff)}"] = occupancy.astype(np.float32)
    return result


def build_conformer_cache(
    fasta: str | Path,
    pdb_root: str | Path,
    output_root: str | Path,
    pdb_pattern: str = "{sample_id}/{sequence}.pdb",
    source: str = "pepflow",
    confidence_mode: str = "imputed",
    imputed_confidence: float = 0.7,
    max_conformers: int | None = None,
    on_missing: str = "error",
    on_mismatch: str = "error",
    sample_id_mode: str = "indexed",
) -> dict[str, Any]:
    pdb_root = Path(pdb_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for index, record in enumerate(read_fasta(fasta), start=1):
        sequence = clean_sequence(record.sequence)
        key = record.sample_id if sample_id_mode == "header" else f"{record.sample_id}_{index}"
        relative = pdb_pattern.format(
            sample_id=key,
            original_id=record.sample_id,
            index=index,
            sequence=sequence,
        )
        pdb_path = pdb_root / relative
        if not pdb_path.exists():
            sample_rows.append({"sample_id": key, "status": "missing", "pdb_path": str(pdb_path)})
            if on_missing == "error":
                raise FileNotFoundError(f"Missing ensemble PDB for {key}: {pdb_path}")
            continue
        blocks = pdb_model_blocks(pdb_path)
        if max_conformers is not None:
            blocks = blocks[: max(int(max_conformers), 0)]
        coords_all: list[np.ndarray] = []
        sample_dir = output_root / key
        sample_dir.mkdir(parents=True, exist_ok=True)
        mismatch_count = 0
        for model_index, block in enumerate(blocks):
            parsed, sites = _parse_block(block)
            if parsed.sequence != sequence or sites.sequence != sequence:
                mismatch_count += 1
                if on_mismatch == "error":
                    raise ValueError(
                        f"Sequence mismatch for {key} conformer {model_index}: "
                        f"fasta={sequence}, pdb={parsed.sequence}, sites={sites.sequence}"
                    )
                continue
            if confidence_mode == "pdb":
                plddt = parsed.plddt.astype(np.float32)
                confidence_imputed = False
            else:
                plddt = np.full(len(sequence), float(imputed_confidence), dtype=np.float32)
                confidence_imputed = True
            output_path = sample_dir / f"conformer_{model_index:04d}.npz"
            np.savez_compressed(
                output_path,
                coords=parsed.coords.astype(np.float32),
                plddt=plddt,
                backbone_coords=parsed.backbone_coords.astype(np.float32),
                backbone_mask=parsed.backbone_mask.astype(np.bool_),
                functional_group_coords=parsed.functional_group_coords.astype(np.float32),
                functional_group_mask=parsed.functional_group_mask.astype(np.bool_),
                chemical_site_coords=sites.coords.astype(np.float32),
                chemical_site_types=sites.types.astype(np.float32),
                chemical_site_orientations=sites.orientations.astype(np.float32),
                chemical_site_orientation_mask=sites.orientation_mask.astype(np.bool_),
                chemical_site_mask=sites.mask.astype(np.bool_),
                sequence=np.asarray(sequence),
                source=np.asarray(source),
                source_pdb=np.asarray(str(pdb_path)),
                source_model_index=np.asarray(model_index),
                confidence_imputed=np.asarray(confidence_imputed),
            )
            coords_all.append(parsed.coords.astype(np.float32))
            manifest_rows.append(
                {
                    "source_index": index - 1,
                    "sample_id": key,
                    "original_id": record.sample_id,
                    "sequence": sequence,
                    "label": record.label,
                    "conformer_index": model_index,
                    "source": source,
                    "confidence_imputed": int(confidence_imputed),
                    "cache_path": str(output_path),
                    "source_pdb": str(pdb_path),
                }
            )
        if coords_all:
            descriptors = ensemble_descriptors(np.stack(coords_all))
            np.savez_compressed(sample_dir / "ensemble_summary.npz", **descriptors)
            sample_rows.append(
                {
                    "sample_id": key,
                    "status": "converted",
                    "conformers": len(coords_all),
                    "mismatches": mismatch_count,
                    "mean_radius_gyration": float(descriptors["radius_gyration"].mean()),
                    "mean_residue_rmsf": float(descriptors["residue_rmsf"].mean()),
                    "pdb_path": str(pdb_path),
                }
            )
        else:
            sample_rows.append(
                {
                    "sample_id": key,
                    "status": "no_valid_conformer",
                    "conformers": 0,
                    "mismatches": mismatch_count,
                    "pdb_path": str(pdb_path),
                }
            )

    manifest_path = output_root / "conformer_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    status_counts: dict[str, int] = {}
    for row in sample_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "method": source,
        "fasta": str(Path(fasta)),
        "pdb_root": str(pdb_root),
        "pdb_pattern": pdb_pattern,
        "sample_id_mode": sample_id_mode,
        "confidence_mode": confidence_mode,
        "imputed_confidence": float(imputed_confidence) if confidence_mode == "imputed" else None,
        "status_counts": status_counts,
        "samples_with_conformers": sum(row["status"] == "converted" for row in sample_rows),
        "total_conformers": len(manifest_rows),
        "manifest": str(manifest_path),
        "samples": sample_rows,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--pdb-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pdb-pattern", default="{sample_id}/{sequence}.pdb")
    parser.add_argument("--source", default="pepflow")
    parser.add_argument("--confidence-mode", choices=["imputed", "pdb"], default="imputed")
    parser.add_argument("--imputed-confidence", type=float, default=0.7)
    parser.add_argument("--max-conformers", type=int)
    parser.add_argument("--on-missing", choices=["error", "skip"], default="error")
    parser.add_argument("--on-mismatch", choices=["error", "skip"], default="error")
    parser.add_argument("--sample-id-mode", choices=["indexed", "header"], default="indexed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_conformer_cache(
        fasta=args.fasta,
        pdb_root=args.pdb_root,
        output_root=args.output_root,
        pdb_pattern=args.pdb_pattern,
        source=args.source,
        confidence_mode=args.confidence_mode,
        imputed_confidence=args.imputed_confidence,
        max_conformers=args.max_conformers,
        on_missing=args.on_missing,
        on_mismatch=args.on_mismatch,
        sample_id_mode=args.sample_id_mode,
    )
    print(json.dumps({key: summary[key] for key in ("status_counts", "total_conformers")}, indent=2))


if __name__ == "__main__":
    main()
