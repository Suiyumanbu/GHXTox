"""Build GHXTox structure caches from ESMFold PDB outputs.

ESMFold writes predicted pLDDT values into the PDB B-factor column. GHXTox keeps
the legacy C-alpha trace and also persists N/CA/C/O plus a heavy side-chain
centroid for each residue so later models can use full residue geometry without
reparsing PDB files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ghxtox.fasta import FastaRecord, read_fasta
from ghxtox.features import clean_sequence
from ghxtox.utils import DEFAULT_STRUCTURE_CACHE_DIR, DEFAULT_TRAIN_FASTA


THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

BACKBONE_ATOMS = ("N", "CA", "C", "O")
BACKBONE_INDEX = {name: index for index, name in enumerate(BACKBONE_ATOMS)}
SIDECHAIN_INDEX = len(BACKBONE_ATOMS)


@dataclass(frozen=True)
class ParsedPdb:
    sequence: str
    coords: np.ndarray
    plddt: np.ndarray
    backbone_coords: np.ndarray
    backbone_mask: np.ndarray


def sample_key(record: FastaRecord, index: int) -> str:
    """Return the same key used by ghxtox.preprocess."""

    return f"{record.sample_id}_{index}"


def _empty_backbone(length: int) -> tuple[np.ndarray, np.ndarray]:
    backbone_coords = np.zeros((length, len(BACKBONE_ATOMS) + 1, 3), dtype=np.float32)
    backbone_mask = np.zeros((length, len(BACKBONE_ATOMS) + 1), dtype=bool)
    return backbone_coords, backbone_mask


def _finalize_residue(entry: dict[str, Any]) -> tuple[str, np.ndarray, float, np.ndarray, np.ndarray]:
    atoms: dict[str, np.ndarray] = entry["atoms"]
    ca = atoms.get("CA")
    if ca is None:
        raise ValueError(f"Residue {entry['residue_key']} is missing CA coordinates.")

    residue_coords, residue_mask = _empty_backbone(1)
    residue_coords = residue_coords[0]
    residue_mask = residue_mask[0]
    for atom_name, atom_index in BACKBONE_INDEX.items():
        coord = atoms.get(atom_name)
        if coord is not None:
            residue_coords[atom_index] = coord
            residue_mask[atom_index] = True

    sidechain_coords = entry["sidechain_coords"]
    if sidechain_coords:
        residue_coords[SIDECHAIN_INDEX] = np.mean(np.stack(sidechain_coords, axis=0), axis=0)
        residue_mask[SIDECHAIN_INDEX] = True
    else:
        residue_coords[SIDECHAIN_INDEX] = ca

    plddt = entry["plddt"]
    if plddt is None:
        plddt = float(entry["plddt_fallback"])
    return entry["residue_one_letter"], ca, plddt, residue_coords, residue_mask


def parse_esmfold_pdb(path: str | Path, chain: str | None = None) -> ParsedPdb:
    residues: list[dict[str, Any]] = []
    residue_index: dict[tuple[str, str, str], int] = {}

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            if chain is not None and chain_id != chain:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = (chain_id, resseq, icode)
            index = residue_index.get(residue_key)
            if index is None:
                resname = line[17:20].strip().upper()
                index = len(residues)
                residue_index[residue_key] = index
                residues.append(
                    {
                        "residue_key": residue_key,
                        "residue_one_letter": THREE_TO_ONE.get(resname, "X"),
                        "atoms": {},
                        "sidechain_coords": [],
                        "plddt": None,
                        "plddt_fallback": float(line[60:66]),
                    }
                )
            entry = residues[index]
            atoms = entry["atoms"]
            if atom_name in atoms:
                continue
            coord = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float32,
            )
            atoms[atom_name] = coord
            if atom_name == "CA":
                entry["plddt"] = float(line[60:66])
            if atom_name not in BACKBONE_INDEX and atom_name != "OXT" and not atom_name.startswith("H"):
                entry["sidechain_coords"].append(coord)

    if not residues:
        chain_note = f" for chain {chain}" if chain is not None else ""
        raise ValueError(f"No CA atoms found in {path}{chain_note}.")

    residues_out: list[str] = []
    coords: list[np.ndarray] = []
    plddt: list[float] = []
    backbone_coords, backbone_mask = _empty_backbone(len(residues))
    for index, entry in enumerate(residues):
        residue_one_letter, ca, residue_plddt, residue_backbone_coords, residue_backbone_mask = _finalize_residue(entry)
        residues_out.append(residue_one_letter)
        coords.append(ca)
        plddt.append(residue_plddt)
        backbone_coords[index] = residue_backbone_coords
        backbone_mask[index] = residue_backbone_mask

    plddt_array = np.asarray(plddt, dtype=np.float32)
    if plddt_array.size and float(plddt_array.max()) > 1.5:
        plddt_array = plddt_array / 100.0

    coords_array = np.asarray(coords, dtype=np.float32)
    center = coords_array.mean(axis=0, keepdims=True)
    coords_array = coords_array - center
    backbone_coords = backbone_coords - center.reshape(1, 1, 3)
    return ParsedPdb(
        sequence="".join(residues_out),
        coords=coords_array,
        plddt=np.clip(plddt_array, 0.0, 1.0),
        backbone_coords=backbone_coords,
        backbone_mask=backbone_mask,
    )


def _format_pdb_name(pattern: str, record: FastaRecord, index: int) -> str:
    key = sample_key(record, index)
    return pattern.format(
        sample_id=key,
        original_id=record.sample_id,
        index=index,
    )


def build_structure_cache(
    fasta_path: str | Path,
    pdb_dir: str | Path,
    output_dir: str | Path,
    pdb_pattern: str = "{sample_id}.pdb",
    chain: str | None = None,
    max_length: int | None = 128,
    on_missing: str = "error",
    on_mismatch: str = "error",
) -> dict[str, int]:
    records = read_fasta(fasta_path)
    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"saved": 0, "missing": 0, "mismatch": 0}
    for index, record in enumerate(records, start=1):
        key = sample_key(record, index)
        sequence = clean_sequence(record.sequence)
        if max_length is not None and len(sequence) > max_length:
            sequence = sequence[:max_length]

        pdb_path = pdb_dir / _format_pdb_name(pdb_pattern, record, index)
        if not pdb_path.exists():
            stats["missing"] += 1
            message = f"Missing PDB for {key}: {pdb_path}"
            if on_missing == "skip":
                print(f"[skip] {message}")
                continue
            raise FileNotFoundError(message)

        parsed = parse_esmfold_pdb(pdb_path, chain=chain)
        if parsed.coords.shape[0] != len(sequence):
            stats["mismatch"] += 1
            message = (
                f"Length mismatch for {key}: fasta={len(sequence)}, "
                f"pdb_ca={parsed.coords.shape[0]}, pdb={pdb_path}"
            )
            if on_mismatch == "skip":
                print(f"[skip] {message}")
                continue
            if on_mismatch == "crop" and parsed.coords.shape[0] >= len(sequence):
                coords = parsed.coords[: len(sequence)]
                plddt = parsed.plddt[: len(sequence)]
                backbone_coords = parsed.backbone_coords[: len(sequence)]
                backbone_mask = parsed.backbone_mask[: len(sequence)]
            else:
                raise ValueError(message)
        else:
            coords = parsed.coords
            plddt = parsed.plddt
            backbone_coords = parsed.backbone_coords
            backbone_mask = parsed.backbone_mask

        if parsed.sequence[: len(sequence)] != sequence[: len(parsed.sequence)]:
            print(
                f"[warn] Sequence differs for {key}: "
                f"fasta={sequence[:20]}..., pdb={parsed.sequence[:20]}..."
            )

        np.savez_compressed(
            output_dir / f"{key}.npz",
            coords=coords.astype(np.float32),
            plddt=plddt.astype(np.float32),
            backbone_coords=backbone_coords.astype(np.float32),
            backbone_mask=backbone_mask.astype(np.bool_),
            sequence=np.asarray(sequence),
            source_pdb=np.asarray(str(pdb_path)),
        )
        stats["saved"] += 1

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert ESMFold PDB files into GHXTox NPZ caches.")
    parser.add_argument("--fasta", default=DEFAULT_TRAIN_FASTA, help="FASTA file used for the dataset split.")
    parser.add_argument("--pdb-dir", default="data/esmfold_pdb/train", help="Directory containing ESMFold PDB files.")
    parser.add_argument("--output-dir", default=DEFAULT_STRUCTURE_CACHE_DIR, help="Output structure cache directory.")
    parser.add_argument(
        "--pdb-pattern",
        default="{sample_id}.pdb",
        help="PDB filename pattern. Available fields: {sample_id}, {original_id}, {index}.",
    )
    parser.add_argument("--chain", default=None, help="Optional chain id to extract.")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--on-missing", choices=["error", "skip"], default="error")
    parser.add_argument("--on-mismatch", choices=["error", "crop", "skip"], default="error")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = build_structure_cache(
        fasta_path=args.fasta,
        pdb_dir=args.pdb_dir,
        output_dir=args.output_dir,
        pdb_pattern=args.pdb_pattern,
        chain=args.chain,
        max_length=args.max_length,
        on_missing=args.on_missing,
        on_mismatch=args.on_mismatch,
    )
    print(
        f"Saved {stats['saved']} structure caches to {args.output_dir} "
        f"(missing={stats['missing']}, mismatch={stats['mismatch']})."
    )


if __name__ == "__main__":
    main()
