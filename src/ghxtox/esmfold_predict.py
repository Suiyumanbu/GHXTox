"""Batch-generate ESMFold structures and GHXTox NPZ caches from FASTA files."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from ghxtox.esmfold_cache import parse_esmfold_pdb, sample_key
from ghxtox.fasta import FastaRecord, read_fasta
from ghxtox.features import clean_sequence
from ghxtox.utils import DEFAULT_DEVICE, DEFAULT_STRUCTURE_CACHE_DIR, DEFAULT_TRAIN_FASTA, resolve_device


def _load_esmfold_model(device: torch.device, chunk_size: int | None) -> Any:
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError(
            "ESMFold generation requires the optional 'fair-esm' package. "
            "Install it in pytorch_env with: pip install 'fair-esm[esmfold]'"
        ) from exc

    try:
        model = esm.pretrained.esmfold_v1()
    except ModuleNotFoundError as exc:
        missing = exc.name or "an ESMFold runtime dependency"
        raise RuntimeError(
            f"ESMFold runtime dependency is missing: {missing}. "
            "On Windows, OpenFold/DeepSpeed installation can fail; the most reliable path is "
            "to run ESMFold generation in a Linux/WSL environment, then use the generated PDB/NPZ "
            "caches with GHXTox on Windows."
        ) from exc
    model = model.eval().to(device)
    if chunk_size is not None and hasattr(model, "set_chunk_size"):
        model.set_chunk_size(chunk_size)
    return model


def _write_npz_from_pdb(
    pdb_path: Path,
    cache_path: Path,
    sequence: str,
    source_id: str,
    on_mismatch: str,
) -> None:
    parsed = parse_esmfold_pdb(pdb_path)
    if parsed.coords.shape[0] != len(sequence):
        message = (
            f"Length mismatch for {source_id}: fasta={len(sequence)}, "
            f"pdb_ca={parsed.coords.shape[0]}, pdb={pdb_path}"
        )
        if on_mismatch == "crop" and parsed.coords.shape[0] >= len(sequence):
            coords = parsed.coords[: len(sequence)]
            plddt = parsed.plddt[: len(sequence)]
            backbone_coords = parsed.backbone_coords[: len(sequence)]
            backbone_mask = parsed.backbone_mask[: len(sequence)]
            functional_group_coords = parsed.functional_group_coords[: len(sequence)]
            functional_group_mask = parsed.functional_group_mask[: len(sequence)]
        else:
            raise ValueError(message)
    else:
        coords = parsed.coords
        plddt = parsed.plddt
        backbone_coords = parsed.backbone_coords
        backbone_mask = parsed.backbone_mask
        functional_group_coords = parsed.functional_group_coords
        functional_group_mask = parsed.functional_group_mask

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        coords=coords.astype(np.float32),
        plddt=plddt.astype(np.float32),
        backbone_coords=backbone_coords.astype(np.float32),
        backbone_mask=backbone_mask.astype(np.bool_),
        functional_group_coords=functional_group_coords.astype(np.float32),
        functional_group_mask=functional_group_mask.astype(np.bool_),
        sequence=np.asarray(sequence),
        source_pdb=np.asarray(str(pdb_path)),
    )


def _predict_one(
    model: Any,
    record: FastaRecord,
    index: int,
    pdb_dir: Path,
    cache_dir: Path,
    max_length: int | None,
    save_pdb: bool,
    save_npz: bool,
    skip_existing: bool,
    on_mismatch: str,
) -> str:
    key = sample_key(record, index)
    sequence = clean_sequence(record.sequence)
    if max_length is not None and len(sequence) > max_length:
        sequence = sequence[:max_length]

    pdb_path = pdb_dir / f"{key}.pdb"
    cache_path = cache_dir / f"{key}.npz"
    if skip_existing and (not save_pdb or pdb_path.exists()) and (not save_npz or cache_path.exists()):
        return "skipped"

    with torch.no_grad():
        pdb_text = model.infer_pdb(sequence)

    pdb_dir.mkdir(parents=True, exist_ok=True)
    if save_pdb:
        pdb_path.write_text(pdb_text, encoding="utf-8")

    if save_npz:
        if save_pdb:
            source_pdb = pdb_path
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, encoding="utf-8") as handle:
                handle.write(pdb_text)
                source_pdb = Path(handle.name)
        try:
            _write_npz_from_pdb(
                pdb_path=source_pdb,
                cache_path=cache_path,
                sequence=sequence,
                source_id=key,
                on_mismatch=on_mismatch,
            )
        finally:
            if not save_pdb:
                source_pdb.unlink(missing_ok=True)
    return "saved"


def generate_esmfold_structures(args: argparse.Namespace) -> dict[str, int]:
    if args.no_pdb and args.no_npz:
        raise ValueError("At least one output must be enabled; do not set both --no-pdb and --no-npz.")

    device = resolve_device(args.device)
    records = read_fasta(args.fasta)
    pdb_dir = Path(args.pdb_dir)
    cache_dir = Path(args.cache_dir)
    model = _load_esmfold_model(device, args.chunk_size)

    stats = {"saved": 0, "skipped": 0, "failed": 0}
    for index, record in enumerate(records, start=1):
        key = sample_key(record, index)
        try:
            status = _predict_one(
                model=model,
                record=record,
                index=index,
                pdb_dir=pdb_dir,
                cache_dir=cache_dir,
                max_length=args.max_length,
                save_pdb=not args.no_pdb,
                save_npz=not args.no_npz,
                skip_existing=args.skip_existing,
                on_mismatch=args.on_mismatch,
            )
            stats[status] += 1
            if index == 1 or index % max(args.log_every, 1) == 0:
                print(f"[{index}/{len(records)}] {key}: {status}")
        except Exception as exc:
            stats["failed"] += 1
            message = f"[{index}/{len(records)}] {key}: failed: {exc}"
            if args.on_error == "skip":
                print(message)
                continue
            raise RuntimeError(message) from exc

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ESMFold PDB and GHXTox NPZ caches from FASTA.")
    parser.add_argument("--fasta", default=DEFAULT_TRAIN_FASTA, help="Input FASTA path.")
    parser.add_argument("--pdb-dir", default="data/esmfold_pdb/train", help="Directory for generated PDB files.")
    parser.add_argument("--cache-dir", default=DEFAULT_STRUCTURE_CACHE_DIR, help="Directory for generated NPZ caches.")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64, help="ESMFold axial attention chunk size.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose requested outputs exist.")
    parser.add_argument("--no-pdb", action="store_true", help="Do not keep generated PDB files.")
    parser.add_argument("--no-npz", action="store_true", help="Do not write GHXTox NPZ structure caches.")
    parser.add_argument("--on-error", choices=["error", "skip"], default="error")
    parser.add_argument("--on-mismatch", choices=["error", "crop"], default="error")
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        stats = generate_esmfold_structures(args)
    except Exception as exc:
        raise SystemExit(f"ESMFold generation failed: {exc}") from None
    print(
        f"ESMFold generation complete: saved={stats['saved']}, "
        f"skipped={stats['skipped']}, failed={stats['failed']}."
    )


if __name__ == "__main__":
    main()
