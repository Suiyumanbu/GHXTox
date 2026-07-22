"""Resumable batch wrapper for the official Abdin--Kim PepFlow sampler.

This wrapper does not reimplement PepFlow. It gives every GHXTox record a
stable output directory so duplicate sequences remain distinguishable and the
resulting multi-model PDB files can be converted reproducibly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from ghxtox.fasta import read_fasta


def sample_key(sample_id: str, index: int) -> str:
    return f"{sample_id}_{index}"


def expected_output(output_dir: Path, sequence: str) -> Path:
    return output_dir / f"{sequence}.pdb"


def build_command(
    python: str,
    pepflow_root: str | Path,
    sequence: str,
    output_dir: str | Path,
    full_model: str | Path,
    num_samples: int,
    chunk_size: int,
    include_energies: bool = True,
) -> list[str]:
    command = [
        python,
        str(Path(pepflow_root) / "generate_peptide_samples.py"),
        "-s",
        sequence,
        "-o",
        str(output_dir),
        "-fm",
        str(full_model),
        "-n",
        str(int(num_samples)),
        "-c",
        str(int(chunk_size)),
    ]
    if include_energies:
        command.append("--e")
    return command


def run_batch(
    fasta: str | Path,
    pepflow_root: str | Path,
    full_model: str | Path,
    output_root: str | Path,
    num_samples: int = 8,
    chunk_size: int = 4,
    min_length: int = 3,
    max_length: int = 30,
    start_index: int = 1,
    limit: int | None = None,
    python: str = sys.executable,
    include_energies: bool = True,
    dry_run: bool = False,
    on_error: str = "error",
    sample_id_mode: str = "indexed",
) -> dict[str, Any]:
    records = read_fasta(fasta)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = list(enumerate(records, start=1))[max(start_index - 1, 0) :]
    if limit is not None:
        selected = selected[: max(int(limit), 0)]

    rows: list[dict[str, Any]] = []
    for position, record in selected:
        key = record.sample_id if sample_id_mode == "header" else sample_key(record.sample_id, position)
        sequence = record.sequence
        sample_dir = output_root / key
        pdb_path = expected_output(sample_dir, sequence)
        row: dict[str, Any] = {
            "source_index": position - 1,
            "sample_id": key,
            "original_id": record.sample_id,
            "sequence": sequence,
            "length": len(sequence),
            "output_dir": str(sample_dir),
            "pdb_path": str(pdb_path),
        }
        if not min_length <= len(sequence) <= max_length:
            row["status"] = "skipped_length"
            rows.append(row)
            continue
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            row["status"] = "cached"
            rows.append(row)
            continue
        command = build_command(
            python,
            pepflow_root,
            sequence,
            sample_dir,
            full_model,
            num_samples,
            chunk_size,
            include_energies=include_energies,
        )
        row["command"] = command
        if dry_run:
            row["status"] = "dry_run"
            rows.append(row)
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False)
        row["returncode"] = int(completed.returncode)
        row["status"] = "generated" if completed.returncode == 0 and pdb_path.exists() else "failed"
        rows.append(row)
        if row["status"] == "failed" and on_error == "error":
            break

    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    summary = {
        "method": "Abdin-Kim PepFlow all-atom conformational sampling",
        "fasta": str(Path(fasta)),
        "num_samples_per_peptide": int(num_samples),
        "chunk_size": int(chunk_size),
        "supported_length_filter": [int(min_length), int(max_length)],
        "sample_id_mode": sample_id_mode,
        "counts": counts,
        "records": rows,
    }
    with (output_root / "ghxtox_pepflow_batch.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--pepflow-root", required=True)
    parser.add_argument("--full-model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--no-energies", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--on-error", choices=["error", "continue"], default="error")
    parser.add_argument("--sample-id-mode", choices=["indexed", "header"], default="indexed")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = run_batch(
        fasta=args.fasta,
        pepflow_root=args.pepflow_root,
        full_model=args.full_model,
        output_root=args.output_root,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        min_length=args.min_length,
        max_length=args.max_length,
        start_index=args.start_index,
        limit=args.limit,
        python=args.python,
        include_energies=not args.no_energies,
        dry_run=args.dry_run,
        on_error=args.on_error,
        sample_id_mode=args.sample_id_mode,
    )
    print(json.dumps({"counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
