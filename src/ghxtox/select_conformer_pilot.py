"""Select a deterministic fold/label/topology-stratified PepFlow pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

from ghxtox.fasta import read_fasta
from ghxtox.peptide_topology_audit import _risk_class


def _read_folds(path: str | Path) -> dict[int, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {int(row["source_index"]): row for row in csv.DictReader(handle)}


def _read_excluded_indices(path: str | Path | None) -> set[int]:
    if path is None:
        return set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "source_index" not in rows[0]:
        raise ValueError(f"Exclusion manifest lacks source_index: {path}")
    return {int(row["source_index"]) for row in rows}


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def select_pilot(
    fasta: str | Path,
    fold_manifest: str | Path,
    output_fasta: str | Path,
    output_manifest: str | Path,
    target_size: int = 250,
    min_length: int = 3,
    max_length: int = 30,
    seed: int = 2026,
    exclude_manifest: str | Path | None = None,
) -> list[dict[str, Any]]:
    records = read_fasta(fasta)
    folds = _read_folds(fold_manifest)
    excluded_indices = _read_excluded_indices(exclude_manifest)
    strata: dict[tuple[int, int | None, str], list[dict[str, Any]]] = {}
    for source_index, record in enumerate(records):
        fold_row = folds.get(source_index)
        if fold_row is None:
            raise ValueError(f"Missing fold assignment for source_index={source_index}.")
        if source_index in excluded_indices:
            continue
        if not min_length <= len(record.sequence) <= max_length:
            continue
        cysteines = record.sequence.count("C")
        risk = _risk_class(cysteines, len(record.sequence), False, False)
        row = {
            "source_index": source_index,
            "sample_id": str(fold_row["sample_id"]),
            "label": record.label,
            "fold": int(fold_row["fold"]),
            "sequence": record.sequence,
            "length": len(record.sequence),
            "cysteines": cysteines,
            "topology_risk": risk,
        }
        strata.setdefault((row["fold"], row["label"], risk), []).append(row)

    for values in strata.values():
        values.sort(key=lambda row: _stable_rank(str(row["sample_id"]), seed))
    selected: list[dict[str, Any]] = []
    keys = sorted(
        strata,
        key=lambda key: (key[0], -1 if key[1] is None else int(key[1]), key[2]),
    )
    while len(selected) < target_size:
        progressed = False
        for key in keys:
            values = strata[key]
            if values and len(selected) < target_size:
                selected.append(values.pop(0))
                progressed = True
        if not progressed:
            break
    selected.sort(key=lambda row: int(row["source_index"]))
    if not selected:
        raise ValueError("No sequence passed the requested pilot length filters.")

    output_fasta = Path(output_fasta)
    output_manifest = Path(output_manifest)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w", encoding="utf-8") as handle:
        for row in selected:
            label = "" if row["label"] is None else f"|{int(row['label'])}"
            handle.write(f">{row['sample_id']}{label}\n{row['sequence']}\n")
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-fasta", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--target-size", type=int, default=250)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--exclude-manifest",
        help="Optional CSV whose source_index rows have already been generated.",
    )
    args = parser.parse_args()
    rows = select_pilot(
        args.fasta,
        args.fold_manifest,
        args.output_fasta,
        args.output_manifest,
        target_size=args.target_size,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
        exclude_manifest=args.exclude_manifest,
    )
    print(f"Selected {len(rows)} peptides across {len(set((r['fold'], r['label'], r['topology_risk']) for r in rows))} strata.")


if __name__ == "__main__":
    main()
