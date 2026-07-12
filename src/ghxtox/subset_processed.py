"""Subset an existing processed tensor payload using unique CD-HIT FASTA IDs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from ghxtox.fasta import read_fasta


def subset_processed(
    processed_path: str | Path,
    manifest_path: str | Path,
    retained_fasta_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = torch.load(processed_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {row["cdhit_id"]: row for row in csv.DictReader(handle)}
    retained_ids = [record.header.split()[0] for record in read_fasta(retained_fasta_path)]
    selected = []
    for retained_id in retained_ids:
        if retained_id not in manifest:
            raise ValueError(f"Retained ID {retained_id!r} is missing from manifest.")
        row = manifest[retained_id]
        index = int(row["source_index"])
        record = records[index]
        if record["sequence"] != row["sequence"]:
            raise ValueError(f"Processed/manifest sequence mismatch at source index {index}.")
        if record.get("label") is not None and int(record["label"]) != int(row["label"]):
            raise ValueError(f"Processed/manifest label mismatch at source index {index}.")
        selected.append(record)
    output = dict(payload)
    output["records"] = selected
    output["cdhit_subset"] = {
        "processed_source": str(processed_path),
        "manifest": str(manifest_path),
        "retained_fasta": str(retained_fasta_path),
        "num_input": len(records),
        "num_output": len(selected),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    positive = sum(int(record["label"]) == 1 for record in selected)
    return {"total": len(selected), "positive": positive, "negative": len(selected) - positive}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Subset processed tensors from a CD-HIT retained FASTA.")
    parser.add_argument("--processed", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--retained-fasta", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = subset_processed(args.processed, args.manifest, args.retained_fasta, args.output)
    print(
        f"Saved {summary['total']} records to {args.output} "
        f"(positive={summary['positive']}, negative={summary['negative']})."
    )


if __name__ == "__main__":
    main()
