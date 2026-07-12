"""Prepare uniquely identified FASTA files and manifests for CD-HIT experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ghxtox.fasta import read_fasta


def prepare_cdhit_fasta(input_path: str | Path, output_path: str | Path, manifest_path: str | Path, prefix: str) -> dict[str, int]:
    records = read_fasta(input_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    positives = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as fasta_handle, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as manifest_handle:
        writer = csv.DictWriter(
            manifest_handle,
            fieldnames=["cdhit_id", "source_index", "source_id", "label", "sequence", "source_header"],
        )
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            label_text = "na" if record.label is None else str(record.label)
            cdhit_id = f"{prefix}_{index:06d}|{label_text}"
            fasta_handle.write(f">{cdhit_id}\n{record.sequence}\n")
            writer.writerow(
                {
                    "cdhit_id": cdhit_id,
                    "source_index": index - 1,
                    "source_id": record.sample_id,
                    "label": label_text,
                    "sequence": record.sequence,
                    "source_header": record.header,
                }
            )
            positives += record.label == 1
    return {"total": len(records), "positive": positives, "negative": len(records) - positives}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare unique FASTA IDs for CD-HIT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prefix", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_cdhit_fasta(args.input, args.output, args.manifest, args.prefix)
    print(
        f"Saved {summary['total']} records to {args.output} "
        f"(positive={summary['positive']}, negative={summary['negative']})."
    )


if __name__ == "__main__":
    main()
