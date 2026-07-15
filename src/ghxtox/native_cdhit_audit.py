"""Prepare and compare native CD-HIT-2D homology audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from ghxtox.fasta import read_fasta


def _read_plain_sequences(path: str | Path) -> list[str]:
    sequences = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"Expected unique sequences in {path}.")
    return sequences


def _reference_sequences(paths: list[str | Path]) -> list[str]:
    sequences: list[str] = []
    seen: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sequence = row["sequence"].strip().upper()
                if sequence not in seen:
                    seen.add(sequence)
                    sequences.append(sequence)
    return sequences


def _write_fasta(rows: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample_id, sequence in rows:
            handle.write(f">{sample_id}\n{sequence}\n")


def prepare_inputs(
    reference_manifests: list[str | Path],
    test_positive: str | Path,
    test_negative: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write unique reference and exact-disjoint candidate FASTA files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = _reference_sequences(reference_manifests)
    reference_set = set(references)
    candidates: list[tuple[str, int]] = []
    for path, label in ((test_positive, 1), (test_negative, 0)):
        candidates.extend(
            (sequence, label)
            for sequence in _read_plain_sequences(path)
            if sequence not in reference_set
        )

    reference_fasta = output_dir / "reference.fasta"
    candidate_fasta = output_dir / "candidates.fasta"
    candidate_manifest = output_dir / "candidate_manifest.csv"
    _write_fasta(
        [(f"reference_{index:05d}", sequence) for index, sequence in enumerate(references, 1)],
        reference_fasta,
    )
    candidate_rows = [
        (f"toxinpred3_candidate_{index:04d}", sequence, label)
        for index, (sequence, label) in enumerate(candidates, 1)
    ]
    _write_fasta([(f"{sample_id}|{label}", sequence) for sample_id, sequence, label in candidate_rows], candidate_fasta)
    with candidate_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "sequence"])
        writer.writeheader()
        for sample_id, sequence, label in candidate_rows:
            writer.writerow({"sample_id": sample_id, "label": label, "sequence": sequence})

    return {
        "reference_unique": len(references),
        "candidate_after_exact_dedup": len(candidates),
        "candidate_positive": sum(label for _, label in candidates),
        "candidate_negative": sum(1 - label for _, label in candidates),
        "reference_fasta": str(reference_fasta),
        "candidate_fasta": str(candidate_fasta),
        "candidate_manifest": str(candidate_manifest),
    }


def prepare_length_buckets(
    reference_fasta: str | Path,
    candidate_fasta: str | Path,
    output_dir: str | Path,
    minimum_length_ratio: float = 0.8,
) -> dict[str, Any]:
    """Partition queries by length and prefilter references to an equivalent ratio range."""

    if not 0 < minimum_length_ratio <= 1:
        raise ValueError("minimum_length_ratio must be in (0, 1].")
    references = read_fasta(reference_fasta)
    candidates = read_fasta(candidate_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets: list[dict[str, Any]] = []
    for query_length in sorted({len(record.sequence) for record in candidates}):
        minimum_reference_length = math.ceil(query_length * minimum_length_ratio)
        maximum_reference_length = math.floor(query_length / minimum_length_ratio)
        bucket_references = [
            record
            for record in references
            if minimum_reference_length <= len(record.sequence) <= maximum_reference_length
        ]
        bucket_candidates = [
            record for record in candidates if len(record.sequence) == query_length
        ]
        bucket_dir = output_dir / f"length_{query_length:02d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)
        reference_path = bucket_dir / "reference.fasta"
        candidate_path = bucket_dir / "candidates.fasta"
        native_output = bucket_dir / "native_retained.fasta"
        _write_fasta([(record.header, record.sequence) for record in bucket_references], reference_path)
        _write_fasta([(record.header, record.sequence) for record in bucket_candidates], candidate_path)
        buckets.append(
            {
                "query_length": query_length,
                "minimum_reference_length": minimum_reference_length,
                "maximum_reference_length": maximum_reference_length,
                "num_references": len(bucket_references),
                "num_candidates": len(bucket_candidates),
                "reference_fasta": str(reference_path),
                "candidate_fasta": str(candidate_path),
                "native_output": str(native_output),
            }
        )
    manifest = {
        "minimum_length_ratio": minimum_length_ratio,
        "num_reference_sequences": len(references),
        "num_candidate_sequences": len(candidates),
        "buckets": buckets,
    }
    manifest_path = output_dir / "bucket_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def combine_bucket_outputs(bucket_manifest: str | Path, output_path: str | Path) -> dict[str, int]:
    """Combine retained query FASTAs from all completed length buckets."""

    manifest = json.loads(Path(bucket_manifest).read_text(encoding="utf-8"))
    retained = []
    expected_candidates = 0
    for bucket in manifest["buckets"]:
        expected_candidates += int(bucket["num_candidates"])
        native_output = Path(bucket["native_output"])
        if not native_output.exists():
            raise FileNotFoundError(f"Missing native bucket output: {native_output}")
        retained.extend(read_fasta(native_output))
    if len({record.sample_id for record in retained}) != len(retained):
        raise ValueError("Combined native retained IDs must be unique.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fasta([(record.header, record.sequence) for record in retained], output_path)
    return {"input_candidates": expected_candidates, "retained": len(retained)}


def summarize_audit(
    candidate_manifest: str | Path,
    fallback_manifest: str | Path,
    native_outputs: dict[str, str | Path],
    output_dir: str | Path,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare one or more native retained FASTAs with the fallback set."""

    with Path(candidate_manifest).open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    candidate_by_sequence = {row["sequence"]: row for row in candidates}
    if len(candidate_by_sequence) != len(candidates):
        raise ValueError("Candidate manifest sequences must be unique.")
    with Path(fallback_manifest).open(encoding="utf-8", newline="") as handle:
        fallback_rows = list(csv.DictReader(handle))
    fallback_sequences = {row["sequence"].strip().upper() for row in fallback_rows}
    if not fallback_sequences <= set(candidate_by_sequence):
        raise ValueError("Fallback manifest is not a subset of the candidate manifest.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    native_sets: dict[str, set[str]] = {}
    for name, path in native_outputs.items():
        native_sequences = {record.sequence for record in read_fasta(path)}
        if not native_sequences <= set(candidate_by_sequence):
            raise ValueError(f"Native output {name} contains unknown sequences.")
        native_sets[name] = native_sequences
        intersection = native_sequences & fallback_sequences
        variants[name] = {
            "retained": len(native_sequences),
            "retained_positive": sum(
                int(candidate_by_sequence[sequence]["label"]) for sequence in native_sequences
            ),
            "retained_negative": sum(
                1 - int(candidate_by_sequence[sequence]["label"]) for sequence in native_sequences
            ),
            "intersection_with_fallback": len(intersection),
            "native_only": len(native_sequences - fallback_sequences),
            "native_only_positive": sum(
                int(candidate_by_sequence[sequence]["label"])
                for sequence in native_sequences - fallback_sequences
            ),
            "native_only_negative": sum(
                1 - int(candidate_by_sequence[sequence]["label"])
                for sequence in native_sequences - fallback_sequences
            ),
            "fallback_only": len(fallback_sequences - native_sequences),
            "fallback_only_positive": sum(
                int(candidate_by_sequence[sequence]["label"])
                for sequence in fallback_sequences - native_sequences
            ),
            "fallback_only_negative": sum(
                1 - int(candidate_by_sequence[sequence]["label"])
                for sequence in fallback_sequences - native_sequences
            ),
            "jaccard_with_fallback": len(intersection) / len(native_sequences | fallback_sequences),
        }

    for row in candidates:
        sequence = row["sequence"]
        output_row: dict[str, Any] = {
            **row,
            "fallback_retained": int(sequence in fallback_sequences),
        }
        for name, native_sequences in native_sets.items():
            output_row[f"{name}_retained"] = int(sequence in native_sequences)
        if len({output_row[key] for key in output_row if key.endswith("_retained")}) > 1:
            comparison_rows.append(output_row)

    comparison_path = output_dir / "retention_disagreements.csv"
    fieldnames = ["sample_id", "label", "sequence", "fallback_retained"] + [
        f"{name}_retained" for name in native_outputs
    ]
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {
        "protocol": protocol or {},
        "candidate_after_exact_dedup": len(candidates),
        "fallback": {
            "retained": len(fallback_sequences),
            "retained_positive": sum(
                int(candidate_by_sequence[sequence]["label"]) for sequence in fallback_sequences
            ),
            "retained_negative": sum(
                1 - int(candidate_by_sequence[sequence]["label"]) for sequence in fallback_sequences
            ),
        },
        "native_variants": variants,
        "num_any_disagreement": len(comparison_rows),
        "outputs": {"retention_disagreements": str(comparison_path)},
    }
    summary_path = output_dir / "native_cdhit_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_named_paths(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Native outputs must use NAME=PATH syntax.")
        name, path = value.split("=", 1)
        if not name or not path or name in parsed:
            raise ValueError(f"Invalid native output: {value}")
        parsed[name] = path
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or summarize a native CD-HIT-2D audit.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-manifests", nargs="+", required=True)
    prepare.add_argument("--test-positive", required=True)
    prepare.add_argument("--test-negative", required=True)
    prepare.add_argument("--output-dir", required=True)
    buckets = subparsers.add_parser("prepare-buckets")
    buckets.add_argument("--reference-fasta", required=True)
    buckets.add_argument("--candidate-fasta", required=True)
    buckets.add_argument("--output-dir", required=True)
    buckets.add_argument("--minimum-length-ratio", type=float, default=0.8)
    combine = subparsers.add_parser("combine-buckets")
    combine.add_argument("--bucket-manifest", required=True)
    combine.add_argument("--output", required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--candidate-manifest", required=True)
    summarize.add_argument("--fallback-manifest", required=True)
    summarize.add_argument("--native-output", nargs="+", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--protocol-field", action="append", default=[])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "prepare":
        result = prepare_inputs(
            args.reference_manifests,
            args.test_positive,
            args.test_negative,
            args.output_dir,
        )
    elif args.command == "prepare-buckets":
        result = prepare_length_buckets(
            args.reference_fasta,
            args.candidate_fasta,
            args.output_dir,
            args.minimum_length_ratio,
        )
    elif args.command == "combine-buckets":
        result = combine_bucket_outputs(args.bucket_manifest, args.output)
    else:
        result = summarize_audit(
            args.candidate_manifest,
            args.fallback_manifest,
            _parse_named_paths(args.native_output),
            args.output_dir,
            protocol=_parse_named_paths(args.protocol_field),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
