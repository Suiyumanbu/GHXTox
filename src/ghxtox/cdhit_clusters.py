"""Parse CD-HIT clusters and extract strict train-disjoint test subsets."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from ghxtox.fasta import FastaRecord, read_fasta


MEMBER_PATTERN = re.compile(r">([^\.]+)\.\.\.")


def parse_clusters(path: str | Path) -> list[list[str]]:
    clusters: list[list[str]] = []
    current: list[str] | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">Cluster "):
                current = []
                clusters.append(current)
                continue
            match = MEMBER_PATTERN.search(line)
            if match and current is not None:
                current.append(match.group(1))
    if not clusters or any(not cluster for cluster in clusters):
        raise ValueError(f"Invalid or empty CD-HIT cluster file: {path}")
    return clusters


def strict_test_ids(
    clusters: list[list[str]],
    train_prefix: str = "train_",
    test_prefix: str = "test1_",
) -> tuple[set[str], dict[str, int]]:
    retained: set[str] = set()
    mixed_clusters = 0
    excluded = 0
    for cluster in clusters:
        has_train = any(member.startswith(train_prefix) for member in cluster)
        test_members = [member for member in cluster if member.startswith(test_prefix)]
        if has_train and test_members:
            mixed_clusters += 1
            excluded += len(test_members)
        elif not has_train:
            retained.update(test_members)
    return retained, {"mixed_clusters": mixed_clusters, "excluded_test": excluded}


def extract_fasta_ids(input_path: str | Path) -> list[str]:
    return [record.header.split()[0] for record in read_fasta(input_path)]


def _write_fasta(records: list[FastaRecord], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f">{record.header}\n{record.sequence}\n")


def extract_strict_subset(
    clusters_path: str | Path,
    test_fasta_path: str | Path,
    output_path: str | Path,
    test_prefix: str,
) -> dict[str, Any]:
    clusters = parse_clusters(clusters_path)
    retained_ids, diagnostics = strict_test_ids(clusters, test_prefix=test_prefix)
    test_records = read_fasta(test_fasta_path)
    input_ids = [record.header.split()[0] for record in test_records]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Strict extraction requires unique test FASTA IDs.")
    unknown = retained_ids - set(input_ids)
    if unknown:
        raise ValueError(f"Cluster file contains unknown test IDs, e.g. {sorted(unknown)[:3]}")
    retained_records = [
        record for record, record_id in zip(test_records, input_ids) if record_id in retained_ids
    ]
    _write_fasta(retained_records, output_path)
    positive = sum(record.label == 1 for record in retained_records)
    return {
        "num_input": len(test_records),
        "num_retained": len(retained_records),
        "num_excluded": len(test_records) - len(retained_records),
        "positive": positive,
        "negative": len(retained_records) - positive,
        **diagnostics,
    }


def cluster_label_conflicts(clusters_path: str | Path, manifest_path: str | Path) -> dict[str, int]:
    clusters = parse_clusters(clusters_path)
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        labels = {row["cdhit_id"]: row["label"] for row in csv.DictReader(handle)}
    conflict_clusters = 0
    conflict_members = 0
    for cluster in clusters:
        cluster_labels = {labels[member] for member in cluster if member in labels and labels[member] != "na"}
        if len(cluster_labels) > 1:
            conflict_clusters += 1
            conflict_members += len(cluster)
    return {"conflict_clusters": conflict_clusters, "conflict_members": conflict_members}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract strict test subset from combined CD-HIT clusters.")
    parser.add_argument("--clusters", required=True)
    parser.add_argument("--test-fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-prefix", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = extract_strict_subset(args.clusters, args.test_fasta, args.output, args.test_prefix)
    print(
        f"Saved {summary['num_retained']} / {summary['num_input']} records to {args.output} "
        f"(positive={summary['positive']}, negative={summary['negative']}, "
        f"excluded={summary['num_excluded']})."
    )


if __name__ == "__main__":
    main()
