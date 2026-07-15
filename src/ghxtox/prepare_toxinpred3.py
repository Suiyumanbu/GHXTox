"""Prepare a homology-screened ToxinPred3 independent external set."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from typing import Any

from ghxtox.folds import assign_reference_groups


CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
SOURCE_URL = "https://github.com/raghavagps/toxinpred3/tree/main/dataset"


def _read_sequences(path: str | Path) -> list[str]:
    sequences = [
        line.strip().upper()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    invalid = [sequence for sequence in sequences if set(sequence) - CANONICAL_AA]
    if invalid:
        raise ValueError(f"{path} contains {len(invalid)} non-canonical sequences.")
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"{path} contains duplicate sequences.")
    return sequences


def _read_reference_manifests(paths: list[str | Path]) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = collections.defaultdict(set)
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                labels[row["sequence"].strip().upper()].add(int(row["label"]))
    return labels


def prepare_external_set(
    train_positive: str | Path,
    train_negative: str | Path,
    test_positive: str | Path,
    test_negative: str | Path,
    reference_manifests: list[str | Path],
    output_dir: str | Path,
    identity_threshold: float = 0.8,
    minimum_per_class: int = 100,
) -> dict[str, Any]:
    sources = {
        "train_positive": (_read_sequences(train_positive), 1),
        "train_negative": (_read_sequences(train_negative), 0),
        "test_positive": (_read_sequences(test_positive), 1),
        "test_negative": (_read_sequences(test_negative), 0),
    }
    candidate_labels: dict[str, set[int]] = collections.defaultdict(set)
    for sequences, label in sources.values():
        for sequence in sequences:
            candidate_labels[sequence].add(label)
    conflicts = [sequence for sequence, labels in candidate_labels.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f"ToxinPred3 files contain {len(conflicts)} cross-label conflicts.")

    reference_labels = _read_reference_manifests(reference_manifests)
    split_audit: dict[str, dict[str, int | float]] = {}
    for name, (sequences, label) in sources.items():
        exact = [sequence for sequence in sequences if sequence in reference_labels]
        split_audit[name] = {
            "rows": len(sequences),
            "unique": len(set(sequences)),
            "exact_overlap_current": len(exact),
            "exact_overlap_fraction": len(exact) / max(len(sequences), 1),
            "label_conflicts": sum(label not in reference_labels[sequence] for sequence in exact),
            "novel_unique": len(set(sequences) - set(reference_labels)),
        }

    candidates: list[tuple[str, int]] = []
    for name in ("test_positive", "test_negative"):
        sequences, label = sources[name]
        candidates.extend(
            (sequence, label) for sequence in sequences if sequence not in reference_labels
        )
    groups, protocol = assign_reference_groups(
        [sequence for sequence, _ in candidates],
        list(reference_labels),
        threshold=identity_threshold,
    )
    retained = [
        (sequence, label)
        for (sequence, label), group in zip(candidates, groups)
        if group.startswith("singleton_")
    ]
    retained_positive = sum(label for _, label in retained)
    retained_negative = len(retained) - retained_positive
    accepted = retained_positive >= minimum_per_class and retained_negative >= minimum_per_class

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "strict_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "label", "sequence", "source_split"]
        )
        writer.writeheader()
        for index, (sequence, label) in enumerate(retained, start=1):
            writer.writerow(
                {
                    "sample_id": f"toxinpred3_strict_{index:04d}",
                    "label": label,
                    "sequence": sequence,
                    "source_split": "independent_test",
                }
            )

    fasta_path = output_dir / "strict.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (sequence, label) in enumerate(retained, start=1):
            handle.write(f">toxinpred3_strict_{index:04d}|{label}\n{sequence}\n")

    summary = {
        "source": {
            "name": "ToxinPred3.0 official dataset",
            "url": SOURCE_URL,
            "role": "independent_test_only",
            "note": "Training files are audited but excluded from GHXTox training.",
        },
        "reference_manifests": [str(path) for path in reference_manifests],
        "split_audit": split_audit,
        "strict_external_set": {
            "input_after_exact_dedup": len(candidates),
            "retained": len(retained),
            "retained_positive": retained_positive,
            "retained_negative": retained_negative,
            "acceptance_rule": f"at least {minimum_per_class} samples per class",
            "accepted": accepted,
            "homology_protocol": protocol,
        },
        "outputs": {"manifest": str(manifest_path), "fasta": str(fasta_path)},
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a strict ToxinPred3 independent external evaluation set."
    )
    parser.add_argument("--train-positive", required=True)
    parser.add_argument("--train-negative", required=True)
    parser.add_argument("--test-positive", required=True)
    parser.add_argument("--test-negative", required=True)
    parser.add_argument("--reference-manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.8)
    parser.add_argument("--minimum-per-class", type=int, default=100)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = prepare_external_set(
        args.train_positive,
        args.train_negative,
        args.test_positive,
        args.test_negative,
        args.reference_manifests,
        args.output_dir,
        identity_threshold=args.identity_threshold,
        minimum_per_class=args.minimum_per_class,
    )
    print(json.dumps(summary["strict_external_set"], indent=2))


if __name__ == "__main__":
    main()
