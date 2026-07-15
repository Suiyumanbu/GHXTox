"""Create reproducible, sequence-group-aware cross-validation folds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from Bio import Align
from sklearn.model_selection import StratifiedGroupKFold


def load_fold_indices(
    manifest: str | Path, validation_fold: int, expected_size: int
) -> tuple[list[int], list[int]]:
    """Load train/validation indices from a fixed fold manifest."""

    assignments: dict[int, int] = {}
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assignments[int(row["source_index"])] = int(row["fold"])
    if set(assignments) != set(range(expected_size)):
        raise ValueError(
            f"Fold manifest contains {len(assignments)} indices, expected exactly {expected_size}."
        )
    validation = [index for index, fold in assignments.items() if fold == validation_fold]
    training = [index for index, fold in assignments.items() if fold != validation_fold]
    if not training or not validation:
        raise ValueError(f"Fold {validation_fold} produced an empty train or validation subset.")
    return training, validation


def _kmers(sequence: str, size: int = 2) -> set[str]:
    if len(sequence) < size:
        return {sequence}
    return {sequence[index : index + size] for index in range(len(sequence) - size + 1)}


def _aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def _identity(first: str, second: str, aligner: Align.PairwiseAligner) -> float:
    if first == second:
        return 1.0
    alignment = aligner.align(first, second)[0]
    return float(alignment.counts().identities) / max(min(len(first), len(second)), 1)


def assign_reference_groups(
    sequences: list[str], reference_sequences: list[str], threshold: float = 0.8
) -> tuple[list[str], dict[str, float | int | str]]:
    """Assign every sequence to its closest retained representative when sufficiently similar.

    This is an explicit Biopython-alignment fallback for environments where the native
    CD-HIT ``.clstr`` mapping is unavailable. It must not be reported as a native CD-HIT run.
    """

    exact: dict[str, int] = {}
    kmer_index: dict[str, set[int]] = defaultdict(set)
    for index, sequence in enumerate(reference_sequences):
        exact.setdefault(sequence, index)
        for kmer in _kmers(sequence):
            kmer_index[kmer].add(index)

    aligner = _aligner()
    groups: list[str] = []
    matched = 0
    exact_matches = 0
    for source_index, sequence in enumerate(sequences):
        reference_index = exact.get(sequence)
        best_identity = 1.0 if reference_index is not None else 0.0
        if reference_index is not None:
            exact_matches += 1
        else:
            candidates: set[int] = set()
            for kmer in _kmers(sequence):
                candidates.update(kmer_index.get(kmer, ()))
            for candidate in candidates:
                reference = reference_sequences[candidate]
                if min(len(sequence), len(reference)) / max(len(sequence), len(reference)) < threshold:
                    continue
                identity = _identity(sequence, reference, aligner)
                if identity > best_identity:
                    best_identity = identity
                    reference_index = candidate
        if reference_index is not None and best_identity >= threshold:
            groups.append(f"reference_{reference_index:06d}")
            matched += 1
        else:
            groups.append(f"singleton_{source_index:06d}")

    return groups, {
        "method": "biopython_reference_alignment_fallback",
        "identity_definition": "identical_aligned_residues / shorter_sequence_length",
        "threshold": threshold,
        "num_reference_sequences": len(reference_sequences),
        "num_exact_reference_matches": exact_matches,
        "num_assigned_to_reference": matched,
        "num_singletons": len(sequences) - matched,
    }


def create_group_folds(
    processed: str | Path,
    reference_processed: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    n_splits: int = 5,
    threshold: float = 0.8,
    seed: int = 42,
) -> dict:
    payload = torch.load(processed, map_location="cpu", weights_only=False)
    reference_payload = torch.load(reference_processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    references = reference_payload["records"]
    sequences = [str(record["sequence"]) for record in records]
    labels = [int(record["label"]) for record in records]
    groups, protocol = assign_reference_groups(
        sequences, [str(record["sequence"]) for record in references], threshold=threshold
    )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_by_index = [-1] * len(records)
    dummy = list(range(len(records)))
    for fold, (_, validation_indices) in enumerate(splitter.split(dummy, labels, groups)):
        for index in validation_indices:
            fold_by_index[int(index)] = fold
    if any(fold < 0 for fold in fold_by_index):
        raise RuntimeError("At least one record was not assigned to a validation fold.")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_index", "sample_id", "label", "sequence", "group_id", "fold"],
        )
        writer.writeheader()
        for index, (record, group_id, fold) in enumerate(zip(records, groups, fold_by_index)):
            writer.writerow(
                {
                    "source_index": index,
                    "sample_id": record.get("sample_id", f"sample_{index}"),
                    "label": labels[index],
                    "sequence": sequences[index],
                    "group_id": group_id,
                    "fold": fold,
                }
            )

    folds = []
    for fold in range(n_splits):
        indices = [index for index, assigned in enumerate(fold_by_index) if assigned == fold]
        positives = sum(labels[index] for index in indices)
        folds.append(
            {
                "fold": fold,
                "num_samples": len(indices),
                "num_positive": positives,
                "num_negative": len(indices) - positives,
                "positive_fraction": positives / max(len(indices), 1),
                "num_groups": len({groups[index] for index in indices}),
            }
        )
    summary = {
        "processed": str(processed),
        "reference_processed": str(reference_processed),
        "seed": seed,
        "n_splits": n_splits,
        "num_samples": len(records),
        "num_positive": sum(labels),
        "num_negative": len(labels) - sum(labels),
        "num_groups": len(set(groups)),
        "group_protocol": protocol,
        "folds": folds,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create fixed stratified sequence-group folds.")
    parser.add_argument("--processed", required=True)
    parser.add_argument("--reference-processed", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = create_group_folds(
        args.processed,
        args.reference_processed,
        args.output_csv,
        args.output_json,
        n_splits=args.n_splits,
        threshold=args.threshold,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
