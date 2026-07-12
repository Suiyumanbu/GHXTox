"""Train-test peptide sequence-similarity audit with global alignment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from Bio import Align

from ghxtox.fasta import FastaRecord, read_fasta


def _kmers(sequence: str, size: int) -> set[str]:
    if len(sequence) < size:
        return {sequence}
    return {sequence[index : index + size] for index in range(len(sequence) - size + 1)}


def _make_aligner() -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def alignment_identity(first: str, second: str, aligner: Align.PairwiseAligner | None = None) -> float:
    """Return alignment identities divided by the shorter sequence length."""

    if first == second:
        return 1.0
    aligner = aligner or _make_aligner()
    alignment = aligner.align(first, second)[0]
    return float(alignment.counts().identities) / max(min(len(first), len(second)), 1)


def audit_similarity(
    train_records: list[FastaRecord],
    query_records: list[FastaRecord],
    thresholds: tuple[float, ...] = (0.9, 0.8),
    kmer_size: int = 2,
) -> dict[str, Any]:
    if min(thresholds) < 0.8:
        raise ValueError("The k-mer candidate audit is intended for identity thresholds >= 0.8.")
    if kmer_size != 2:
        raise ValueError("Use kmer_size=2; larger k-mers can miss 0.8-identity short peptides.")
    index: dict[str, set[int]] = defaultdict(set)
    exact_index: dict[str, int] = {}
    for train_index, record in enumerate(train_records):
        exact_index.setdefault(record.sequence, train_index)
        for kmer in _kmers(record.sequence, kmer_size):
            index[kmer].add(train_index)

    aligner = _make_aligner()
    rows = []
    for query_index, query in enumerate(query_records):
        exact_train_index = exact_index.get(query.sequence)
        candidates: set[int] = set()
        for kmer in _kmers(query.sequence, kmer_size):
            candidates.update(index.get(kmer, ()))
        if exact_train_index is not None:
            candidates.add(exact_train_index)

        best_identity = 0.0
        best_train_index: int | None = None
        if exact_train_index is not None:
            best_identity = 1.0
            best_train_index = exact_train_index
        else:
            for train_index in candidates:
                train = train_records[train_index]
                identity = alignment_identity(query.sequence, train.sequence, aligner)
                if identity > best_identity:
                    best_identity = identity
                    best_train_index = train_index

        nearest = train_records[best_train_index] if best_train_index is not None else None
        row = {
            "query_index": query_index,
            "query_id": query.sample_id,
            "query_sequence": query.sequence,
            "query_label": query.label,
            "query_length": len(query.sequence),
            "candidate_count": len(candidates),
            "max_identity": best_identity,
            "nearest_train_index": best_train_index,
            "nearest_train_id": nearest.sample_id if nearest else None,
            "nearest_train_sequence": nearest.sequence if nearest else None,
            "nearest_train_label": nearest.label if nearest else None,
            "nearest_label_match": bool(nearest and nearest.label == query.label),
        }
        rows.append(row)

    threshold_summary = {}
    for threshold in thresholds:
        high = [row for row in rows if row["max_identity"] >= threshold]
        retained = [row for row in rows if row["max_identity"] < threshold]
        threshold_summary[str(threshold)] = {
            "high_similarity": len(high),
            "high_similarity_positive": sum(row["query_label"] == 1 for row in high),
            "high_similarity_negative": sum(row["query_label"] == 0 for row in high),
            "high_similarity_label_match": sum(row["nearest_label_match"] for row in high),
            "high_similarity_label_conflict": sum(not row["nearest_label_match"] for row in high),
            "retained": len(retained),
            "retained_positive": sum(row["query_label"] == 1 for row in retained),
            "retained_negative": sum(row["query_label"] == 0 for row in retained),
        }

    return {
        "protocol": {
            "method": "biopython_global_alignment_audit",
            "identity_definition": "identical_aligned_residues / shorter_sequence_length",
            "match_score": 2.0,
            "mismatch_score": -1.0,
            "open_gap_score": -2.0,
            "extend_gap_score": -0.5,
            "candidate_filter": f"shared_{kmer_size}mer",
            "thresholds": list(thresholds),
            "note": "Pre-audit compatible with high-identity screening; not a native CD-HIT run.",
        },
        "summary": {
            "num_train": len(train_records),
            "num_query": len(query_records),
            "full_shorter_sequence_identity": sum(row["max_identity"] == 1.0 for row in rows),
            "exact_sequence_matches": sum(
                row["query_sequence"] == row["nearest_train_sequence"] for row in rows
            ),
            "thresholds": threshold_summary,
        },
        "rows": rows,
    }


def _write_rows(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit train-test peptide sequence identity.")
    parser.add_argument("--train", required=True, help="Training FASTA.")
    parser.add_argument("--query", required=True, help="Test/query FASTA.")
    parser.add_argument("--output", required=True, help="Summary JSON output.")
    parser.add_argument("--rows", required=True, help="Per-query CSV output.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.8])
    parser.add_argument("--kmer-size", type=int, default=2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = audit_similarity(
        read_fasta(args.train),
        read_fasta(args.query),
        thresholds=tuple(args.thresholds),
        kmer_size=args.kmer_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2) + "\n", encoding="utf-8")
    _write_rows(result["rows"], args.rows)
    print(f"Similarity audit saved to {output_path} and {args.rows}")
    for threshold, summary in result["summary"]["thresholds"].items():
        print(
            f"identity>={threshold}: high={summary['high_similarity']} "
            f"retained={summary['retained']} "
            f"retained_pos={summary['retained_positive']} retained_neg={summary['retained_negative']}"
        )


if __name__ == "__main__":
    main()
