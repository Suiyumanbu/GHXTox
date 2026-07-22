"""Audit sequence-visible topology risks before peptide 3D modelling.

FASTA alone cannot prove head-to-tail cyclization or disulfide connectivity.
This module therefore separates explicit header annotations from conservative
sequence heuristics instead of silently treating every cysteine-rich peptide
as cyclic.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re
from typing import Any

from ghxtox.fasta import read_fasta


CYCLIC_PATTERN = re.compile(
    r"\b(cyclic|cyclized|cyclised|cyclo[-_ ]?peptide|head[-_ ]?to[-_ ]?tail|macrocycl\w*)\b",
    re.IGNORECASE,
)
DISULFIDE_PATTERN = re.compile(
    r"\b(disulfide|disulphide|s[-_ ]?s|cystine|knottin|cyclotide|conotoxin)\b",
    re.IGNORECASE,
)


def _risk_class(cysteines: int, length: int, explicit_cyclic: bool, explicit_disulfide: bool) -> str:
    if explicit_cyclic:
        return "explicit_cyclic"
    if explicit_disulfide:
        return "explicit_disulfide"
    fraction = cysteines / max(length, 1)
    if cysteines >= 6 or (cysteines >= 4 and fraction >= 0.15):
        return "high_cysteine_topology_risk"
    if cysteines >= 2:
        return "possible_disulfide"
    return "no_sequence_visible_topology_flag"


def audit_fasta(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(read_fasta(path), start=1):
        sequence = record.sequence.upper()
        cysteines = sequence.count("C")
        explicit_cyclic = bool(CYCLIC_PATTERN.search(record.header))
        explicit_disulfide = bool(DISULFIDE_PATTERN.search(record.header))
        rows.append(
            {
                "split": split or Path(path).stem,
                "source_index": index - 1,
                "sample_id": record.sample_id,
                "label": record.label,
                "length": len(sequence),
                "cysteines": cysteines,
                "cysteine_fraction": cysteines / max(len(sequence), 1),
                "even_cysteine_count": int(cysteines >= 2 and cysteines % 2 == 0),
                "explicit_cyclic_annotation": int(explicit_cyclic),
                "explicit_disulfide_annotation": int(explicit_disulfide),
                "topology_risk": _risk_class(
                    cysteines,
                    len(sequence),
                    explicit_cyclic,
                    explicit_disulfide,
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Any] = {}
    for split in sorted({str(row["split"]) for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        labels = Counter(row["label"] for row in selected)
        risks = Counter(str(row["topology_risk"]) for row in selected)
        risk_labels: dict[str, dict[str, float | int]] = {}
        for risk in sorted(risks):
            risk_rows = [row for row in selected if row["topology_risk"] == risk]
            labeled = [row for row in risk_rows if row["label"] in {0, 1}]
            positives = sum(row["label"] == 1 for row in labeled)
            risk_labels[risk] = {
                "samples": len(risk_rows),
                "positive": positives,
                "negative": len(labeled) - positives,
                "positive_fraction": positives / max(len(labeled), 1),
            }
        topology_flagged = sum(
            risk != "no_sequence_visible_topology_flag"
            for risk in (row["topology_risk"] for row in selected)
        )
        high_risk = risks["explicit_cyclic"] + risks["explicit_disulfide"] + risks[
            "high_cysteine_topology_risk"
        ]
        by_split[split] = {
            "samples": len(selected),
            "positive": labels.get(1, 0),
            "negative": labels.get(0, 0),
            "unlabeled": labels.get(None, 0),
            "cysteine_ge_2": sum(int(row["cysteines"]) >= 2 for row in selected),
            "cysteine_ge_4": sum(int(row["cysteines"]) >= 4 for row in selected),
            "length_3_to_30": sum(3 <= int(row["length"]) <= 30 for row in selected),
            "length_3_to_30_fraction": sum(3 <= int(row["length"]) <= 30 for row in selected)
            / max(len(selected), 1),
            "topology_flagged": topology_flagged,
            "topology_flagged_fraction": topology_flagged / max(len(selected), 1),
            "high_risk": high_risk,
            "high_risk_fraction": high_risk / max(len(selected), 1),
            "risk_counts": dict(sorted(risks.items())),
            "risk_label_distribution": risk_labels,
        }
    return {
        "interpretation": (
            "Sequence heuristics identify samples that should not automatically be modelled as "
            "unconstrained linear peptides. They do not establish cyclization or disulfide connectivity."
        ),
        "splits": by_split,
    }


def run_audit(inputs: list[str], split_names: list[str] | None, output_dir: str | Path) -> dict[str, Any]:
    if split_names is not None and len(split_names) != len(inputs):
        raise ValueError("--split-names must contain one name for every --inputs path.")
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(inputs):
        split = split_names[index] if split_names is not None else Path(path).stem
        rows.extend(audit_fasta(path, split=split))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sample_topology_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["inputs"] = [str(Path(path)) for path in inputs]
    summary["sample_csv"] = str(csv_path)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--split-names", nargs="+")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_audit(args.inputs, args.split_names, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
