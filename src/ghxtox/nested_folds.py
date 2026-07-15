"""Build and validate nested group-aware train/validation/calibration/test splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


ROLES = ("train", "validation", "calibration", "test")


def _read_base_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["source_index"]))
    if [int(row["source_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Base fold manifest source indices must be contiguous and unique.")
    return rows


def load_nested_indices(
    manifest: str | Path,
    outer_fold: int,
    expected_size: int,
) -> dict[str, list[int]]:
    indices = {role: [] for role in ROLES}
    seen: set[int] = set()
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["outer_fold"]) != outer_fold:
                continue
            index = int(row["source_index"])
            role = row["role"]
            if role not in indices:
                raise ValueError(f"Unknown nested split role {role!r}.")
            if index in seen:
                raise ValueError(f"Source index {index} is duplicated in outer fold {outer_fold}.")
            seen.add(index)
            indices[role].append(index)
    if seen != set(range(expected_size)):
        raise ValueError(
            f"Outer fold {outer_fold} covers {len(seen)}/{expected_size} unique source indices."
        )
    if any(not indices[role] for role in ROLES):
        raise ValueError(f"Outer fold {outer_fold} contains an empty role.")
    return indices


def _role_summary(rows: list[dict[str, str]], role_by_index: dict[int, str]) -> dict[str, dict]:
    result = {}
    for role in ROLES:
        selected = [row for row in rows if role_by_index[int(row["source_index"])] == role]
        positives = sum(int(row["label"]) for row in selected)
        result[role] = {
            "num_samples": len(selected),
            "num_positive": positives,
            "num_negative": len(selected) - positives,
            "positive_fraction": positives / max(len(selected), 1),
            "num_groups": len({row["group_id"] for row in selected}),
        }
    return result


def _validate_roles(rows: list[dict[str, str]], role_by_index: dict[int, str], outer_fold: int) -> None:
    if set(role_by_index) != set(range(len(rows))):
        raise RuntimeError(f"Outer fold {outer_fold} does not assign every source index exactly once.")
    role_groups = {
        role: {
            row["group_id"]
            for row in rows
            if role_by_index[int(row["source_index"])] == role
        }
        for role in ROLES
    }
    for first_index, first in enumerate(ROLES):
        for second in ROLES[first_index + 1 :]:
            overlap = role_groups[first] & role_groups[second]
            if overlap:
                raise RuntimeError(
                    f"Outer fold {outer_fold} leaks {len(overlap)} groups between {first} and {second}."
                )
    for role in ROLES:
        labels = {
            int(row["label"])
            for row in rows
            if role_by_index[int(row["source_index"])] == role
        }
        if labels != {0, 1}:
            raise RuntimeError(f"Outer fold {outer_fold} role {role} does not contain both classes.")


def create_nested_group_folds(
    base_manifest: str | Path,
    output_csv: str | Path,
    output_json: str | Path,
    inner_splits: int = 8,
    seed: int = 2026,
) -> dict:
    if inner_splits < 4:
        raise ValueError("inner_splits must be at least 4 so train remains the majority role.")
    rows = _read_base_manifest(base_manifest)
    outer_folds = sorted({int(row["fold"]) for row in rows})
    output_rows: list[dict] = []
    summaries = []
    test_counts: Counter[int] = Counter()

    for outer_fold in outer_folds:
        outer_test = [index for index, row in enumerate(rows) if int(row["fold"]) == outer_fold]
        outer_train = [index for index, row in enumerate(rows) if int(row["fold"]) != outer_fold]
        labels = [int(rows[index]["label"]) for index in outer_train]
        groups = [rows[index]["group_id"] for index in outer_train]
        splitter = StratifiedGroupKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=seed + outer_fold,
        )
        inner_fold_by_source: dict[int, int] = {}
        dummy = list(range(len(outer_train)))
        for inner_fold, (_, held_positions) in enumerate(splitter.split(dummy, labels, groups)):
            for position in held_positions:
                inner_fold_by_source[outer_train[int(position)]] = inner_fold
        if set(inner_fold_by_source) != set(outer_train):
            raise RuntimeError(f"Inner split assignment is incomplete for outer fold {outer_fold}.")

        role_by_index: dict[int, str] = {}
        for index in outer_test:
            role_by_index[index] = "test"
            test_counts[index] += 1
        for index in outer_train:
            inner_fold = inner_fold_by_source[index]
            role_by_index[index] = (
                "validation" if inner_fold == 0 else "calibration" if inner_fold == 1 else "train"
            )
        _validate_roles(rows, role_by_index, outer_fold)
        role_summary = _role_summary(rows, role_by_index)
        summaries.append({"outer_fold": outer_fold, "roles": role_summary})

        for index, row in enumerate(rows):
            output_rows.append(
                {
                    "outer_fold": outer_fold,
                    "source_index": index,
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "sequence": row["sequence"],
                    "group_id": row["group_id"],
                    "base_fold": row["fold"],
                    "inner_fold": inner_fold_by_source.get(index, -1),
                    "role": role_by_index[index],
                }
            )

    if test_counts != Counter({index: 1 for index in range(len(rows))}):
        raise RuntimeError("Every sample must appear as outer test exactly once.")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "base_manifest": str(base_manifest),
        "seed": seed,
        "num_samples": len(rows),
        "num_groups": len({row["group_id"] for row in rows}),
        "outer_folds": len(outer_folds),
        "inner_splits": inner_splits,
        "role_definition": {
            "test": "the original fixed outer fold",
            "validation": "inner fold 0, used for checkpoint/model selection",
            "calibration": "inner fold 1, never used for gradient updates or checkpoint selection",
            "train": f"inner folds 2..{inner_splits - 1}",
        },
        "folds": summaries,
        "validation": {
            "every_sample_test_exactly_once": True,
            "groups_disjoint_within_every_outer_fold": True,
            "all_roles_contain_both_classes": True,
        },
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create nested group-aware CV and calibration splits.")
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--inner-splits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = create_nested_group_folds(
        args.base_manifest,
        args.output_csv,
        args.output_json,
        inner_splits=args.inner_splits,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
