"""Attach PepFlow conformer-ensemble descriptors to processed GHXTox records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONFORMER_RESIDUE_FEATURE_DIM = 8
CONFORMER_GLOBAL_FEATURE_DIM = 12


def _off_diagonal_mean(matrix: np.ndarray) -> np.ndarray:
    length = int(matrix.shape[0])
    if length <= 1:
        return np.zeros(length, dtype=np.float32)
    return (matrix.sum(axis=1) - np.diag(matrix)) / float(length - 1)


def ensemble_feature_tensors(summary_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw ensemble statistics into bounded, scale-normalized tensors."""

    with np.load(summary_path) as payload:
        rmsf = payload["residue_rmsf"].astype(np.float32)
        pair_mean = payload["pair_distance_mean"].astype(np.float32)
        pair_std = payload["pair_distance_std"].astype(np.float32)
        occupancies = [
            payload[f"contact_occupancy_{cutoff}"].astype(np.float32)
            for cutoff in (6, 8, 10)
        ]
        conformer_rmsd = payload["conformer_rmsd"].astype(np.float32)
        radius_gyration = payload["radius_gyration"].astype(np.float32)
        num_conformers = int(payload["aligned_coords"].shape[0])

    residue_features = np.stack(
        [
            rmsf / 10.0,
            _off_diagonal_mean(pair_mean) / 20.0,
            pair_mean.std(axis=1) / 20.0,
            _off_diagonal_mean(pair_std) / 5.0,
            *[_off_diagonal_mean(value) for value in occupancies],
            _off_diagonal_mean(occupancies[1] * (1.0 - occupancies[1])),
        ],
        axis=1,
    )
    global_features = np.asarray(
        [
            min(num_conformers, 8) / 8.0,
            rmsf.mean() / 10.0,
            rmsf.std() / 10.0,
            rmsf.max(initial=0.0) / 10.0,
            conformer_rmsd.mean() / 10.0,
            conformer_rmsd.std() / 10.0,
            conformer_rmsd.max(initial=0.0) / 10.0,
            radius_gyration.mean() / 20.0,
            radius_gyration.std() / 20.0,
            pair_std.mean() / 5.0,
            pair_std.max(initial=0.0) / 10.0,
            np.mean(occupancies[1] * (1.0 - occupancies[1])),
        ],
        dtype=np.float32,
    )
    residue_features = np.nan_to_num(residue_features, nan=0.0, posinf=3.0, neginf=-3.0)
    global_features = np.nan_to_num(global_features, nan=0.0, posinf=3.0, neginf=-3.0)
    return torch.from_numpy(residue_features), torch.from_numpy(global_features)


def _manifest_summaries(manifest: str | Path, cache_root: str | Path) -> dict[str, Path]:
    cache_root = Path(cache_root)
    summaries: dict[str, Path] = {}
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            original_id = str(row.get("original_id") or row["sample_id"])
            indexed_id = str(row["sample_id"])
            path = cache_root / indexed_id / "ensemble_summary.npz"
            if not path.exists():
                raise FileNotFoundError(f"Missing ensemble summary for {original_id}: {path}")
            previous = summaries.setdefault(original_id, path)
            if previous != path:
                raise ValueError(f"Multiple ensemble summaries were mapped to {original_id}.")
    return summaries


def attach_conformer_features(
    processed: str | Path,
    conformer_manifest: str | Path,
    cache_root: str | Path,
    output: str | Path,
    pilot_output: str | Path | None = None,
    pilot_fold_manifest: str | Path | None = None,
    source_fold_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Attach descriptors by original sample ID and optionally create pilot-only files."""

    payload = torch.load(processed, map_location="cpu", weights_only=False)
    records = payload["records"]
    summaries = _manifest_summaries(conformer_manifest, cache_root)
    covered_indices: list[int] = []
    for source_index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        summary_path = summaries.get(sample_id)
        length = int(record["aa_ids"].shape[0])
        if summary_path is None:
            record["conformer_residue_features"] = torch.zeros(
                length, CONFORMER_RESIDUE_FEATURE_DIM, dtype=torch.float32
            )
            record["conformer_global_features"] = torch.zeros(
                CONFORMER_GLOBAL_FEATURE_DIM, dtype=torch.float32
            )
            record["conformer_available"] = False
            continue
        residue_features, global_features = ensemble_feature_tensors(summary_path)
        if residue_features.shape[0] != length:
            raise ValueError(
                f"Length mismatch for {sample_id}: processed={length}, ensemble={residue_features.shape[0]}"
            )
        record["conformer_residue_features"] = residue_features
        record["conformer_global_features"] = global_features
        record["conformer_available"] = True
        record["conformer_summary_path"] = str(summary_path)
        covered_indices.append(source_index)

    missing_ids = sorted(set(summaries) - {str(record["sample_id"]) for record in records})
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} manifest IDs are absent from processed data, e.g. {missing_ids[:3]}")
    payload.setdefault("metadata", {})["conformer_ensemble"] = {
        "source": "PepFlow",
        "manifest": str(conformer_manifest),
        "cache_root": str(cache_root),
        "covered_records": len(covered_indices),
        "residue_feature_dim": CONFORMER_RESIDUE_FEATURE_DIM,
        "global_feature_dim": CONFORMER_GLOBAL_FEATURE_DIM,
        "missing_policy": "zero features with hard availability mask",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    if pilot_output is not None:
        pilot_records = [records[index] for index in covered_indices]
        pilot_payload = {
            "records": pilot_records,
            "metadata": {
                **payload.get("metadata", {}),
                "source_indices": covered_indices,
                "subset": "PepFlow-covered training pilot",
            },
        }
        pilot_output = Path(pilot_output)
        pilot_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pilot_payload, pilot_output)

        if pilot_fold_manifest is not None:
            if source_fold_manifest is None:
                raise ValueError("source_fold_manifest is required with pilot_fold_manifest.")
            folds: dict[int, dict[str, str]] = {}
            with Path(source_fold_manifest).open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    folds[int(row["source_index"])] = row
            pilot_fold_manifest = Path(pilot_fold_manifest)
            pilot_fold_manifest.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = ["source_index", "original_source_index", "sample_id", "label", "sequence", "group_id", "fold"]
            with pilot_fold_manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for pilot_index, original_index in enumerate(covered_indices):
                    source_row = folds[original_index]
                    writer.writerow(
                        {
                            "source_index": pilot_index,
                            "original_source_index": original_index,
                            "sample_id": records[original_index]["sample_id"],
                            "label": records[original_index]["label"],
                            "sequence": records[original_index]["sequence"],
                            "group_id": source_row.get("group_id", ""),
                            "fold": source_row["fold"],
                        }
                    )

    return {
        "processed": str(processed),
        "output": str(output),
        "total_records": len(records),
        "covered_records": len(covered_indices),
        "coverage_fraction": len(covered_indices) / max(len(records), 1),
        "pilot_output": str(pilot_output) if pilot_output is not None else None,
        "pilot_fold_manifest": str(pilot_fold_manifest) if pilot_fold_manifest is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--conformer-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output")
    parser.add_argument("--pilot-fold-manifest")
    parser.add_argument("--source-fold-manifest")
    args = parser.parse_args()
    summary = attach_conformer_features(
        args.processed,
        args.conformer_manifest,
        args.cache_root,
        args.output,
        pilot_output=args.pilot_output,
        pilot_fold_manifest=args.pilot_fold_manifest,
        source_fold_manifest=args.source_fold_manifest,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
