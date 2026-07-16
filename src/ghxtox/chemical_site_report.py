"""Summarize paired chemical-site experiments from saved validation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ghxtox.utils import save_json


METRICS = ("balanced_accuracy", "f1", "mcc", "auroc", "auprc")


def _checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _shared_state_audit(
    control: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    control_state = control["model_state"]
    candidate_state = candidate["model_state"]
    shared_names = sorted(set(control_state) & set(candidate_state))
    compared_names = [
        name for name in shared_names if not name.startswith("chemical_site_branch.")
    ]
    max_abs_diff = 0.0
    changed_names: list[str] = []
    for name in compared_names:
        left = control_state[name]
        right = candidate_state[name]
        if left.shape != right.shape:
            changed_names.append(name)
            continue
        difference = float((left - right).abs().max().item()) if left.numel() else 0.0
        max_abs_diff = max(max_abs_diff, difference)
        if difference != 0.0:
            changed_names.append(name)
    return {
        "shared_nonchemical_tensors": len(compared_names),
        "changed_nonchemical_tensors": len(changed_names),
        "max_abs_difference": max_abs_diff,
        "changed_tensor_names": changed_names[:20],
    }


def summarize(
    control_pattern: str,
    candidate_pattern: str,
    folds: list[int],
) -> dict[str, Any]:
    fold_results = []
    audits = []
    for fold in folds:
        control_path = control_pattern.format(fold=fold)
        candidate_path = candidate_pattern.format(fold=fold)
        control = _checkpoint(control_path)
        candidate = _checkpoint(candidate_path)
        control_metrics = {
            metric: float(control["val_metrics"][metric]) for metric in METRICS
        }
        candidate_metrics = {
            metric: float(candidate["val_metrics"][metric]) for metric in METRICS
        }
        fold_results.append(
            {
                "fold": fold,
                "control_checkpoint": control_path,
                "candidate_checkpoint": candidate_path,
                "control_epoch": int(control.get("epoch", -1)),
                "candidate_epoch": int(candidate.get("epoch", -1)),
                "control": control_metrics,
                "candidate": candidate_metrics,
                "candidate_minus_control": {
                    metric: candidate_metrics[metric] - control_metrics[metric]
                    for metric in METRICS
                },
            }
        )
        audit = _shared_state_audit(control, candidate)
        audit["fold"] = fold
        audits.append(audit)

    control_mean = {
        metric: sum(row["control"][metric] for row in fold_results) / len(fold_results)
        for metric in METRICS
    }
    candidate_mean = {
        metric: sum(row["candidate"][metric] for row in fold_results) / len(fold_results)
        for metric in METRICS
    }
    delta_mean = {
        metric: candidate_mean[metric] - control_mean[metric] for metric in METRICS
    }
    positive_folds = {
        metric: sum(
            row["candidate_minus_control"][metric] > 0.0 for row in fold_results
        )
        for metric in METRICS
    }
    return {
        "protocol": {
            "folds": folds,
            "control_pattern": control_pattern,
            "candidate_pattern": candidate_pattern,
            "metrics": list(METRICS),
            "external_sets_used_for_selection": False,
        },
        "fold_results": fold_results,
        "unweighted_fold_mean": {
            "control": control_mean,
            "candidate": candidate_mean,
            "candidate_minus_control": delta_mean,
        },
        "positive_delta_folds": positive_folds,
        "frozen_base_audit": {
            "all_nonchemical_tensors_unchanged": all(
                audit["changed_nonchemical_tensors"] == 0 for audit in audits
            ),
            "folds": audits,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-pattern",
        default="runs/chemical_site_final_control_fold{fold}/best_model.pt",
    )
    parser.add_argument("--candidate-pattern", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = summarize(args.control_pattern, args.candidate_pattern, args.folds)
    save_json(summary, args.output)
    delta = summary["unweighted_fold_mean"]["candidate_minus_control"]
    print(
        f"Saved {args.output}: "
        f"delta_mcc={delta['mcc']:+.6f}, delta_auprc={delta['auprc']:+.6f}"
    )


if __name__ == "__main__":
    main()
