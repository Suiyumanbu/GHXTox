"""Run a fixed chemical-site component ablation on a group-aware fold."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from ghxtox.train import build_arg_parser as build_train_arg_parser
from ghxtox.train import train
from ghxtox.utils import load_json


VARIANT_OVERRIDES: dict[str, dict[str, object]] = {
    "full": {},
    "single_site": {"chemical_site_max_site_slots": 1},
    "no_hydrophobic": {"chemical_site_use_hydrophobic_sites": False},
    "no_orientation": {"chemical_site_use_orientation": False},
    "raw_only": {"chemical_site_use_normalized_rbf": False},
    "normalized_only": {"chemical_site_use_raw_rbf": False},
    "no_interaction_types": {"chemical_site_use_interaction_types": False},
    "no_plddt": {"chemical_site_use_plddt": False},
    "random_init": {"chemical_site_zero_init_residual": False},
    "include_same_residue": {"chemical_site_exclude_same_residue_edges": False},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANT_OVERRIDES), required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--base-config",
        default="configs/chemical_site_interaction_pareto.json",
    )
    parser.add_argument(
        "--train",
        default="data/processed/train_chemical_sites_final_esm2.pt",
    )
    parser.add_argument(
        "--fold-manifest",
        default="data/folds/train_cdhit080_fallback_folds.csv",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = copy.deepcopy(load_json(args.base_config))
    config["model"].update(VARIANT_OVERRIDES[args.variant])
    config["experiment"] = {
        "family": "chemical_site_component_ablation",
        "variant": args.variant,
        "base_config": str(args.base_config),
        "fold": args.fold,
    }
    output_dir = args.output_dir or f"runs/chemical_site_ablation_{args.variant}_fold{args.fold}"
    train_args = build_train_arg_parser().parse_args(
        [
            "--train",
            str(Path(args.train)),
            "--fold-manifest",
            str(Path(args.fold_manifest)),
            "--fold",
            str(args.fold),
            "--config",
            str(Path(args.base_config)),
            "--output-dir",
            output_dir,
            "--device",
            args.device,
        ]
    )
    train(config, train_args)


if __name__ == "__main__":
    main()
