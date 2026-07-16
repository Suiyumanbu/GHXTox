"""Attach full-atom chemistry-aware site tensors to an existing ESM2 dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ghxtox.chemical_sites import (
    CHEMICAL_SITE_TYPE_NAMES,
    MAX_CHEMICAL_SITES,
    parse_chemical_sites,
)
from ghxtox.esmfold_cache import parse_esmfold_pdb
from ghxtox.geometry_features import structure_feature_matrix


def attach_chemical_sites(
    processed_path: str | Path,
    pdb_dir: str | Path,
    output_path: str | Path,
) -> dict[str, int | float]:
    payload = torch.load(processed_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    pdb_paths = sorted(Path(pdb_dir).glob("*.pdb"))
    if len(pdb_paths) != len(records):
        raise ValueError(f"Record/PDB count mismatch: records={len(records)}, pdb={len(pdb_paths)}")

    site_count = 0
    residue_with_site_count = 0
    residue_count = 0
    output_records = []
    for index, (record, pdb_path) in enumerate(zip(records, pdb_paths, strict=True), start=1):
        parsed = parse_esmfold_pdb(pdb_path)
        sites = parse_chemical_sites(pdb_path)
        sequence = record["sequence"]
        if parsed.sequence != sequence or sites.sequence != sequence:
            raise ValueError(
                f"Sequence mismatch at record {index}: processed={sequence}, "
                f"structure={parsed.sequence}, sites={sites.sequence}, pdb={pdb_path}"
            )
        item = dict(record)
        item.update(
            {
                "coords": torch.from_numpy(parsed.coords),
                "plddt": torch.from_numpy(parsed.plddt),
                "backbone_coords": torch.from_numpy(parsed.backbone_coords),
                "backbone_mask": torch.from_numpy(parsed.backbone_mask),
                "functional_group_coords": torch.from_numpy(parsed.functional_group_coords),
                "functional_group_mask": torch.from_numpy(parsed.functional_group_mask),
                "structure_features": structure_feature_matrix(
                    torch.from_numpy(parsed.coords), torch.from_numpy(parsed.plddt)
                ),
                "chemical_site_coords": torch.from_numpy(sites.coords),
                "chemical_site_types": torch.from_numpy(sites.types),
                "chemical_site_orientations": torch.from_numpy(sites.orientations),
                "chemical_site_orientation_mask": torch.from_numpy(sites.orientation_mask),
                "chemical_site_mask": torch.from_numpy(sites.mask),
                "structure_source": str(pdb_path),
            }
        )
        output_records.append(item)
        site_count += int(sites.mask.sum())
        residue_with_site_count += int(sites.mask.any(axis=1).sum())
        residue_count += len(sequence)
        if index % 500 == 0 or index == len(records):
            print(f"attached={index}/{len(records)} sites={site_count}")

    output_payload = dict(payload)
    output_payload["records"] = output_records
    output_payload["chemical_site_schema"] = {
        "type_names": list(CHEMICAL_SITE_TYPE_NAMES),
        "max_sites_per_residue": MAX_CHEMICAL_SITES,
        "source": "full-atom ESMFold PDB",
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, output_path)
    return {
        "records": len(output_records),
        "residues": residue_count,
        "sites": site_count,
        "residues_with_sites": residue_with_site_count,
        "residue_coverage": residue_with_site_count / max(residue_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(attach_chemical_sites(args.processed, args.pdb_dir, args.output))


if __name__ == "__main__":
    main()
