"""Build RDKit peptide atom graphs and attach them to processed datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


ELEMENTS = (6, 7, 8, 16, 15)
ATOM_FEATURE_DIM = 30
EXTENDED_ATOM_FEATURE_DIM = 34
EDGE_FEATURE_DIM = 11


def _one_hot(value: Any, choices: tuple[Any, ...], include_other: bool = True) -> list[float]:
    values = [float(value == choice) for choice in choices]
    if include_other:
        values.append(float(value not in choices))
    return values


def peptide_atom_graph(sequence: str, feature_set: str = "base") -> dict[str, torch.Tensor]:
    """Convert a canonical peptide sequence into a directed molecular graph."""

    if feature_set not in {"base", "extended"}:
        raise ValueError(f"Unsupported atom feature set {feature_set!r}.")
    atom_feature_dim = EXTENDED_ATOM_FEATURE_DIM if feature_set == "extended" else ATOM_FEATURE_DIM

    try:
        from rdkit import Chem
        from rdkit.Chem import rdPartialCharges
    except ImportError as exc:
        raise RuntimeError("RDKit is required for atom graphs. Install it with `conda install -c conda-forge rdkit`.") from exc

    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        raise ValueError(f"RDKit could not construct a peptide for sequence {sequence!r}.")
    rdPartialCharges.ComputeGasteigerCharges(mol)

    atom_rows: list[list[float]] = []
    atom_residue_indices: list[int] = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        atom_name = info.GetName().strip() if info is not None else ""
        residue_number = info.GetResidueNumber() if info is not None else 0
        backbone = atom_name if atom_name in {"N", "CA", "C", "O"} else "sidechain"
        charge_text = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0"
        try:
            partial_charge = float(charge_text)
        except ValueError:
            partial_charge = 0.0
        if partial_charge != partial_charge or abs(partial_charge) == float("inf"):
            partial_charge = 0.0

        core = (
            _one_hot(atom.GetAtomicNum(), ELEMENTS)
            + _one_hot(min(atom.GetDegree(), 5), (0, 1, 2, 3, 4))
            + [float(atom.GetFormalCharge()) / 2.0]
            + _one_hot(
                atom.GetHybridization(),
                (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3),
            )
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
            + _one_hot(
                atom.GetChiralTag(),
                (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW),
            )
            + _one_hot(backbone, ("N", "CA", "C", "O"))
        )
        extended = (
            [
                min(float(atom.GetTotalNumHs()), 4.0) / 4.0,
                min(float(atom.GetValence(Chem.ValenceType.IMPLICIT)), 6.0) / 6.0,
                float(atom.IsInRingSize(5)),
                float(atom.IsInRingSize(6)),
            ]
            if feature_set == "extended"
            else []
        )
        row = core + extended + [
            float(residue_number) / max(len(sequence), 1),
            max(-2.0, min(2.0, partial_charge)) / 2.0,
        ]
        if len(row) != atom_feature_dim:
            raise AssertionError(f"Expected {atom_feature_dim} atom features, got {len(row)}.")
        atom_rows.append(row)
        atom_residue_indices.append(max(residue_number - 1, 0))

    edges: list[list[int]] = []
    edge_rows: list[list[float]] = []
    bond_types = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
    stereo_types = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_info = begin.GetPDBResidueInfo()
        end_info = end.GetPDBResidueInfo()
        begin_name = begin_info.GetName().strip() if begin_info is not None else ""
        end_name = end_info.GetName().strip() if end_info is not None else ""
        begin_residue = begin_info.GetResidueNumber() if begin_info is not None else -1
        end_residue = end_info.GetResidueNumber() if end_info is not None else -1
        peptide_bond = (
            abs(begin_residue - end_residue) == 1
            and {begin_name, end_name} == {"C", "N"}
        )
        features = (
            _one_hot(bond.GetBondType(), bond_types, include_other=False)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing()), float(peptide_bond)]
            + _one_hot(bond.GetStereo(), stereo_types)
        )
        if len(features) != EDGE_FEATURE_DIM:
            raise AssertionError(f"Expected {EDGE_FEATURE_DIM} edge features, got {len(features)}.")
        for source, target in ((begin.GetIdx(), end.GetIdx()), (end.GetIdx(), begin.GetIdx())):
            edges.append([source, target])
            edge_rows.append(features)

    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    return {
        "atom_features": torch.tensor(atom_rows, dtype=torch.float32),
        "atom_residue_index": torch.tensor(atom_residue_indices, dtype=torch.long),
        "atom_edge_index": edge_index,
        "atom_edge_features": torch.tensor(edge_rows, dtype=torch.float32),
    }


def attach_atom_graphs(
    input_path: str | Path,
    output_path: str | Path,
    feature_set: str = "base",
) -> dict[str, Any]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    records = payload["records"]
    for record in records:
        record.update(peptide_atom_graph(record["sequence"], feature_set=feature_set))
    atom_feature_dim = EXTENDED_ATOM_FEATURE_DIM if feature_set == "extended" else ATOM_FEATURE_DIM
    payload["atom_graph"] = {
        "feature_set": feature_set,
        "atom_feature_dim": atom_feature_dim,
        "edge_feature_dim": EDGE_FEATURE_DIM,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {"records": len(records), "output": str(output_path)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach RDKit atom graphs to a processed GHXTox dataset.")
    parser.add_argument("--input", required=True, help="Existing processed .pt file.")
    parser.add_argument("--output", required=True, help="Output .pt file with atom graphs.")
    parser.add_argument(
        "--feature-set",
        choices=["base", "extended"],
        default="base",
        help="Use the original 30D features or the paper-aligned extended 34D features.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    stats = attach_atom_graphs(args.input, args.output, feature_set=args.feature_set)
    print(f"Saved atom graphs for {stats['records']} records to {stats['output']}.")


if __name__ == "__main__":
    main()
