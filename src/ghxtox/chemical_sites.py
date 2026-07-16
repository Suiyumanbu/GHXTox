"""Chemistry-aware pseudo-sites extracted from full-atom peptide PDB files.

The representation deliberately keeps chemically distinct moieties separate.
For example, Tyr contributes an aromatic site and a hydroxyl site instead of a
single side-chain centroid.  Site orientations are either a bond direction or
an unsigned plane normal; downstream invariant features therefore use absolute
orientation cosines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ghxtox.esmfold_cache import THREE_TO_ONE


CHEMICAL_SITE_TYPE_NAMES = (
    "positive",
    "negative",
    "donor",
    "acceptor",
    "aromatic",
    "hydrophobic",
    "sulfur",
    "conformational",
)
CHEMICAL_SITE_TYPE_TO_INDEX = {
    name: index for index, name in enumerate(CHEMICAL_SITE_TYPE_NAMES)
}
CHEMICAL_SITE_TYPE_DIM = len(CHEMICAL_SITE_TYPE_NAMES)
MAX_CHEMICAL_SITES = 2


@dataclass(frozen=True)
class SiteSpec:
    atoms: tuple[str, ...]
    types: tuple[str, ...]
    direction_atoms: tuple[str, str] | None = None
    plane_atoms: tuple[str, str, str] | None = None


SITE_SPECS: dict[str, tuple[SiteSpec, ...]] = {
    "ALA": (SiteSpec(("CB",), ("hydrophobic",)),),
    "VAL": (SiteSpec(("CG1", "CG2"), ("hydrophobic",)),),
    "LEU": (SiteSpec(("CD1", "CD2"), ("hydrophobic",)),),
    "ILE": (SiteSpec(("CD1", "CG2"), ("hydrophobic",)),),
    "PRO": (
        SiteSpec(("CB", "CG", "CD"), ("hydrophobic", "conformational"), plane_atoms=("CB", "CG", "CD")),
    ),
    "CYS": (SiteSpec(("SG",), ("donor", "hydrophobic", "sulfur"), direction_atoms=("CB", "SG")),),
    "MET": (SiteSpec(("SD",), ("hydrophobic", "sulfur"), direction_atoms=("CG", "SD")),),
    "SER": (SiteSpec(("OG",), ("donor", "acceptor"), direction_atoms=("CB", "OG")),),
    "THR": (SiteSpec(("OG1",), ("donor", "acceptor"), direction_atoms=("CB", "OG1")),),
    "LYS": (SiteSpec(("NZ",), ("positive", "donor"), direction_atoms=("CE", "NZ")),),
    "ARG": (
        SiteSpec(
            ("NE", "CZ", "NH1", "NH2"),
            ("positive", "donor"),
            plane_atoms=("CZ", "NH1", "NH2"),
        ),
    ),
    "ASP": (
        SiteSpec(("CG", "OD1", "OD2"), ("negative", "acceptor"), plane_atoms=("CG", "OD1", "OD2")),
    ),
    "GLU": (
        SiteSpec(("CD", "OE1", "OE2"), ("negative", "acceptor"), plane_atoms=("CD", "OE1", "OE2")),
    ),
    "ASN": (
        SiteSpec(("CG", "OD1", "ND2"), ("donor", "acceptor"), plane_atoms=("CG", "OD1", "ND2")),
    ),
    "GLN": (
        SiteSpec(("CD", "OE1", "NE2"), ("donor", "acceptor"), plane_atoms=("CD", "OE1", "NE2")),
    ),
    "HIS": (
        SiteSpec(
            ("CG", "ND1", "CD2", "CE1", "NE2"),
            ("donor", "acceptor", "aromatic"),
            plane_atoms=("CG", "ND1", "CD2"),
        ),
    ),
    "PHE": (
        SiteSpec(
            ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
            ("aromatic", "hydrophobic"),
            plane_atoms=("CG", "CD1", "CD2"),
        ),
    ),
    "TYR": (
        SiteSpec(
            ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
            ("aromatic", "hydrophobic"),
            plane_atoms=("CG", "CD1", "CD2"),
        ),
        SiteSpec(("OH",), ("donor", "acceptor"), direction_atoms=("CZ", "OH")),
    ),
    "TRP": (
        SiteSpec(
            ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
            ("aromatic", "hydrophobic"),
            plane_atoms=("CG", "CD1", "CD2"),
        ),
        SiteSpec(("NE1",), ("donor",), direction_atoms=("CD1", "NE1")),
    ),
}


@dataclass(frozen=True)
class ChemicalSiteData:
    sequence: str
    coords: np.ndarray
    types: np.ndarray
    orientations: np.ndarray
    orientation_mask: np.ndarray
    mask: np.ndarray


def _unit(vector: np.ndarray) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.zeros(3, dtype=np.float32), False
    return (vector / norm).astype(np.float32), True


def _orientation(
    atoms: dict[str, np.ndarray],
    center: np.ndarray,
    spec: SiteSpec,
) -> tuple[np.ndarray, bool]:
    if spec.plane_atoms is not None and all(name in atoms for name in spec.plane_atoms):
        first, second, third = (atoms[name] for name in spec.plane_atoms)
        return _unit(np.cross(second - first, third - first))
    if spec.direction_atoms is not None and all(name in atoms for name in spec.direction_atoms):
        start, end = (atoms[name] for name in spec.direction_atoms)
        return _unit(end - start)
    if "CA" in atoms:
        return _unit(center - atoms["CA"])
    return np.zeros(3, dtype=np.float32), False


def _pdb_residues(path: str | Path) -> list[tuple[str, dict[str, np.ndarray]]]:
    residues: list[tuple[str, dict[str, np.ndarray]]] = []
    indices: dict[tuple[str, str, str], int] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            key = (line[21].strip(), line[22:26].strip(), line[26].strip())
            index = indices.get(key)
            if index is None:
                index = len(residues)
                indices[key] = index
                residues.append((line[17:20].strip().upper(), {}))
            atoms = residues[index][1]
            if atom_name in atoms:
                continue
            atoms[atom_name] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float32,
            )
    if not residues:
        raise ValueError(f"No ATOM records found in {path}.")
    return residues


def parse_chemical_sites(path: str | Path) -> ChemicalSiteData:
    residues = _pdb_residues(path)
    ca_coords = []
    for index, (_, atoms) in enumerate(residues):
        if "CA" not in atoms:
            raise ValueError(f"Residue {index + 1} in {path} is missing CA coordinates.")
        ca_coords.append(atoms["CA"])
    center = np.stack(ca_coords).mean(axis=0)

    length = len(residues)
    coords = np.zeros((length, MAX_CHEMICAL_SITES, 3), dtype=np.float32)
    types = np.zeros((length, MAX_CHEMICAL_SITES, CHEMICAL_SITE_TYPE_DIM), dtype=np.float32)
    orientations = np.zeros((length, MAX_CHEMICAL_SITES, 3), dtype=np.float32)
    orientation_mask = np.zeros((length, MAX_CHEMICAL_SITES), dtype=bool)
    mask = np.zeros((length, MAX_CHEMICAL_SITES), dtype=bool)
    sequence = []

    for residue_index, (resname, atoms) in enumerate(residues):
        sequence.append(THREE_TO_ONE.get(resname, "X"))
        for site_index, spec in enumerate(SITE_SPECS.get(resname, ())[:MAX_CHEMICAL_SITES]):
            if not all(name in atoms for name in spec.atoms):
                continue
            site_coord = np.stack([atoms[name] for name in spec.atoms]).mean(axis=0)
            coords[residue_index, site_index] = site_coord - center
            for site_type in spec.types:
                types[residue_index, site_index, CHEMICAL_SITE_TYPE_TO_INDEX[site_type]] = 1.0
            orientation, valid_orientation = _orientation(atoms, site_coord, spec)
            orientations[residue_index, site_index] = orientation
            orientation_mask[residue_index, site_index] = valid_orientation
            mask[residue_index, site_index] = True

    return ChemicalSiteData(
        sequence="".join(sequence),
        coords=coords,
        types=types,
        orientations=orientations,
        orientation_mask=orientation_mask,
        mask=mask,
    )


def site_type_indices(names: Iterable[str]) -> tuple[int, ...]:
    return tuple(CHEMICAL_SITE_TYPE_TO_INDEX[name] for name in names)
