"""Amino-acid vocabularies and physicochemical lookup tables."""

from __future__ import annotations

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "X"

AA_TO_IDX = {PAD_TOKEN: 0, UNK_TOKEN: 1}
AA_TO_IDX.update({aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)})
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

FUNCTIONAL_GROUPS = {
    "none": 0,
    "thiol": 1,
    "carboxylate": 2,
    "amide": 3,
    "imidazole": 4,
    "primary_amine": 5,
    "guanidinium": 6,
    "thioether": 7,
    "aromatic": 8,
    "indole": 9,
    "phenol": 10,
    "hydroxyl": 11,
    "proline_ring": 12,
}

AA_TO_GROUP = {
    "A": "none",
    "C": "thiol",
    "D": "carboxylate",
    "E": "carboxylate",
    "F": "aromatic",
    "G": "none",
    "H": "imidazole",
    "I": "none",
    "K": "primary_amine",
    "L": "none",
    "M": "thioether",
    "N": "amide",
    "P": "proline_ring",
    "Q": "amide",
    "R": "guanidinium",
    "S": "hydroxyl",
    "T": "hydroxyl",
    "V": "none",
    "W": "indole",
    "Y": "phenol",
}

# Kyte-Doolittle hydropathy, roughly normalized in feature extraction.
HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

MOLECULAR_WEIGHT = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

POSITIVE_AA = {"K", "R", "H"}
NEGATIVE_AA = {"D", "E"}
AROMATIC_AA = {"F", "W", "Y", "H"}
POLAR_AA = {"R", "N", "D", "Q", "E", "H", "K", "S", "T", "Y", "C"}
SULFUR_AA = {"C", "M"}
SPECIAL_TURN_AA = {"G", "P"}
