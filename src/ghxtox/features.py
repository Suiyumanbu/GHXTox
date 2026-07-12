"""Deterministic sequence and physicochemical feature extraction."""

from __future__ import annotations

import math

import torch

from ghxtox.constants import (
    AA_TO_GROUP,
    AA_TO_IDX,
    AROMATIC_AA,
    FUNCTIONAL_GROUPS,
    HYDROPATHY,
    MOLECULAR_WEIGHT,
    NEGATIVE_AA,
    POLAR_AA,
    POSITIVE_AA,
    SPECIAL_TURN_AA,
    SULFUR_AA,
    UNK_TOKEN,
)


RESIDUE_FEATURE_DIM = 42

ALIPHATIC_AA = {"A", "I", "L", "M", "V"}
BRANCHED_AA = {"I", "L", "V"}
SMALL_AA = {"A", "G", "S"}
HYDROXYL_AA = {"S", "T", "Y"}
AMIDE_AA = {"N", "Q"}
CARBOXYLATE_AA = {"D", "E"}
PRIMARY_AMINE_AA = {"K"}
GUANIDINIUM_AA = {"R"}
IMIDAZOLE_AA = {"H"}
PHENOL_AA = {"Y"}
INDOLE_AA = {"W"}


def clean_sequence(sequence: str) -> str:
    return "".join(aa if aa in AA_TO_IDX else UNK_TOKEN for aa in sequence.strip().upper())


def encode_amino_acids(sequence: str) -> torch.Tensor:
    sequence = clean_sequence(sequence)
    return torch.tensor([AA_TO_IDX.get(aa, AA_TO_IDX[UNK_TOKEN]) for aa in sequence], dtype=torch.long)


def encode_functional_groups(sequence: str) -> torch.Tensor:
    sequence = clean_sequence(sequence)
    ids = [
        FUNCTIONAL_GROUPS.get(AA_TO_GROUP.get(aa, "none"), FUNCTIONAL_GROUPS["none"])
        for aa in sequence
    ]
    return torch.tensor(ids, dtype=torch.long)


def _build_sequence_property_graph(sequence: str) -> list[list[float]]:
    length = len(sequence)
    graph = [[0.0 for _ in range(length)] for _ in range(length)]
    groups = [AA_TO_GROUP.get(aa, "none") for aa in sequence]
    charges = [1.0 if aa in POSITIVE_AA else -1.0 if aa in NEGATIVE_AA else 0.0 for aa in sequence]
    hydros = [HYDROPATHY.get(aa, 0.0) / 4.5 for aa in sequence]

    for i in range(length):
        for j in range(i + 1, length):
            seq_sep = j - i
            weight = 0.0
            if seq_sep <= 3:
                weight += 1.0 / seq_sep
            if groups[i] == groups[j] and groups[i] != "none":
                weight += 0.45
            if charges[i] * charges[j] < 0:
                weight += 0.35
            hydro_similarity = max(0.0, 1.0 - abs(hydros[i] - hydros[j]))
            if hydro_similarity > 0.55:
                weight += 0.25 * hydro_similarity
            if sequence[i] == "C" and sequence[j] == "C":
                weight += 0.6

            graph[i][j] = graph[j][i] = weight
    return graph


def _modularity(graph: list[list[float]], communities: list[int]) -> float:
    degrees = [sum(row) for row in graph]
    two_m = sum(degrees)
    if two_m <= 0.0:
        return 0.0

    score = 0.0
    for i, row in enumerate(graph):
        for j, weight in enumerate(row):
            if communities[i] == communities[j]:
                score += weight - degrees[i] * degrees[j] / two_m
    return score / two_m


def _renumber_communities(communities: list[int]) -> list[int]:
    mapping: dict[int, int] = {}
    renumbered = []
    for community in communities:
        if community not in mapping:
            mapping[community] = len(mapping)
        renumbered.append(mapping[community])
    return renumbered


def louvain_community_features(sequence: str) -> torch.Tensor:
    """Return Louvain-style community descriptors for the 2D sequence graph.

    The graph is a deterministic residue network built from local sequence
    adjacency and physicochemical affinity. Peptides are short, so a compact
    greedy modularity pass is sufficient and avoids an extra runtime dependency.
    """

    sequence = clean_sequence(sequence)
    length = len(sequence)
    if length == 0:
        return torch.zeros((0, 3), dtype=torch.float32)

    graph = _build_sequence_property_graph(sequence)
    try:
        import networkx as nx

        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(range(length))
        for i, row in enumerate(graph):
            for j in range(i + 1, length):
                if row[j] > 0.0:
                    nx_graph.add_edge(i, j, weight=row[j])
        communities_found = nx.community.louvain_communities(nx_graph, weight="weight", seed=0)
        communities = [0 for _ in range(length)]
        ordered = sorted(communities_found, key=lambda members: min(members) if members else length)
        for community_id, members in enumerate(ordered):
            for node in members:
                communities[node] = community_id
        return _community_feature_tensor(graph, communities)
    except Exception:
        pass

    communities = list(range(length))
    best_q = _modularity(graph, communities)

    for _ in range(20):
        moved = False
        for node in range(length):
            neighbor_communities = {
                communities[j] for j, weight in enumerate(graph[node]) if weight > 0.0
            }
            neighbor_communities.add(communities[node])
            current = communities[node]
            best_community = current
            local_best_q = best_q

            for candidate in sorted(neighbor_communities):
                if candidate == current:
                    continue
                trial = communities.copy()
                trial[node] = candidate
                q_value = _modularity(graph, trial)
                if q_value > local_best_q + 1e-9:
                    local_best_q = q_value
                    best_community = candidate

            if best_community != current:
                communities[node] = best_community
                best_q = local_best_q
                moved = True
        communities = _renumber_communities(communities)
        if not moved:
            break

    return _community_feature_tensor(graph, communities)


def _community_feature_tensor(graph: list[list[float]], communities: list[int]) -> torch.Tensor:
    length = len(communities)
    num_communities = max(communities) + 1 if communities else 1
    community_sizes = [communities.count(idx) for idx in range(num_communities)]
    degrees = [sum(row) for row in graph]
    max_degree = max(max(degrees), 1.0)
    denom = max(num_communities - 1, 1)
    rows = [
        [
            communities[i] / denom,
            community_sizes[communities[i]] / max(length, 1),
            degrees[i] / max_degree,
        ]
        for i in range(length)
    ]
    return torch.tensor(rows, dtype=torch.float32)


def residue_feature_matrix(sequence: str) -> torch.Tensor:
    """Return per-residue deterministic physicochemical descriptors.

    The final three columns are Louvain-style 2D topology descriptors:
    normalized community id, community size fraction, and weighted degree.
    """

    sequence = clean_sequence(sequence)
    length = max(len(sequence), 1)
    rows: list[list[float]] = []

    charges = [1.0 if aa in POSITIVE_AA else -1.0 if aa in NEGATIVE_AA else 0.0 for aa in sequence]
    hydros = [HYDROPATHY.get(aa, 0.0) / 4.5 for aa in sequence]
    positives = [1.0 if aa in POSITIVE_AA else 0.0 for aa in sequence]
    negatives = [1.0 if aa in NEGATIVE_AA else 0.0 for aa in sequence]
    polars = [1.0 if aa in POLAR_AA else 0.0 for aa in sequence]
    cysteines = [1.0 if aa == "C" else 0.0 for aa in sequence]
    aromatics = [1.0 if aa in AROMATIC_AA else 0.0 for aa in sequence]
    sulfurs = [1.0 if aa in SULFUR_AA else 0.0 for aa in sequence]
    hydroxyls = [1.0 if aa in HYDROXYL_AA else 0.0 for aa in sequence]
    amides = [1.0 if aa in AMIDE_AA else 0.0 for aa in sequence]
    carboxylates = [1.0 if aa in CARBOXYLATE_AA else 0.0 for aa in sequence]
    aliphatics = [1.0 if aa in ALIPHATIC_AA else 0.0 for aa in sequence]

    for i, aa in enumerate(sequence):
        left = max(0, i - 2)
        right = min(len(sequence), i + 3)
        window = slice(left, right)
        window_size = max(right - left, 1)
        rel_pos = i / max(length - 1, 1)
        n_term = math.exp(-i / 8.0)
        c_term = math.exp(-(length - 1 - i) / 8.0)
        rows.append(
            [
                hydros[i],
                charges[i],
                polars[i],
                aromatics[i],
                sulfurs[i],
                positives[i],
                negatives[i],
                1.0 if aa in SPECIAL_TURN_AA else 0.0,
                (MOLECULAR_WEIGHT.get(aa, 120.0) - 75.0) / 130.0,
                sum(charges[window]) / window_size,
                sum(hydros[window]) / window_size,
                sum(cysteines[window]) / window_size,
                sum(aromatics[window]) / window_size,
                rel_pos,
                n_term - c_term,
                1.0 if aa in ALIPHATIC_AA else 0.0,
                1.0 if aa in BRANCHED_AA else 0.0,
                1.0 if aa in SMALL_AA else 0.0,
                1.0 if aa == "G" else 0.0,
                1.0 if aa == "P" else 0.0,
                1.0 if aa == "C" else 0.0,
                1.0 if aa == "M" else 0.0,
                hydroxyls[i],
                amides[i],
                carboxylates[i],
                1.0 if aa in PRIMARY_AMINE_AA else 0.0,
                1.0 if aa in GUANIDINIUM_AA else 0.0,
                1.0 if aa in IMIDAZOLE_AA else 0.0,
                1.0 if aa in PHENOL_AA else 0.0,
                1.0 if aa in INDOLE_AA else 0.0,
                aromatics[i],
                sum(positives[window]) / window_size,
                sum(negatives[window]) / window_size,
                sum(polars[window]) / window_size,
                sum(sulfurs[window]) / window_size,
                sum(hydroxyls[window]) / window_size,
                sum(amides[window]) / window_size,
                sum(carboxylates[window]) / window_size,
                sum(aliphatics[window]) / window_size,
            ]
        )
    base = torch.tensor(rows, dtype=torch.float32)
    return torch.cat([base, louvain_community_features(sequence)], dim=-1)


def sequence_global_features(sequence: str) -> dict[str, float]:
    sequence = clean_sequence(sequence)
    length = max(len(sequence), 1)
    return {
        "length": float(length),
        "net_charge": sum(1 for aa in sequence if aa in POSITIVE_AA)
        - sum(1 for aa in sequence if aa in NEGATIVE_AA),
        "aromatic_fraction": sum(1 for aa in sequence if aa in AROMATIC_AA) / length,
        "cysteine_fraction": sequence.count("C") / length,
        "mean_hydropathy": sum(HYDROPATHY.get(aa, 0.0) for aa in sequence) / length,
    }
