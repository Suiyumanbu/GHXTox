from __future__ import annotations

import numpy as np
import pytest
import torch

from ghxtox.hybrid_default import combine_probabilities, validate_alignment


def test_combine_probabilities_uses_frozen_convex_weight() -> None:
    structure = np.asarray([0.2, 0.8])
    prott5 = np.asarray([0.6, 0.4])
    combined = combine_probabilities(structure, prott5, prott5_weight=0.62)
    np.testing.assert_allclose(combined, [0.448, 0.552])


def test_validate_alignment_accepts_sample_id_aliases_but_not_sequence_changes() -> None:
    payload = {
        "sequences": ["ACD", "GG"],
        "labels": torch.tensor([1, 0]),
        "sample_ids": ["another_a", "another_b"],
    }
    validate_alignment(["ACD", "GG"], [1, 0], payload)
    with pytest.raises(ValueError, match="sequence alignment"):
        validate_alignment(["ACD", "GA"], [1, 0], payload)


def test_validate_alignment_rejects_label_mismatch() -> None:
    payload = {"sequences": ["ACD"], "labels": torch.tensor([0])}
    with pytest.raises(ValueError, match="label alignment"):
        validate_alignment(["ACD"], [1], payload)
