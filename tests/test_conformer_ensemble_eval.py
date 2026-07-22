from pathlib import Path

import numpy as np
import torch

from ghxtox.conformer_ensemble_eval import record_from_conformer_cache


def test_record_from_conformer_cache_preserves_base_confidence(tmp_path: Path) -> None:
    cache = tmp_path / "c.npz"
    np.savez_compressed(
        cache,
        sequence=np.asarray("AC"),
        coords=np.zeros((2, 3), dtype=np.float32),
        plddt=np.full(2, 0.7, dtype=np.float32),
        backbone_coords=np.zeros((2, 5, 3), dtype=np.float32),
        backbone_mask=np.ones((2, 5), dtype=bool),
        functional_group_coords=np.zeros((2, 3), dtype=np.float32),
        functional_group_mask=np.ones(2, dtype=bool),
        chemical_site_coords=np.zeros((2, 2, 3), dtype=np.float32),
        chemical_site_types=np.zeros((2, 2, 8), dtype=np.float32),
        chemical_site_orientations=np.zeros((2, 2, 3), dtype=np.float32),
        chemical_site_orientation_mask=np.zeros((2, 2), dtype=bool),
        chemical_site_mask=np.zeros((2, 2), dtype=bool),
    )
    base = {"sequence": "AC", "plddt": torch.tensor([0.9, 0.8]), "unchanged": 1}
    item = record_from_conformer_cache(base, cache, confidence_mode="base")
    assert torch.equal(item["plddt"], base["plddt"])
    assert item["unchanged"] == 1
    assert item["structure_features"].shape == (2, 16)
