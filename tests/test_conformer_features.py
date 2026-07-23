from pathlib import Path

import numpy as np
import torch

from ghxtox.features import RESIDUE_FEATURE_DIM
from ghxtox.conformer_features import (
    CONFORMER_GLOBAL_FEATURE_DIM,
    CONFORMER_RESIDUE_FEATURE_DIM,
    attach_conformer_features,
    ensemble_feature_tensors,
)
from ghxtox.data import collate_peptides
from ghxtox.models import GHXToxModel


def _summary(path: Path, length: int = 3) -> None:
    coords = np.zeros((2, length, 3), dtype=np.float32)
    coords[1, -1, 1] = 2.0
    distances = np.linalg.norm(coords[:, :, None] - coords[:, None, :], axis=-1)
    np.savez_compressed(
        path,
        aligned_coords=coords,
        ensemble_mean_coords=coords.mean(axis=0),
        residue_rmsf=np.asarray([0.0, 0.2, 1.0], dtype=np.float32),
        conformer_rmsd=np.asarray([0.5, 0.5], dtype=np.float32),
        radius_gyration=np.asarray([1.0, 1.5], dtype=np.float32),
        pair_distance_mean=distances.mean(axis=0),
        pair_distance_std=distances.std(axis=0),
        contact_occupancy_6=(distances < 6).mean(axis=0).astype(np.float32),
        contact_occupancy_8=(distances < 8).mean(axis=0).astype(np.float32),
        contact_occupancy_10=(distances < 10).mean(axis=0).astype(np.float32),
    )


def _record(sample_id: str, available: bool = False) -> dict:
    length = 3
    record = {
        "sample_id": sample_id,
        "sequence": "ACD",
        "label": 1,
        "aa_ids": torch.tensor([1, 2, 3]),
        "group_ids": torch.tensor([0, 0, 0]),
        "residue_features": torch.zeros(length, RESIDUE_FEATURE_DIM),
        "coords": torch.zeros(length, 3),
        "plddt": torch.ones(length),
        "structure_features": torch.zeros(length, 16),
    }
    if available:
        record["conformer_residue_features"] = torch.ones(length, CONFORMER_RESIDUE_FEATURE_DIM)
        record["conformer_global_features"] = torch.ones(CONFORMER_GLOBAL_FEATURE_DIM)
        record["conformer_available"] = True
    return record


def test_ensemble_features_have_stable_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "ensemble_summary.npz"
    _summary(path)
    residue, global_value = ensemble_feature_tensors(path)
    assert residue.shape == (3, CONFORMER_RESIDUE_FEATURE_DIM)
    assert global_value.shape == (CONFORMER_GLOBAL_FEATURE_DIM,)
    assert torch.isfinite(residue).all()


def test_attach_maps_indexed_cache_to_original_id(tmp_path: Path) -> None:
    processed = tmp_path / "processed.pt"
    torch.save({"records": [_record("pep_1"), _record("pep_2")]}, processed)
    cache = tmp_path / "cache" / "pep_1_1"
    cache.mkdir(parents=True)
    _summary(cache / "ensemble_summary.npz")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "source_index,sample_id,original_id,sequence,label,conformer_index,cache_path\n"
        "0,pep_1_1,pep_1,ACD,1,0,/server/conformer_0000.npz\n",
        encoding="utf-8",
    )
    output = tmp_path / "augmented.pt"
    summary = attach_conformer_features(processed, manifest, tmp_path / "cache", output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert summary["covered_records"] == 1
    assert payload["records"][0]["conformer_available"] is True
    assert payload["records"][1]["conformer_available"] is False


def test_unavailable_conformer_branch_is_exact_fallback() -> None:
    config = {
        "model": {
            "hidden_dim": 16,
            "num_attention_heads": 4,
            "num_sequence_layers": 1,
            "num_egnn_layers": 1,
            "structure_feature_dim": 16,
            "conformer_ensemble_branch": True,
            "dropout": 0.0,
        }
    }
    model = GHXToxModel(config).eval()
    unavailable = collate_peptides([_record("pep_1")], include_atom=False)
    available = collate_peptides([_record("pep_1", available=True)], include_atom=False)
    with torch.no_grad():
        first = model(unavailable)
        second = model(available)
    assert torch.equal(first["logits"], first["base_logits"])
    # Zero initialization makes checkpoint warm-start behavior identical before training.
    assert torch.equal(second["logits"], second["base_logits"])
    with torch.no_grad():
        model.conformer_ensemble_branch[-1].bias.fill_(1.0)
    with torch.no_grad():
        changed = model(available)
        still_fallback = model(unavailable)
    assert not torch.equal(changed["logits"], changed["base_logits"])
    assert torch.equal(still_fallback["logits"], still_fallback["base_logits"])
