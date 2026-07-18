from __future__ import annotations

import pytest
import torch

from ghxtox.prott5_tokens import FORMAT_VERSION, validate_token_cache


def _write_cache(tmp_path, offsets: list[int]) -> tuple[object, object]:
    fasta = tmp_path / "split.fasta"
    fasta.write_text(">a|1\nACD\n>b|0\nGG\n", encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "embeddings": torch.randn(5, 8, dtype=torch.float16),
            "offsets": torch.tensor(offsets),
            "labels": torch.tensor([1, 0]),
            "sample_ids": ["a|1", "b|0"],
            "sequences": ["ACD", "GG"],
            "metadata": {"model_path": "test-model"},
        },
        cache / "shard_00000.pt",
    )
    return fasta, cache


def test_validate_token_cache_accepts_residue_aligned_flat_storage(tmp_path) -> None:
    fasta, cache = _write_cache(tmp_path, [0, 3, 5])
    result = validate_token_cache(fasta, cache, model_path="test-model")
    assert result["num_samples"] == 2
    assert result["num_residues"] == 5
    assert result["feature_dim"] == 8
    assert result["storage_dtype"] == "float16"


def test_validate_token_cache_rejects_bad_offsets(tmp_path) -> None:
    fasta, cache = _write_cache(tmp_path, [0, 2, 5])
    with pytest.raises(ValueError, match="offsets"):
        validate_token_cache(fasta, cache)
