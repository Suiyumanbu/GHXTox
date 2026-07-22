from pathlib import Path

import numpy as np

from ghxtox.conformer_cache import build_conformer_cache, ensemble_descriptors, pdb_model_blocks


def _model(offset: float) -> str:
    return "\n".join(
        [
            "MODEL        1",
            f"ATOM      1  N   ALA A   1    {offset + 0:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           N",
            f"ATOM      2  CA  ALA A   1    {offset + 1:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM      3  C   ALA A   1    {offset + 2:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM      4  O   ALA A   1    {offset + 3:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           O",
            f"ATOM      5  CB  ALA A   1    {offset + 1:8.3f}{1:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM      6  N   CYS A   2    {offset + 3:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           N",
            f"ATOM      7  CA  CYS A   2    {offset + 4:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM      8  C   CYS A   2    {offset + 5:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM      9  O   CYS A   2    {offset + 6:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           O",
            f"ATOM     10  CB  CYS A   2    {offset + 4:8.3f}{1:8.3f}{0:8.3f}  1.00  0.00           C",
            f"ATOM     11  SG  CYS A   2    {offset + 4:8.3f}{2:8.3f}{0:8.3f}  1.00  0.00           S",
            "ENDMDL",
        ]
    )


def test_multimodel_cache_and_descriptors(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(">pep|1\nAC\n", encoding="utf-8")
    pdb_dir = tmp_path / "pdb" / "pep_1"
    pdb_dir.mkdir(parents=True)
    pdb_path = pdb_dir / "AC.pdb"
    second = _model(10.0).replace("MODEL        1", "MODEL        2")
    pdb_path.write_text(_model(0.0) + "\n" + second + "\nEND\n", encoding="utf-8")

    assert len(pdb_model_blocks(pdb_path)) == 2
    summary = build_conformer_cache(fasta, tmp_path / "pdb", tmp_path / "cache")
    assert summary["total_conformers"] == 2
    cache = np.load(tmp_path / "cache" / "pep_1" / "conformer_0000.npz")
    assert cache["coords"].shape == (2, 3)
    assert bool(cache["confidence_imputed"])
    ensemble = np.load(tmp_path / "cache" / "pep_1" / "ensemble_summary.npz")
    assert np.allclose(ensemble["residue_rmsf"], 0.0, atol=1e-5)


def test_ensemble_descriptors_detect_flexibility() -> None:
    coords = np.asarray(
        [
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    descriptors = ensemble_descriptors(coords)
    assert descriptors["residue_rmsf"].max() > 0
    assert descriptors["contact_occupancy_6"].shape == (3, 3)
