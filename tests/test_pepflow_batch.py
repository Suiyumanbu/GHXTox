from pathlib import Path

from ghxtox.pepflow_batch import build_command, run_batch


def test_build_command_uses_official_pepflow_entrypoint() -> None:
    command = build_command(
        "python",
        "/opt/pepflow",
        "ACDE",
        "/tmp/out",
        "/opt/pepflow/params/full_model.pth",
        8,
        4,
    )
    assert command[1].endswith("generate_peptide_samples.py")
    assert command[command.index("-n") + 1] == "8"
    assert command[-1] == "--e"


def test_dry_run_keeps_duplicate_sequences_separate(tmp_path: Path) -> None:
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">a|1\nACDE\n>b|0\nACDE\n", encoding="utf-8")
    summary = run_batch(
        fasta,
        "/opt/pepflow",
        "/opt/pepflow/params/full_model.pth",
        tmp_path / "out",
        dry_run=True,
    )
    assert summary["counts"] == {"dry_run": 2}
    paths = [row["pdb_path"] for row in summary["records"]]
    assert paths[0] != paths[1]


def test_header_mode_preserves_preindexed_ids(tmp_path: Path) -> None:
    fasta = tmp_path / "pilot.fasta"
    fasta.write_text(">peptide_42|1\nACDE\n", encoding="utf-8")
    summary = run_batch(
        fasta,
        "/opt/pepflow",
        "/opt/pepflow/params/full_model.pth",
        tmp_path / "out",
        dry_run=True,
        sample_id_mode="header",
    )
    assert summary["records"][0]["sample_id"] == "peptide_42"
