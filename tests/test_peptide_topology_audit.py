from pathlib import Path

from ghxtox.peptide_topology_audit import audit_fasta, run_audit


def test_topology_audit_separates_annotations_from_sequence_heuristics(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(
        ">linear|0\nAAAA\n"
        ">possible|1\nACCA\n"
        ">rich|1\nCCCCCC\n"
        ">declared cyclic peptide|0\nAGGG\n",
        encoding="utf-8",
    )

    rows = audit_fasta(fasta, split="train")

    assert rows[0]["topology_risk"] == "no_sequence_visible_topology_flag"
    assert rows[1]["topology_risk"] == "possible_disulfide"
    assert rows[2]["topology_risk"] == "high_cysteine_topology_risk"
    assert rows[3]["topology_risk"] == "explicit_cyclic"

    summary = run_audit([str(fasta)], ["train"], tmp_path / "out")
    assert summary["splits"]["train"]["samples"] == 4
    assert summary["splits"]["train"]["topology_flagged"] == 3
    assert (tmp_path / "out" / "sample_topology_audit.csv").exists()
