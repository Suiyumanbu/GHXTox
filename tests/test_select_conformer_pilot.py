from pathlib import Path

from ghxtox.select_conformer_pilot import select_pilot


def test_select_pilot_preserves_source_ids_and_strata(tmp_path: Path) -> None:
    fasta = tmp_path / "train.fasta"
    fasta.write_text(
        ">pep|1\nACCA\n>pep|0\nAAAA\n>pep|1\nCCCC\n>pep|0\nGGGG\n",
        encoding="utf-8",
    )
    folds = tmp_path / "folds.csv"
    folds.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        "0,pep_1,1,ACCA,g0,0\n1,pep_2,0,AAAA,g1,0\n"
        "2,pep_3,1,CCCC,g2,1\n3,pep_4,0,GGGG,g3,1\n",
        encoding="utf-8",
    )
    rows = select_pilot(fasta, folds, tmp_path / "pilot.fasta", tmp_path / "pilot.csv", target_size=4)
    assert {row["sample_id"] for row in rows} == {"pep_1", "pep_2", "pep_3", "pep_4"}
    text = (tmp_path / "pilot.fasta").read_text(encoding="utf-8")
    assert ">pep_1|1" in text


def test_select_pilot_excludes_already_generated_source_indices(tmp_path: Path) -> None:
    fasta = tmp_path / "train.fasta"
    fasta.write_text(">a|1\nAAAA\n>b|0\nBBBB\n>c|1\nCCCC\n", encoding="utf-8")
    folds = tmp_path / "folds.csv"
    folds.write_text(
        "source_index,sample_id,label,sequence,group_id,fold\n"
        "0,pep_1,1,AAAA,g0,0\n1,pep_2,0,BBBB,g1,1\n2,pep_3,1,CCCC,g2,2\n",
        encoding="utf-8",
    )
    excluded = tmp_path / "generated.csv"
    excluded.write_text("source_index,sample_id\n1,pep_2\n", encoding="utf-8")
    rows = select_pilot(
        fasta,
        folds,
        tmp_path / "remaining.fasta",
        tmp_path / "remaining.csv",
        target_size=10,
        max_length=15,
        exclude_manifest=excluded,
    )
    assert [row["source_index"] for row in rows] == [0, 2]
