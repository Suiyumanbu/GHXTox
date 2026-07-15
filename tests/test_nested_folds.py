import csv

from ghxtox.nested_folds import create_nested_group_folds, load_nested_indices


def test_nested_group_folds_keep_groups_disjoint(tmp_path) -> None:
    base = tmp_path / "base.csv"
    rows = []
    index = 0
    for fold in range(5):
        for group_offset in range(16):
            label = group_offset % 2
            for duplicate in range(2 if group_offset == 0 else 1):
                rows.append(
                    {
                        "source_index": index,
                        "sample_id": f"sample_{index}",
                        "label": label,
                        "sequence": f"SEQ{index}",
                        "group_id": f"fold{fold}_group{group_offset}",
                        "fold": fold,
                    }
                )
                index += 1
    with base.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_csv = tmp_path / "nested.csv"
    output_json = tmp_path / "nested.json"
    summary = create_nested_group_folds(base, output_csv, output_json, inner_splits=4, seed=7)
    assert summary["validation"]["groups_disjoint_within_every_outer_fold"] is True
    for outer_fold in range(5):
        roles = load_nested_indices(output_csv, outer_fold, len(rows))
        assert set().union(*map(set, roles.values())) == set(range(len(rows)))
        assert all(roles[role] for role in roles)
