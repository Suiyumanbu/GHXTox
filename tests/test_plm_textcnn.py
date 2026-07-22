from __future__ import annotations

import csv

import numpy as np
import torch

from ghxtox.plm_textcnn import (
    PooledPLMTextCNN,
    _read_labeled_fasta,
    _probability_metrics,
    borderline_smote,
    ensemble_prediction_files,
    import_feature_csv,
    predict_fold_ensemble,
)


def test_textcnn_forward_shape() -> None:
    model = PooledPLMTextCNN(input_dim=32, projection_dim=16, window_sizes=(3, 4, 5), num_filters=8)
    logits = model(torch.randn(7, 32))
    assert logits.shape == (7,)


def test_borderline_smote_balances_only_positive_class() -> None:
    negatives = np.asarray([[float(index), 0.0] for index in range(8)], dtype=np.float32)
    positives = np.asarray([[2.1, 0.1], [3.1, -0.1], [4.1, 0.1], [5.1, -0.1]], dtype=np.float32)
    features = np.concatenate([negatives, positives], axis=0)
    labels = np.asarray([0] * len(negatives) + [1] * len(positives), dtype=np.int64)
    output_features, output_labels, report = borderline_smote(
        features,
        labels,
        k_neighbors=2,
        m_neighbors=4,
        seed=42,
        neighbor_backend="torch",
    )
    assert output_features.shape == (16, 2)
    assert int((output_labels == 0).sum()) == 8
    assert int((output_labels == 1).sum()) == 8
    assert report.synthetic_positive == 4
    assert report.danger_positive > 0


def test_probability_metrics_known_confusion_matrix() -> None:
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.4, 0.6, 0.1], dtype=np.float64)
    metrics = _probability_metrics(labels, probabilities, threshold=0.5)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["mcc"] == 0.0
    assert "brier" in metrics
    assert "ece_10" in metrics


def test_import_feature_csv_requires_fasta_label_alignment(tmp_path) -> None:
    fasta = tmp_path / "split.fasta"
    fasta.write_text(">sample_a|1\nACD\n>sample_b|0\nGG\n", encoding="utf-8")
    csv_path = tmp_path / "features.csv"
    np.savetxt(
        csv_path,
        np.asarray([[0.1, 0.2, 1.0], [0.3, 0.4, 0.0]], dtype=np.float32),
        delimiter=",",
    )
    output = tmp_path / "features.pt"
    metadata = import_feature_csv(csv_path, fasta, output)
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert _read_labeled_fasta(fasta) == (["sample_a|1", "sample_b|0"], ["ACD", "GG"], [1, 0])
    assert payload["features"].shape == (2, 2)
    assert payload["labels"].tolist() == [1, 0]
    assert metadata["feature_dim"] == 2


def test_predict_fold_ensemble_writes_aligned_probabilities(tmp_path) -> None:
    features_path = tmp_path / "features.pt"
    features = torch.randn(4, 8)
    torch.save(
        {
            "features": features,
            "labels": torch.tensor([0, 1, 0, 1]),
            "sample_ids": ["a", "b", "c", "d"],
            "sequences": ["AA", "CC", "DD", "EE"],
            "metadata": {"source_model": "test"},
        },
        features_path,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for fold in range(5):
        model = PooledPLMTextCNN(
            input_dim=8, projection_dim=8, window_sizes=(2, 3), num_filters=4
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "model": {
                    "input_dim": 8,
                    "projection_dim": 8,
                    "window_sizes": [2, 3],
                    "num_filters": 4,
                    "dropout": 0.2,
                },
            },
            checkpoint_dir / f"fold{fold}_best_model.pt",
        )
    output = tmp_path / "predictions.csv"
    summary = predict_fold_ensemble(
        features_path,
        checkpoint_dir,
        output,
        device_name="cpu",
        batch_size=4,
        threshold=0.5,
    )
    assert output.exists()
    assert summary["protocol"]["ensemble"].startswith("arithmetic mean")
    assert summary["metrics"]["tp"] + summary["metrics"]["fn"] == 2


def test_ensemble_prediction_files_preserves_fold_column(tmp_path) -> None:
    paths = []
    for member, probabilities in enumerate(([0.1, 0.9], [0.3, 0.7])):
        path = tmp_path / f"member{member}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source_index",
                    "fold",
                    "sample_id",
                    "sequence",
                    "label",
                    "toxicity_probability",
                ],
            )
            writer.writeheader()
            for index, probability in enumerate(probabilities):
                writer.writerow(
                    {
                        "source_index": index,
                        "fold": index,
                        "sample_id": f"sample_{index}",
                        "sequence": "ACD"[: index + 2],
                        "label": index,
                        "toxicity_probability": probability,
                    }
                )
        paths.append(path)
    output = tmp_path / "ensemble.csv"
    summary = ensemble_prediction_files(paths, output, threshold=0.5)
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fold"] == "0"
    assert float(rows[0]["toxicity_probability"]) == 0.2
    assert summary["metrics"]["accuracy"] == 1.0
