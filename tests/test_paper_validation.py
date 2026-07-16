import numpy as np

from ghxtox.paper_validation import paired_model_comparison, select_representative_cases


def test_paired_model_comparison_detects_identical_models():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    predictions = scores >= 0.5
    result = paired_model_comparison(
        labels,
        scores,
        scores,
        predictions,
        predictions,
        bootstrap_iterations=100,
        permutation_iterations=100,
        seed=7,
    )
    for metric in result["metrics"].values():
        assert metric["delta_v2_minus_v1"] == 0.0
        assert metric["paired_bootstrap_95_ci"] == {"lower": 0.0, "upper": 0.0}
        assert metric["paired_randomization_two_sided_p"] == 1.0


def test_representative_case_selection_covers_confusion_groups():
    rows = [
        {"sample_id": "tp_hi", "label": "1", "toxicity_probability": "0.95"},
        {"sample_id": "tp_edge", "label": "1", "toxicity_probability": "0.51"},
        {"sample_id": "tn_hi", "label": "0", "toxicity_probability": "0.02"},
        {"sample_id": "tn_edge", "label": "0", "toxicity_probability": "0.49"},
        {"sample_id": "fp_hi", "label": "0", "toxicity_probability": "0.91"},
        {"sample_id": "fp_edge", "label": "0", "toxicity_probability": "0.52"},
        {"sample_id": "fn_hi", "label": "1", "toxicity_probability": "0.01"},
        {"sample_id": "fn_edge", "label": "1", "toxicity_probability": "0.48"},
    ]
    selected = select_representative_cases(rows, threshold=0.5)
    assert len(selected) == 8
    assert {row["case_group"] for row in selected} == {"TP", "TN", "FP", "FN"}
    assert {row["selection_role"] for row in selected} == {
        "confidence_extreme",
        "threshold_boundary",
    }
