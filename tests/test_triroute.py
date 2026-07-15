import torch

from ghxtox.triroute import ConfidenceRouter, _calibration_metrics


def test_router_weights_sum_to_one_and_follow_reliability() -> None:
    router = ConfidenceRouter()
    confidence = torch.tensor([[0.80, 0.20, 0.15], [0.40, 0.75, 0.90]])
    weights = router.weights(confidence)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)
    assert weights[0, 0] > weights[0, 1]
    assert weights[0, 0] > weights[0, 2]
    assert weights[1, 2] > weights[1, 1]
    assert weights[1, 1] > weights[1, 0]


def test_calibration_metrics_are_zero_for_perfect_probabilities() -> None:
    probabilities = torch.tensor([0.0, 1.0, 0.0, 1.0])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    metrics = _calibration_metrics(probabilities, labels)
    assert metrics["brier"] == 0.0
    assert metrics["ece_10"] == 0.0
