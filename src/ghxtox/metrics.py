"""Small metric helpers with an optional scikit-learn fast path."""

from __future__ import annotations

import math

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    labels = labels.detach().cpu().float()
    probs = torch.sigmoid(logits.detach().cpu().float())
    preds = (probs >= threshold).float()

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    total = max(tp + tn + fp + fn, 1.0)

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    balanced_acc = 0.5 * (recall + specificity)
    mcc_den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1.0))
    metrics = {
        "accuracy": (tp + tn) / total,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": ((tp * tn) - (fp * fn)) / mcc_den,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = labels.numpy()
        y_score = probs.numpy()
        if len(set(y_true.tolist())) > 1:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
            metrics["auprc"] = float(average_precision_score(y_true, y_score))
        else:
            metrics["auroc"] = float("nan")
            metrics["auprc"] = float("nan")
    except Exception:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics
