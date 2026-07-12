"""Generate publication-ready evaluation figures from saved prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


COLORS = {
    "ink": "#17202A",
    "muted": "#667085",
    "test1": "#176B87",
    "test2": "#C44E52",
    "grid": "#D8DEE5",
    "paper": "#FFFFFF",
}


def _load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray([float(row["toxicity_probability"]) for row in rows], dtype=np.float64)
    return labels, probabilities


def _curve_data(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities)
    predictions = probabilities >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "confusion_matrix": matrix,
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "prevalence": float(labels.mean()),
        "num_samples": int(labels.size),
        "num_positive": int(labels.sum()),
        "num_negative": int((labels == 0).sum()),
    }


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["paper"])
    axis.grid(True, color=COLORS["grid"], linewidth=0.7, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(COLORS["muted"])
    axis.spines["bottom"].set_color(COLORS["muted"])
    axis.tick_params(colors=COLORS["ink"], labelsize=9)


def _write_curve_csv(output_path: Path, first: dict, second: dict, x_key: str, y_key: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", x_key, y_key])
        for name, values in (("test1", first), ("test2", second)):
            for x_value, y_value in zip(values[x_key], values[y_key]):
                writer.writerow([name, f"{float(x_value):.9g}", f"{float(y_value):.9g}"])


def generate_evaluation_figure(
    test1_predictions: str | Path,
    test2_predictions: str | Path,
    output_dir: str | Path,
    threshold: float = 0.85,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test1 = _curve_data(*_load_predictions(test1_predictions), threshold)
    test2 = _curve_data(*_load_predictions(test2_predictions), threshold)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "figure.facecolor": COLORS["paper"],
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), constrained_layout=True)

    for row, (name, values, color) in enumerate(
        (("Test1", test1, COLORS["test1"]), ("Test2", test2, COLORS["test2"]))
    ):
        roc_axis, pr_axis, cm_axis = axes[row]
        _style_axis(roc_axis)
        roc_axis.plot(values["fpr"], values["tpr"], color=color, linewidth=2.2)
        roc_axis.plot([0, 1], [0, 1], color=COLORS["muted"], linestyle="--", linewidth=1.0)
        roc_axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
        roc_axis.set_title(f"{name} ROC | AUROC {values['auroc']:.3f}", loc="left", fontsize=11)

        _style_axis(pr_axis)
        pr_axis.plot(values["recall"], values["precision"], color=color, linewidth=2.2)
        pr_axis.axhline(values["prevalence"], color=COLORS["muted"], linestyle="--", linewidth=1.0)
        pr_axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
        pr_axis.set_title(f"{name} PR | AUPRC {values['auprc']:.3f}", loc="left", fontsize=11)

        matrix = values["confusion_matrix"]
        cmap = LinearSegmentedColormap.from_list("ghxtox_cm", ["#F5F8FA", color])
        image = cm_axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max(int(matrix.max()), 1))
        cm_axis.grid(False)
        cm_axis.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
        cm_axis.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
        cm_axis.set_title(f"{name} confusion | threshold {threshold:.2f}", loc="left", fontsize=11)
        cutoff = matrix.max() * 0.52
        for y_index in range(2):
            for x_index in range(2):
                value = int(matrix[y_index, x_index])
                cm_axis.text(
                    x_index,
                    y_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color="white" if value > cutoff else COLORS["ink"],
                )
        figure.colorbar(image, ax=cm_axis, fraction=0.045, pad=0.04)

    figure.suptitle(
        "GHXTox default model evaluation",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    figure_path = output_dir / "default_model_evaluation.png"
    pdf_path = output_dir / "default_model_evaluation.pdf"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    _write_curve_csv(output_dir / "roc_curve_points.csv", test1, test2, "fpr", "tpr")
    _write_curve_csv(output_dir / "pr_curve_points.csv", test1, test2, "recall", "precision")
    summary = {
        "protocol": {
            "checkpoint": "runs/plm_fusion_esm2_geometry_confidence/best_model.pt",
            "threshold": float(threshold),
            "test1_predictions": str(test1_predictions),
            "test2_predictions": str(test2_predictions),
        },
        "test1": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in test1.items()
            if key not in {"fpr", "tpr", "roc_thresholds", "precision", "recall", "pr_thresholds"}
        },
        "test2": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in test2.items()
            if key not in {"fpr", "tpr", "roc_thresholds", "precision", "recall", "pr_thresholds"}
        },
    }
    (output_dir / "figure_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GHXTox ROC, PR, and confusion-matrix figures.")
    parser.add_argument("--test1-predictions", required=True)
    parser.add_argument("--test2-predictions", required=True)
    parser.add_argument("--output-dir", default="reports/figures")
    parser.add_argument("--threshold", type=float, default=0.85)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = generate_evaluation_figure(
        args.test1_predictions,
        args.test2_predictions,
        args.output_dir,
        args.threshold,
    )
    print(f"Figures saved to {args.output_dir}")
    for name in ("test1", "test2"):
        values = result[name]
        print(
            f"{name}: n={values['num_samples']} auroc={values['auroc']:.6f} "
            f"auprc={values['auprc']:.6f} confusion={values['confusion_matrix']}"
        )


if __name__ == "__main__":
    main()
