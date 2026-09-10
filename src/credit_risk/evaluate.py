"""Evaluation metrics and plots.

Accuracy alone is misleading on imbalanced credit data -- a model that
approves everyone can look "85% accurate". The metrics here focus on ranking
quality (ROC-AUC, PR-AUC) and on the business cost of the two error types.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import REPORT_DIR, DecisionPolicy


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def best_f1_threshold(y_true, y_prob) -> float:
    """Pick the probability cut-off that maximises F1 on the given data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    # `thresholds` is one shorter than precision/recall.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def cost_analysis(
    y_true,
    y_prob,
    threshold: float,
    cost_false_negative: float = 5.0,
    cost_false_positive: float = 1.0,
) -> dict:
    """Approving a defaulter costs far more than declining a good borrower.

    The default 5:1 ratio is a placeholder -- a real lender would supply its
    own loss-given-default and lost-margin figures.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = fn * cost_false_negative + fp * cost_false_positive
    return {
        "cost_false_negative": cost_false_negative,
        "cost_false_positive": cost_false_positive,
        "missed_defaults": int(fn),
        "rejected_good_borrowers": int(fp),
        "total_relative_cost": float(total),
    }


def review_workload(y_true, y_prob, policy: DecisionPolicy | None = None) -> dict:
    """How the three decision lanes split the portfolio, and how risky each is."""
    policy = policy or DecisionPolicy()
    bands = pd.Series([policy.band(p) for p in y_prob])
    truth = pd.Series(np.asarray(y_true))

    out = {}
    for band in ("RECOMMEND_APPROVE", "MANUAL_REVIEW", "RECOMMEND_DECLINE"):
        mask = bands == band
        n = int(mask.sum())
        out[band] = {
            "applications": n,
            "share_of_portfolio": round(n / len(bands), 4),
            "actual_default_rate": round(float(truth[mask].mean()), 4) if n else None,
        }
    return out


def save_metrics(metrics: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def plot_diagnostics(y_true, y_prob, out_dir: Path = REPORT_DIR) -> list[Path]:
    """ROC curve, precision-recall curve, and a probability calibration plot."""
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    from .plotting import ACCENT, INK, MUTED, despine, style

    out_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    base_rate = float(np.mean(y_true))
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    with style():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

        axes[0].plot(fpr, tpr, lw=2.2, color=ACCENT, label=f"Model (AUC {auc:.3f})")
        axes[0].plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, label="Random")
        axes[0].set(xlabel="False positive rate", ylabel="True positive rate",
                    title="ROC curve", xlim=(0, 1), ylim=(0, 1))
        axes[0].legend(loc="lower right")

        axes[1].plot(recall, precision, lw=2.2, color=ACCENT, label=f"Model (AP {ap:.3f})")
        axes[1].axhline(base_rate, ls="--", lw=1.2, color=MUTED,
                        label=f"Base rate ({base_rate:.1%})")
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title="Precision-recall curve", xlim=(0, 1), ylim=(0, 1))
        axes[1].legend(loc="upper right")

        axes[2].plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, label="Perfect calibration")
        axes[2].plot(prob_pred, prob_true, "o-", lw=2.2, color=ACCENT,
                     markersize=6, label="Model")
        top = max(prob_pred.max(), prob_true.max()) * 1.1
        axes[2].set(xlabel="Predicted probability", ylabel="Observed default rate",
                    title="Calibration", xlim=(0, top), ylim=(0, top))
        axes[2].legend(loc="upper left")

        for ax in axes:
            despine(ax)
            ax.title.set_color(INK)

        fig.tight_layout()
        path = out_dir / "diagnostics.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return [path]
