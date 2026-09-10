"""Explainability: turn a probability into reasons a human can check.

Two layers:
  * global  -- which features drive the model overall (permutation importance,
               plus SHAP summary values when the `shap` package is installed)
  * local   -- for one applicant, the features that pushed their risk up or
               down, rendered as plain-English reason codes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import FEATURES, REPORT_DIR

# Human-readable labels for the reason codes shown to a reviewer.
FEATURE_LABELS = {
    "age": "age",
    "annual_income": "annual income",
    "employment_length": "years employed",
    "loan_amount": "requested loan amount",
    "loan_term_months": "loan term",
    "interest_rate": "offered interest rate",
    "debt_to_income": "debt-to-income ratio",
    "credit_score": "credit score",
    "num_open_accounts": "number of open accounts",
    "num_past_delinquencies": "past delinquencies",
    "credit_history_length": "length of credit history",
    "revolving_utilization": "revolving credit utilisation",
    "home_ownership": "home ownership",
    "loan_purpose": "loan purpose",
    "employment_status": "employment status",
    "education": "education level",
}


# Building a SHAP explainer is far more expensive than using one, so the
# explainer for a given fitted model is built once and reused.
_EXPLAINER_CACHE: dict[int, object] = {}


def _tree_explainer(model):
    """A TreeExplainer on the fast path, cached per fitted model.

    `feature_perturbation="tree_path_dependent"` walks the trees directly and
    needs no background dataset. The default interventional path is exact but
    orders of magnitude slower -- on a 400-tree forest it turned a 0.03s
    explanation into a 17s one, which would be unusable in a live review queue.
    """
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        import shap

        _EXPLAINER_CACHE[key] = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent"
        )
    return _EXPLAINER_CACHE[key]


def _base_feature(encoded_name: str) -> str:
    """Map a one-hot column such as `loan_purpose_medical` back to its source."""
    for feature in sorted(FEATURES, key=len, reverse=True):
        if encoded_name == feature or encoded_name.startswith(feature + "_"):
            return feature
    return encoded_name


def global_importance(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10
) -> pd.DataFrame:
    """Permutation importance on the held-out set, aggregated to raw features.

    Permutation importance is model-agnostic and measures what the model
    actually relies on, rather than how often a feature was split on.
    """
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=0, scoring="roc_auc", n_jobs=-1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_values(pipeline: Pipeline, X: pd.DataFrame, max_samples: int = 500):
    """SHAP values in the *encoded* feature space, or None if shap is absent."""
    try:
        import shap
    except ImportError:
        return None, None

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    sample = X.sample(min(max_samples, len(X)), random_state=0)
    encoded = pd.DataFrame(pre.transform(sample), columns=pre.get_feature_names_out())

    try:
        values = _tree_explainer(model).shap_values(encoded)
    except Exception:
        try:
            # Linear models have an exact, fast explainer of their own.
            explainer = shap.LinearExplainer(model, encoded)
            values = explainer.shap_values(encoded)
        except Exception:
            # Last resort: model-agnostic, but expensive -- keep the sample small.
            subset = encoded.iloc[: min(60, len(encoded))]
            background = shap.kmeans(encoded, min(25, len(encoded)))
            explainer = shap.KernelExplainer(
                lambda d: model.predict_proba(d)[:, 1], background
            )
            values = explainer.shap_values(subset, silent=True)
            encoded = subset

    values = np.asarray(values)
    if values.ndim == 3:  # some versions return one matrix per class
        values = values[..., 1]
    return values, encoded


def aggregate_shap(values: np.ndarray, encoded: pd.DataFrame) -> pd.DataFrame:
    """Collapse one-hot SHAP columns back onto the original feature names."""
    frame = pd.DataFrame(np.abs(values), columns=encoded.columns)
    frame.columns = [_base_feature(c) for c in frame.columns]
    return (
        frame.T.groupby(level=0).sum().T.mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )


def plot_global_importance(importance: pd.DataFrame, out_dir: Path = REPORT_DIR) -> Path:
    import matplotlib.pyplot as plt

    from .plotting import ACCENT, ACCENT_LIGHT, INK, despine, style

    out_dir.mkdir(parents=True, exist_ok=True)
    top = importance.head(15).iloc[::-1]
    value_col = "importance" if "importance" in top.columns else "mean_abs_shap"
    labels = [FEATURE_LABELS.get(f, f) for f in top["feature"]]
    values = top[value_col].to_numpy()

    with style():
        fig, ax = plt.subplots(figsize=(9, 6))
        # Fade the long tail so the eye lands on the drivers that matter.
        cutoff = values.max() * 0.15
        colours = [ACCENT if v >= cutoff else ACCENT_LIGHT for v in values]
        bars = ax.barh(labels, values, color=colours, height=0.72)

        for bar, value in zip(bars, values, strict=True):
            ax.text(value + values.max() * 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{value:.4f}", va="center", ha="left", fontsize=8.5, color=INK)

        ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
        ax.set_title("Most influential borrower features", color=INK)
        ax.set_xlim(0, values.max() * 1.18)
        ax.grid(axis="y", visible=False)
        despine(ax)

        fig.tight_layout()
        path = out_dir / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return path


def explain_application(
    pipeline: Pipeline,
    application: pd.DataFrame,
    background: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Reason codes for a single application, most influential factor first.

    Each entry says which feature moved the risk, in which direction, and by
    how much -- so a reviewer can sanity-check the recommendation instead of
    trusting a bare number.
    """
    if len(application) != 1:
        raise ValueError("explain_application expects exactly one row")

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    encoded_names = list(pre.get_feature_names_out())
    encoded_row = pre.transform(application)

    contributions: dict[str, float] = {}
    try:
        values = np.asarray(_tree_explainer(model).shap_values(encoded_row))
        if values.ndim == 3:
            values = values[..., 1]
        row = values.reshape(-1)
        for name, value in zip(encoded_names, row, strict=False):
            base = _base_feature(name)
            contributions[base] = contributions.get(base, 0.0) + float(value)
        method = "shap"
    except Exception:
        # Fallback: how much does the prediction move if this feature is set to
        # the population median / mode? Slower conceptually, but always works.
        baseline = float(pipeline.predict_proba(application)[0, 1])
        for feature in application.columns:
            if feature not in FEATURES:
                continue
            perturbed = application.copy()
            column = background[feature]
            perturbed[feature] = (
                column.median() if pd.api.types.is_numeric_dtype(column)
                else column.mode().iloc[0]
            )
            contributions[feature] = baseline - float(
                pipeline.predict_proba(perturbed)[0, 1]
            )
        method = "counterfactual_shift"

    ordered = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    reasons = []
    for feature, value in ordered[:top_n]:
        raw = application.iloc[0].get(feature)
        reasons.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": None if pd.isna(raw) else (raw.item() if hasattr(raw, "item") else raw),
                "direction": "increases risk" if value > 0 else "decreases risk",
                "contribution": round(float(value), 4),
                "method": method,
            }
        )
    return reasons
