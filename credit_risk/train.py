from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from credit_risk.config import ALL_FEATURES, RANDOM_STATE, TARGET, get_artifacts_dir
from credit_risk.data import generate_borrowers
from credit_risk.explain import CreditExplainer
from credit_risk.pipeline import FittedBundle, fit_bundle, predict_pd
from credit_risk.policy import eligibility_label, recommend, risk_band


def _ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = y_true[order]
    n_pos = max(y.sum(), 1)
    n_neg = max(len(y) - y.sum(), 1)
    cdf_pos = np.cumsum(y) / n_pos
    cdf_neg = np.cumsum(1 - y) / n_neg
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def evaluate(y_true: np.ndarray, pd_hat: np.ndarray, threshold: float = 0.5) -> dict:
    y_hat = (pd_hat >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_true, pd_hat)
    roc_points = [
        {"fpr": round(float(a), 4), "tpr": round(float(b), 4)}
        for a, b in zip(fpr[:: max(len(fpr) // 40, 1)], tpr[:: max(len(tpr) // 40, 1)])
    ]
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    auc = float(roc_auc_score(y_true, pd_hat))
    return {
        "roc_auc": round(auc, 4),
        "gini": round(2 * auc - 1, 4),
        "pr_auc": round(float(average_precision_score(y_true, pd_hat)), 4),
        "brier": round(float(brier_score_loss(y_true, pd_hat)), 4),
        "ks": round(_ks_statistic(y_true, pd_hat), 4),
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "default_rate": round(float(y_true.mean()), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": roc_points,
        "n_test": int(len(y_true)),
    }


def _select_queue(test: pd.DataFrame, pd_hat: np.ndarray, n: int = 18) -> pd.DataFrame:
    scored = test.copy()
    scored["pd_default"] = pd_hat
    scored["risk_band"] = scored["pd_default"].map(risk_band)
    parts = []
    for band, count in [("low", 5), ("moderate", 5), ("elevated", 5), ("severe", 3)]:
        pool = scored[scored["risk_band"] == band]
        if pool.empty:
            continue
        take = min(count, len(pool))
        parts.append(pool.sample(take, random_state=RANDOM_STATE))
    queue = pd.concat(parts).head(n)
    return queue.sort_values("pd_default", ascending=False)


def serialize_application(row: pd.Series, shap_rows: list[dict] | None = None) -> dict:
    rec, rationale = recommend(
        float(row["pd_default"]), float(row["credit_score"]), float(row["debt_to_income"])
    )
    payload = {
        "application_id": row["application_id"],
        "borrower_name": row["borrower_name"],
        "features": {name: _jsonable(row[name]) for name in ALL_FEATURES},
        "pd_default": round(float(row["pd_default"]), 4),
        "risk_band": risk_band(float(row["pd_default"])),
        "policy_recommendation": rec,
        "policy_rationale": rationale,
        "eligibility": eligibility_label(rec),
        "status": "pending",
        "decision": None,
        "decision_notes": None,
    }
    if shap_rows is not None:
        payload["shap"] = shap_rows
        payload["top_drivers"] = shap_rows[:5]
    return payload


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def train(
    n_samples: int = 12000,
    output_dir: Path | None = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    artifacts_dir = Path(output_dir or get_artifacts_dir())
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = generate_borrowers(n_samples, seed=random_state)
    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data[TARGET],
        random_state=random_state,
    )
    bundle = fit_bundle(train_df, random_state=random_state)
    pd_test = predict_pd(bundle.pipeline, test_df)
    metrics = evaluate(test_df[TARGET].to_numpy(), pd_test)

    background = train_df.sample(min(100, len(train_df)), random_state=random_state)
    explainer = CreditExplainer(bundle, background)
    importance = explainer.global_importance(
        test_df.sample(min(400, len(test_df)), random_state=random_state)
    )

    queue_df = _select_queue(test_df, pd_test)
    queue_shap = explainer.explain_frame(queue_df)
    queue = [
        serialize_application(row, shap_rows)
        for (_, row), shap_rows in zip(queue_df.iterrows(), queue_shap)
    ]

    joblib.dump(
        {
            "pipeline": bundle.pipeline,
            "feature_names_in": bundle.feature_names_in,
            "transformed_feature_names": bundle.transformed_feature_names,
            "numeric_features": bundle.numeric_features,
            "categorical_features": bundle.categorical_features,
        },
        artifacts_dir / "model.joblib",
    )
    background[ALL_FEATURES].to_csv(artifacts_dir / "shap_background.csv", index=False)
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifacts_dir / "global_importance.json").write_text(
        json.dumps(importance, indent=2)
    )
    (artifacts_dir / "queue.json").write_text(json.dumps(queue, indent=2))
    (artifacts_dir / "model_card.json").write_text(
        json.dumps(
            {
                "name": "Northbridge default-risk XGBoost",
                "task": "binary classification — 12-month default",
                "algorithm": "XGBoost (hist) inside a scikit-learn ColumnTransformer pipeline",
                "explainability": "TreeSHAP, interventional, probability output, grouped to original features",
                "training_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "positive_rate_train": round(float(train_df[TARGET].mean()), 4),
                "decision_support": True,
                "automated_decisioning": False,
            },
            indent=2,
        )
    )
    return metrics


def load_bundle(output_dir: Path | None = None) -> FittedBundle:
    artifacts_dir = Path(output_dir or get_artifacts_dir())
    payload = joblib.load(artifacts_dir / "model.joblib")
    return FittedBundle(
        pipeline=payload["pipeline"],
        feature_names_in=payload["feature_names_in"],
        transformed_feature_names=payload["transformed_feature_names"],
        numeric_features=payload["numeric_features"],
        categorical_features=payload["categorical_features"],
    )


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
