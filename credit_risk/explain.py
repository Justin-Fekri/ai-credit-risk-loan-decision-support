from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import shap


from credit_risk.config import FEATURE_LABELS, MODEL_FEATURES, PURPOSE_LABELS
from credit_risk.pipeline import FittedBundle


def original_feature_name(transformed_name: str) -> str:
    _, rest = transformed_name.split("__", 1)
    for feature in MODEL_FEATURES:
        if rest == feature or rest.startswith(f"{feature}_"):
            return feature
    return rest


def group_shap(
    shap_row: np.ndarray, transformed_names: list[str]
) -> dict[str, float]:
    grouped: dict[str, float] = defaultdict(float)
    for value, name in zip(shap_row, transformed_names):
        grouped[original_feature_name(name)] += float(value)
    return dict(grouped)


def format_feature_value(feature: str, value) -> str:
    if feature == "annual_income" or feature == "loan_amount" or feature == "monthly_debt":
        return f"${float(value):,.0f}"
    if feature == "debt_to_income" or feature == "revolving_utilization":
        return f"{float(value) * 100:.1f}%"
    if feature == "loan_purpose":
        return PURPOSE_LABELS.get(str(value), str(value).replace("_", " "))
    if feature in {"employment_length_years", "credit_history_length_years"}:
        return f"{float(value):.1f} yrs"
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        return f"{float(value):.2f}"
    return str(value).replace("_", " ")


class CreditExplainer:
    """Tree SHAP over the fitted XGBoost, rolled back to original borrower fields."""

    def __init__(self, bundle: FittedBundle, background: pd.DataFrame):
        self.bundle = bundle
        self.preprocess = bundle.pipeline.named_steps["preprocess"]
        self.model = bundle.pipeline.named_steps["model"]
        background_t = self.preprocess.transform(background[MODEL_FEATURES])
        self.explainer = shap.TreeExplainer(
            self.model,
            data=background_t,
            feature_perturbation="interventional",
            model_output="probability",
        )
        self.expected_value = float(np.ravel(self.explainer.expected_value)[-1])

    def explain_frame(self, frame: pd.DataFrame) -> list[list[dict]]:
        transformed = self.preprocess.transform(frame[MODEL_FEATURES])
        shap_values = self.explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        rows: list[list[dict]] = []
        names = self.bundle.transformed_feature_names
        for i, (_, record) in enumerate(frame.iterrows()):
            grouped = group_shap(shap_values[i], names)
            contributions = []
            for feature, shap_value in grouped.items():
                raw = record[feature]
                contributions.append(
                    {
                        "feature": feature,
                        "label": FEATURE_LABELS.get(feature, feature),
                        "value": format_feature_value(feature, raw),
                        "raw_value": _jsonable(raw),
                        "shap": round(float(shap_value), 6),
                        "direction": "raises_risk" if shap_value > 0 else "lowers_risk",
                    }
                )
            contributions.sort(key=lambda item: abs(item["shap"]), reverse=True)
            rows.append(contributions)
        return rows

    def global_importance(self, frame: pd.DataFrame) -> list[dict]:
        explanations = self.explain_frame(frame)
        acc: dict[str, list[float]] = defaultdict(list)
        for row in explanations:
            for item in row:
                acc[item["feature"]].append(abs(item["shap"]))
        ranked = []
        for feature, values in acc.items():
            ranked.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(feature, feature),
                    "mean_abs_shap": round(float(np.mean(values)), 6),
                }
            )
        ranked.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
        return ranked


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value
