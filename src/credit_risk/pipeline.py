"""Model definitions: preprocessing + estimator, wrapped in one sklearn Pipeline.

Everything lives inside a single `Pipeline` so that preprocessing is fitted on
training folds only. That is what keeps cross-validation scores honest.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, XGB_PARAMS


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Impute missing values, one-hot encode categoricals, optionally scale.

    Tree models do not need scaling; logistic regression does.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(name: str, scale_pos_weight: float = 1.0) -> Pipeline:
    """Return an unfitted pipeline for one of the supported model names."""
    name = name.lower()

    if name == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )
        pre = build_preprocessor(scale_numeric=True)

    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        pre = build_preprocessor()

    elif name == "xgboost":
        from xgboost import XGBClassifier  # imported lazily: optional dependency

        estimator = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
        pre = build_preprocessor()

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Choose from: logistic_regression, random_forest, xgboost."
        )

    return Pipeline([("preprocess", pre), ("model", estimator)])


AVAILABLE_MODELS = ("logistic_regression", "random_forest", "xgboost")
