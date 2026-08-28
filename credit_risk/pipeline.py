from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credit_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET


def build_preprocessor() -> ColumnTransformer:
    """Trees do not need scaling; only expand categoricals for XGBoost."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_estimator(scale_pos_weight: float, random_state: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=4,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=4,
        random_state=random_state,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def build_pipeline(scale_pos_weight: float, random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", build_estimator(scale_pos_weight, random_state)),
        ]
    )


@dataclass
class FittedBundle:
    pipeline: Pipeline
    feature_names_in: list[str]
    transformed_feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def fit_bundle(frame: pd.DataFrame, random_state: int = 42) -> FittedBundle:
    X = frame[MODEL_FEATURES]
    y = frame[TARGET].to_numpy()
    pos = max(int(y.sum()), 1)
    neg = max(int(len(y) - pos), 1)
    pipe = build_pipeline(scale_pos_weight=neg / pos, random_state=random_state)
    pipe.fit(X, y)
    transformed = pipe.named_steps["preprocess"].get_feature_names_out()
    return FittedBundle(
        pipeline=pipe,
        feature_names_in=list(MODEL_FEATURES),
        transformed_feature_names=list(transformed),
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
    )


def predict_pd(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(frame[MODEL_FEATURES])[:, 1]
