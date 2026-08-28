"""Model selection, calibration, threshold tuning, and persistence.

The training routine does four things a plain `fit()` does not:

1. compares candidate models by cross-validated ROC-AUC instead of trusting one
2. calibrates the winner so the output probability means "x% chance of default"
3. picks the operating threshold by *cost*, not by the 0.5 default
4. saves metadata alongside the model so any score can be traced back to the
   exact model and data that produced it
"""

from __future__ import annotations

import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import __version__
from .config import (
    CV_FOLDS,
    FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    DecisionPolicy,
)
from .data import load_dataset
from .evaluate import classification_metrics, cost_analysis
from .pipeline import AVAILABLE_MODELS, build_model


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Stratified split so both halves carry the same default rate."""
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def _imbalance_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    return float((len(y) - positives) / max(positives, 1))


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates=AVAILABLE_MODELS,
    folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """Cross-validated ROC-AUC for each candidate, best first.

    Models whose optional dependency is missing (e.g. xgboost without an OpenMP
    runtime) are skipped with a note rather than crashing the run.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    weight = _imbalance_weight(y_train)
    rows = []

    for name in candidates:
        try:
            pipeline = build_model(name, scale_pos_weight=weight)
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
            )
        except Exception as exc:
            # A missing optional dependency (xgboost, or its OpenMP runtime)
            # should degrade the run, not end it.
            print(f"  ! {name:22s} unavailable -- skipping ({type(exc).__name__})")
            continue
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
        )
        print(f"  - {name:22s} CV ROC-AUC {scores.mean():.4f} (+/- {scores.std():.4f})")

    if not rows:
        raise RuntimeError("No candidate model could be trained.")

    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(
        drop=True
    )


def calibrate(pipeline, X_train, y_train):
    """Wrap the pipeline so predicted probabilities match observed frequencies.

    Tree ensembles are well ranked but poorly calibrated out of the box. A
    reviewer reading "38% chance of default" needs that number to be real.
    """
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def tune_threshold(
    y_true,
    y_prob,
    cost_false_negative: float = 5.0,
    cost_false_positive: float = 1.0,
) -> tuple[float, dict]:
    """Sweep candidate cut-offs and keep the one with the lowest expected cost."""
    grid = np.linspace(0.05, 0.95, 91)
    best_threshold, best_cost, best_detail = 0.5, float("inf"), {}
    for threshold in grid:
        detail = cost_analysis(
            y_true, y_prob, threshold, cost_false_negative, cost_false_positive
        )
        if detail["total_relative_cost"] < best_cost:
            best_threshold, best_cost, best_detail = float(threshold), detail[
                "total_relative_cost"
            ], detail
    return best_threshold, best_detail


def train(
    data_path: Path | None = None,
    model_path: Path = MODEL_PATH,
    candidates=AVAILABLE_MODELS,
    cost_false_negative: float = 5.0,
    cost_false_positive: float = 1.0,
) -> dict:
    """Full training run. Returns a summary dict and writes the model bundle."""
    df = load_dataset(data_path) if data_path else load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"Rows: {len(df):,}  |  train {len(X_train):,}  test {len(X_test):,}")
    print(f"Default rate: {y_train.mean():.1%}\n")
    print("Cross-validating candidate models...")
    leaderboard = compare_models(X_train, y_train, candidates)
    winner = leaderboard.iloc[0]["model"]
    print(f"\nSelected model: {winner}")

    base = build_model(winner, scale_pos_weight=_imbalance_weight(y_train))
    base.fit(X_train, y_train)

    print("Calibrating probabilities...")
    model = calibrate(build_model(winner, scale_pos_weight=_imbalance_weight(y_train)),
                      X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold, cost_detail = tune_threshold(
        y_test, y_prob, cost_false_negative, cost_false_positive
    )
    metrics = classification_metrics(y_test, y_prob, threshold)

    print(
        f"\nTest ROC-AUC {metrics['roc_auc']:.4f}  |  PR-AUC {metrics['pr_auc']:.4f}"
        f"  |  Brier {metrics['brier_score']:.4f}"
    )
    print(
        f"Cost-optimal threshold {threshold:.2f}  ->  "
        f"recall {metrics['recall']:.3f}, precision {metrics['precision']:.3f}"
    )

    bundle = {
        "model": model,
        "uncalibrated_model": base,
        "model_name": winner,
        "threshold": threshold,
        "policy": DecisionPolicy(),
        "features": FEATURES,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "test_metrics": metrics,
        "cost_analysis": cost_detail,
        "training_metadata": {
            "package_version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rows": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "default_rate": round(float(df[TARGET].mean()), 4),
            "random_state": RANDOM_STATE,
            "data_path": str(data_path) if data_path else "default",
            "python": platform.python_version(),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nSaved model bundle -> {model_path}")

    return {
        "model_name": winner,
        "threshold": threshold,
        "metrics": metrics,
        "leaderboard": leaderboard,
        "policy": asdict(DecisionPolicy()),
    }


def load_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `credit-risk train` first."
        )
    return joblib.load(model_path)
