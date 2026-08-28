"""Central configuration: paths, column groups, and model settings.

Keeping these in one place means the rest of the pipeline never hard-codes
a file path or a column name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = DATA_DIR / "loan_applications.csv"
MODEL_PATH = MODEL_DIR / "credit_risk_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

# --- Schema ----------------------------------------------------------------

TARGET = "default"  # 1 = borrower defaulted, 0 = repaid

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_length",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "debt_to_income",
    "credit_score",
    "num_open_accounts",
    "num_past_delinquencies",
    "credit_history_length",
    "revolving_utilization",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "employment_status",
    "education",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Modelling -------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class DecisionPolicy:
    """How model probabilities are turned into a *recommendation*.

    The system never approves or denies on its own. It sorts applications
    into three lanes so a human reviewer knows where to spend their time.
    """

    approve_below: float = 0.20   # low predicted default risk
    decline_above: float = 0.55   # high predicted default risk
    # anything in between falls into MANUAL REVIEW

    def band(self, probability: float) -> str:
        if probability < self.approve_below:
            return "RECOMMEND_APPROVE"
        if probability > self.decline_above:
            return "RECOMMEND_DECLINE"
        return "MANUAL_REVIEW"


XGB_PARAMS = {
    # Tuned by cross-validated sweep on the training split. Shallow trees plus a
    # high min_child_weight keep the model from memorising noise: credit data has
    # a low signal-to-noise ratio and deep trees overfit it badly.
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
