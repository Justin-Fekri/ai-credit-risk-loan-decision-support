from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_artifacts_dir() -> Path:
    return Path(os.environ.get("CREDIT_RISK_ARTIFACTS", ROOT / "artifacts"))

NUMERIC_FEATURES = [
    "annual_income",
    "loan_amount",
    "debt_to_income",
    "credit_score",
    "employment_length_years",
    "credit_history_length_years",
    "revolving_utilization",
    "num_open_accounts",
    "num_delinquencies_24m",
    "num_inquiries_6m",
    "public_records",
    "monthly_debt",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "employment_status",
    "verification_status",
]

AUDIT_FEATURES = ["age"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_FEATURES = MODEL_FEATURES + AUDIT_FEATURES
TARGET = "defaulted"

FEATURE_LABELS = {
    "annual_income": "Annual income",
    "loan_amount": "Requested amount",
    "debt_to_income": "Debt-to-income ratio",
    "credit_score": "Credit score",
    "employment_length_years": "Years in role",
    "credit_history_length_years": "Credit history (years)",
    "revolving_utilization": "Revolving utilization",
    "num_open_accounts": "Open credit lines",
    "num_delinquencies_24m": "Delinquencies (24 mo)",
    "num_inquiries_6m": "Inquiries (6 mo)",
    "public_records": "Public records",
    "monthly_debt": "Monthly debt service",
    "age": "Age (audit only)",
    "home_ownership": "Housing",
    "loan_purpose": "Loan purpose",
    "employment_status": "Employment",
    "verification_status": "Income verification",
}

HOME_OWNERSHIP = ["MORTGAGE", "OWN", "RENT", "OTHER"]
LOAN_PURPOSE = [
    "debt_consolidation",
    "home_improvement",
    "major_purchase",
    "medical",
    "small_business",
    "car",
    "other",
]
EMPLOYMENT_STATUS = ["employed", "self_employed", "retired", "unemployed"]
VERIFICATION_STATUS = ["verified", "source_verified", "not_verified"]

PURPOSE_LABELS = {
    "debt_consolidation": "Debt consolidation",
    "home_improvement": "Home improvement",
    "major_purchase": "Major purchase",
    "medical": "Medical",
    "small_business": "Small business",
    "car": "Vehicle",
    "other": "Other",
}

RANDOM_STATE = 42
