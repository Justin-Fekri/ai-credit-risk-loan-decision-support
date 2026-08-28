"""Synthetic borrower generation with a known credit-risk data generating process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import (
    EMPLOYMENT_STATUS,
    HOME_OWNERSHIP,
    LOAN_PURPOSE,
    RANDOM_STATE,
    TARGET,
    VERIFICATION_STATUS,
)

FIRST_NAMES = [
    "Amara", "Bennett", "Camila", "Dev", "Elena", "Farid", "Grace", "Hiro",
    "Imani", "Jonah", "Keiko", "Leila", "Mateo", "Nadia", "Oscar", "Priya",
    "Quinn", "Rafael", "Sofia", "Tariq", "Uma", "Victor", "Willa", "Yara",
    "Zane", "Aisha", "Bruno", "Chloe", "Diego", "Eva",
]
LAST_NAMES = [
    "Okoye", "Hart", "Reyes", "Patel", "Voss", "Nakamura", "Brooks", "Khan",
    "Silva", "Nguyen", "Adler", "Moreau", "Singh", "Costa", "Berg", "Walsh",
    "Ibrahim", "Laurent", "Park", "Diaz", "Okafor", "Klein", "Rossi", "Chen",
]


def _clip(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def generate_borrowers(n: int = 12000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a tabular credit book with correlated features and a default label.

    The label is *not* a simple rule. It is a noisy logistic function of
    underwriting-relevant attributes so a tree model has something real to learn
    while remaining fully synthetic (no real consumer data).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 72, size=n)
    credit_history_length_years = _clip(
        (age - 18) * rng.uniform(0.35, 0.95, size=n) + rng.normal(0, 1.2, size=n),
        0.5,
        50,
    )

    employment_status = rng.choice(
        EMPLOYMENT_STATUS, size=n, p=[0.72, 0.14, 0.08, 0.06]
    )
    log_income = rng.normal(10.95, 0.48, size=n)
    log_income = np.where(employment_status == "unemployed", log_income - 0.85, log_income)
    log_income = np.where(employment_status == "retired", log_income - 0.25, log_income)
    log_income = np.where(employment_status == "self_employed", log_income + 0.12, log_income)
    annual_income = _clip(np.round(np.exp(log_income), -2), 18000, 320000)

    employment_length_years = _clip(
        rng.gamma(3.2, 1.8, size=n) - (employment_status == "unemployed") * 4,
        0,
        40,
    )
    employment_length_years = np.where(
        employment_status == "unemployed",
        rng.uniform(0, 1.5, size=n),
        employment_length_years,
    )

    home_ownership = rng.choice(
        HOME_OWNERSHIP, size=n, p=[0.42, 0.18, 0.36, 0.04]
    )
    # Older / higher-income borrowers more often own or mortgage.
    home_ownership = np.where(
        (age > 38) & (annual_income > 85000) & (rng.random(n) < 0.35),
        "MORTGAGE",
        home_ownership,
    )

    loan_purpose = rng.choice(
        LOAN_PURPOSE,
        size=n,
        p=[0.38, 0.14, 0.12, 0.08, 0.09, 0.11, 0.08],
    )
    loan_amount = _clip(
        np.round(annual_income * rng.uniform(0.08, 0.42, size=n), -2),
        2500,
        60000,
    )
    loan_amount = np.where(
        loan_purpose == "small_business",
        _clip(loan_amount * 1.25, 5000, 60000),
        loan_amount,
    )

    monthly_debt = _clip(
        np.round(annual_income / 12 * rng.uniform(0.08, 0.42, size=n), 0),
        150,
        9000,
    )
    proposed_payment = loan_amount / 36 * 1.08
    debt_to_income = _clip(
        (monthly_debt + proposed_payment) / (annual_income / 12), 0.04, 0.72
    )

    # Credit score: latent quality plus noise, then FICO-like scaling.
    latent = (
        0.45 * ((annual_income - 60000) / 40000)
        + 0.35 * ((credit_history_length_years - 10) / 8)
        - 0.55 * ((debt_to_income - 0.28) / 0.15)
        + rng.normal(0, 0.85, size=n)
    )
    credit_score = np.rint(_clip(680 + latent * 55, 520, 830)).astype(int)

    revolving_utilization = _clip(
        0.28
        + 0.35 * ((700 - credit_score) / 120)
        + rng.normal(0, 0.14, size=n),
        0.02,
        1.15,
    )
    num_open_accounts = _clip(
        rng.normal(7.5, 3.2, size=n) + (age > 40) * 1.4, 1, 24
    ).astype(int)
    num_delinquencies_24m = rng.poisson(
        _clip(0.15 + 1.8 * ((650 - credit_score) / 180), 0.02, 2.8), size=n
    )
    num_delinquencies_24m = _clip(num_delinquencies_24m, 0, 8).astype(int)
    num_inquiries_6m = rng.poisson(1.1 + (loan_purpose == "small_business") * 0.6, size=n)
    num_inquiries_6m = _clip(num_inquiries_6m, 0, 9).astype(int)
    public_records = rng.binomial(1, _clip(0.04 + (credit_score < 600) * 0.08, 0, 0.3), size=n)

    verification_status = rng.choice(
        VERIFICATION_STATUS, size=n, p=[0.46, 0.32, 0.22]
    )
    verification_status = np.where(
        (annual_income > 120000) & (rng.random(n) < 0.4),
        "verified",
        verification_status,
    )

    logit = (
        -2.35
        - 0.013 * (credit_score - 660)
        + 4.4 * np.maximum(0, debt_to_income - 0.36)
        + 2.8 * np.maximum(0, revolving_utilization - 0.55)
        + 0.68 * num_delinquencies_24m
        + 0.30 * num_inquiries_6m
        + 1.05 * public_records
        + 0.42 * (home_ownership == "RENT")
        + 0.78 * (loan_purpose == "small_business")
        + 0.45 * (loan_purpose == "medical")
        + 0.95 * (employment_status == "unemployed")
        + 0.24 * (employment_status == "self_employed")
        - 0.045 * np.minimum(employment_length_years, 12)
        - 0.0000024 * annual_income
        + 0.000013 * loan_amount
        + 0.30 * (verification_status == "not_verified")
        + rng.normal(0, 0.58, size=n)
    )
    pd_true = 1 / (1 + np.exp(-logit))
    defaulted = (rng.random(n) < pd_true).astype(int)

    first = rng.choice(FIRST_NAMES, size=n)
    last = rng.choice(LAST_NAMES, size=n)

    frame = pd.DataFrame(
        {
            "application_id": [f"NB-26-{10000 + i}" for i in range(n)],
            "borrower_name": [f"{f} {l}" for f, l in zip(first, last)],
            "age": age.astype(int),
            "annual_income": annual_income.astype(float),
            "employment_length_years": np.round(employment_length_years, 1),
            "employment_status": employment_status,
            "home_ownership": home_ownership,
            "loan_amount": loan_amount.astype(float),
            "loan_purpose": loan_purpose,
            "debt_to_income": np.round(debt_to_income, 3),
            "credit_score": credit_score,
            "credit_history_length_years": np.round(credit_history_length_years, 1),
            "revolving_utilization": np.round(revolving_utilization, 3),
            "num_open_accounts": num_open_accounts,
            "num_delinquencies_24m": num_delinquencies_24m,
            "num_inquiries_6m": num_inquiries_6m,
            "public_records": public_records.astype(int),
            "monthly_debt": monthly_debt.astype(float),
            "verification_status": verification_status,
            TARGET: defaulted,
        }
    )
    return frame
