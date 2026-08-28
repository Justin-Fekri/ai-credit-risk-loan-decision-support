"""Dataset generation and loading.

Public lending datasets (LendingClub, the UCI German Credit data) cannot be
redistributed inside this repository, so the project ships with a synthetic
generator instead. The generated data mimics the *structure* and the known
relationships of real consumer-credit data: default risk rises with
debt-to-income, utilisation and past delinquencies, and falls with credit
score, income and employment length.

Swap `generate_dataset` for a real CSV with the same columns and the rest of
the pipeline works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE, RAW_DATA_PATH, TARGET


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(scores: np.ndarray, target_rate: float) -> float:
    """Find the intercept that makes the average default probability hit target.

    Bisection on a monotonic function -- the mean of sigmoid(intercept + scores)
    rises with the intercept, so the search always converges. This keeps the
    portfolio default rate a tunable input instead of a magic constant.
    """
    low, high = -20.0, 20.0
    for _ in range(200):
        mid = (low + high) / 2
        if _sigmoid(mid + scores).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def generate_dataset(
    n_rows: int = 12_000,
    seed: int = RANDOM_STATE,
    target_default_rate: float = 0.18,
) -> pd.DataFrame:
    """Build a synthetic loan-application table.

    `target_default_rate` defaults to 18%, in the range reported for
    unsecured consumer instalment lending.
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n_rows)
    employment_length = np.clip(rng.gamma(shape=2.0, scale=3.0, size=n_rows), 0, 40)

    # Income correlates with age and employment length, with heavy right tail.
    annual_income = np.clip(
        rng.lognormal(mean=10.8, sigma=0.45, size=n_rows)
        + 900 * employment_length
        + 400 * (age - 21),
        18_000,
        400_000,
    )

    credit_history_length = np.clip(age - 18 - rng.exponential(3, n_rows), 0, 50)
    num_open_accounts = rng.poisson(5, n_rows) + 1
    num_past_delinquencies = rng.poisson(0.45, n_rows)
    revolving_utilization = np.clip(rng.beta(2, 4, n_rows) * 1.3, 0, 1.5)

    # Credit score driven by history, delinquencies and utilisation.
    credit_score = np.clip(
        720
        + 3.0 * credit_history_length
        - 38.0 * num_past_delinquencies
        - 90.0 * revolving_utilization
        + rng.normal(0, 35, n_rows),
        300,
        850,
    )

    loan_amount = np.clip(
        rng.lognormal(mean=9.4, sigma=0.6, size=n_rows), 1_000, 60_000
    )
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n_rows, p=[0.1, 0.2, 0.4, 0.2, 0.1])

    # Riskier borrowers are offered higher rates -- as in real underwriting.
    interest_rate = np.clip(
        22.0 - 0.018 * credit_score + 0.02 * loan_term_months + rng.normal(0, 1.5, n_rows),
        3.0,
        30.0,
    )

    other_debt = annual_income * rng.beta(2, 6, n_rows) * 0.8
    debt_to_income = np.clip((other_debt + loan_amount * 0.35) / annual_income, 0.01, 1.5)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"], n_rows, p=[0.45, 0.40, 0.15]
    )
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "car", "other"],
        n_rows,
        p=[0.35, 0.22, 0.14, 0.09, 0.10, 0.10],
    )
    employment_status = rng.choice(
        ["employed", "self_employed", "part_time", "unemployed"],
        n_rows,
        p=[0.68, 0.15, 0.12, 0.05],
    )
    education = rng.choice(
        ["high_school", "bachelors", "masters", "phd"], n_rows, p=[0.35, 0.42, 0.18, 0.05]
    )

    # --- Latent default risk ------------------------------------------------
    status_risk = pd.Series(employment_status).map(
        {"employed": 0.0, "self_employed": 0.25, "part_time": 0.45, "unemployed": 1.10}
    ).to_numpy()
    home_risk = pd.Series(home_ownership).map(
        {"OWN": -0.30, "MORTGAGE": -0.15, "RENT": 0.15}
    ).to_numpy()
    purpose_risk = pd.Series(loan_purpose).map(
        {
            "debt_consolidation": 0.20,
            "credit_card": 0.15,
            "medical": 0.25,
            "home_improvement": -0.10,
            "car": -0.05,
            "other": 0.05,
        }
    ).to_numpy()

    risk_score = (
        -0.0085 * (credit_score - 650)
        + 2.10 * debt_to_income
        + 1.25 * revolving_utilization
        + 0.38 * num_past_delinquencies
        - 0.045 * employment_length
        - 0.55 * np.log(annual_income / 50_000)
        + 0.055 * interest_rate
        + 0.004 * (loan_amount / 1_000)
        + status_risk
        + home_risk
        + purpose_risk
        # --- non-linear effects, as seen in real underwriting ---------------
        # The subprime cliff: risk accelerates below roughly 620.
        + 0.9 * np.maximum(0, (620 - credit_score) / 100) ** 2
        # Thin-file penalty that flattens out once history is established.
        + 0.6 * np.exp(-credit_history_length / 4)
        # Stretched borrowers in unstable work are worse than either alone.
        + 1.1 * (debt_to_income > 0.45) * (employment_length < 2)
        # High utilisation compounds prior delinquencies.
        + 0.30 * revolving_utilization * num_past_delinquencies
        # Very large loans relative to income are disproportionately risky.
        + 0.8 * np.maximum(0, loan_amount / annual_income - 0.4)
        # Age risk is U-shaped, not linear: young thin-file borrowers and
        # retirement-age borrowers on fixed incomes both carry more risk.
        + 0.5 * ((age - 45) / 18.0) ** 2
        # Segment-specific effects a linear model in raw features cannot express.
        + 0.9 * (employment_status == "self_employed") * (debt_to_income > 0.40)
        + 0.7 * (home_ownership == "RENT") * (credit_history_length < 4)
        + rng.normal(0, 0.45, n_rows)  # irreducible noise -- no perfect model
    )
    # Shift the whole risk distribution so the portfolio hits the target rate.
    logit = _solve_intercept(risk_score, target_default_rate) + risk_score
    default = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "application_id": [f"APP{i:06d}" for i in range(1, n_rows + 1)],
            "age": age,
            "annual_income": annual_income.round(2),
            "employment_length": employment_length.round(1),
            "loan_amount": loan_amount.round(2),
            "loan_term_months": loan_term_months,
            "interest_rate": interest_rate.round(2),
            "debt_to_income": debt_to_income.round(4),
            "credit_score": credit_score.round(0).astype(int),
            "num_open_accounts": num_open_accounts,
            "num_past_delinquencies": num_past_delinquencies,
            "credit_history_length": credit_history_length.round(1),
            "revolving_utilization": revolving_utilization.round(4),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "employment_status": employment_status,
            "education": education,
            TARGET: default,
        }
    )

    # Real applications have gaps: knock out a few values so the imputation
    # step in the pipeline is doing genuine work.
    for col, frac in (("employment_length", 0.03), ("revolving_utilization", 0.02)):
        missing = rng.random(n_rows) < frac
        df.loc[missing, col] = np.nan

    return df


def save_dataset(df: pd.DataFrame, path: Path = RAW_DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the dataset, generating it on first run if it does not exist."""
    if not Path(path).exists():
        save_dataset(generate_dataset(), path)
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {path} is missing columns: {missing}")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("string")
    return df
