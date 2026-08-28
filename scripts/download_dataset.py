"""Fetch a real public credit dataset and map it onto this project's schema.

The default pipeline runs on synthetic data so the repository works offline.
This script provides the honest alternative: the UCI Statlog (German Credit)
dataset -- 1,000 real loan applications, publicly redistributable, no API key.

    python scripts/download_dataset.py --output data/german_credit.csv
    credit-risk train --data data/german_credit.csv

The mapping is necessarily approximate: the German Credit data has different
fields from a modern US credit file, so several columns are derived or filled
with neutral values. Those choices are listed in DERIVATION_NOTES and printed
when the script runs, because a reviewer should know exactly which numbers are
real and which are constructed.
"""

from __future__ import annotations

import argparse
import io
import ssl
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# The .data file is space-separated with no header; these are the documented names.
RAW_COLUMNS = [
    "checking_account", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits",
    "job", "dependents", "telephone", "foreign_worker", "target",
]

PURPOSE_MAP = {
    "A40": "car", "A41": "car", "A42": "home_improvement", "A43": "other",
    "A44": "home_improvement", "A45": "home_improvement", "A46": "other",
    "A48": "other", "A49": "other", "A410": "other",
}
HOUSING_MAP = {"A151": "RENT", "A152": "OWN", "A153": "OWN"}
EMPLOYMENT_YEARS = {"A71": 0.0, "A72": 0.5, "A73": 2.5, "A74": 5.5, "A75": 10.0}
JOB_STATUS = {
    "A171": "unemployed", "A172": "part_time",
    "A173": "employed", "A174": "self_employed",
}
JOB_EDUCATION = {
    "A171": "high_school", "A172": "high_school",
    "A173": "bachelors", "A174": "masters",
}
# Credit history codes, worst to best, converted to a delinquency count.
HISTORY_DELINQUENCIES = {"A30": 3, "A31": 2, "A32": 1, "A33": 2, "A34": 3}

DERIVATION_NOTES = [
    "annual_income is NOT in the source data; it is estimated from the "
    "installment rate and credit amount, and should be treated as derived.",
    "credit_score is NOT in the source data; it is constructed from credit "
    "history, savings and checking status onto a 300-850 scale.",
    "interest_rate is not recorded; a flat rate by duration is imputed.",
    "revolving_utilization is approximated from existing credit lines.",
    "The German Credit target is 1=good, 2=bad; it is recoded to 0/1.",
]


def _ssl_context() -> ssl.SSLContext:
    """Use certifi's CA bundle when present.

    A stock python.org install on macOS ships without root certificates, so a
    plain urlopen fails with CERTIFICATE_VERIFY_FAILED even though the URL is
    fine. certifi supplies the bundle; without it we fall back to the system
    store rather than disabling verification.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(url: str = URL) -> pd.DataFrame:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    return pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=RAW_COLUMNS)


def to_project_schema(raw: pd.DataFrame) -> pd.DataFrame:
    n = len(raw)

    # The source records an installment rate as a % of disposable income, which
    # lets us back out a rough income figure.
    monthly_payment = raw["credit_amount"] / raw["duration_months"]
    disposable_monthly = monthly_payment / (raw["installment_rate"] / 100.0)
    annual_income = (disposable_monthly * 12 * 1.6).clip(15_000, 300_000)

    savings_bonus = raw["savings"].map(
        {"A61": 0, "A62": 20, "A63": 40, "A64": 60, "A65": 10}
    ).fillna(0)
    checking_bonus = raw["checking_account"].map(
        {"A11": -40, "A12": 0, "A13": 30, "A14": 50}
    ).fillna(0)
    history_bonus = raw["credit_history"].map(
        {"A30": -60, "A31": -40, "A32": 20, "A33": -20, "A34": 40}
    ).fillna(0)
    credit_score = (640 + savings_bonus + checking_bonus + history_bonus).clip(300, 850)

    duration = raw["duration_months"]
    out = pd.DataFrame(
        {
            "application_id": [f"GC{i:05d}" for i in range(1, n + 1)],
            "age": raw["age"],
            "annual_income": annual_income.round(2),
            "employment_length": raw["employment_since"].map(EMPLOYMENT_YEARS).fillna(2.0),
            "loan_amount": raw["credit_amount"].round(2),
            "loan_term_months": duration,
            "interest_rate": (6.0 + 0.15 * duration).clip(3, 30).round(2),
            "debt_to_income": (raw["credit_amount"] / annual_income).clip(0.01, 1.5).round(4),
            "credit_score": credit_score.astype(int),
            "num_open_accounts": raw["existing_credits"],
            "num_past_delinquencies": raw["credit_history"].map(HISTORY_DELINQUENCIES).fillna(1),
            "credit_history_length": (raw["age"] - 18).clip(0, 50),
            "revolving_utilization": (raw["existing_credits"] / 4.0).clip(0, 1.5).round(4),
            "home_ownership": raw["housing"].map(HOUSING_MAP).fillna("RENT"),
            "loan_purpose": raw["purpose"].map(PURPOSE_MAP).fillna("other"),
            "employment_status": raw["job"].map(JOB_STATUS).fillna("employed"),
            "education": raw["job"].map(JOB_EDUCATION).fillna("high_school"),
            "default": (raw["target"] == 2).astype(int),
        }
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/german_credit.csv")
    args = parser.parse_args()

    try:
        raw = download()
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print("The synthetic dataset still works: `credit-risk generate-data`.", file=sys.stderr)
        return 1

    df = to_project_schema(raw)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"\nWrote {len(df):,} real applications -> {path}")
    print(f"Default rate: {df['default'].mean():.1%}")
    print("\nIMPORTANT -- how this mapping was derived:")
    for note in DERIVATION_NOTES:
        print(f"  * {note}")
    print(f"\nTrain on it with:\n  credit-risk train --data {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
