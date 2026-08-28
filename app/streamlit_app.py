"""Streamlit front-end: enter a borrower's details, see the risk assessment.

Run with:  streamlit run app/streamlit_app.py

This is the human-in-the-loop surface. It deliberately shows the reasoning and
the uncertainty next to the number, and never renders an "Approved" /
"Rejected" verdict -- only a recommendation for a reviewer to act on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.config import FEATURES, MODEL_PATH  # noqa: E402
from credit_risk.data import load_dataset  # noqa: E402
from credit_risk.predict import score_application  # noqa: E402
from credit_risk.train import load_bundle  # noqa: E402

st.set_page_config(page_title="Credit Risk Decision Support", page_icon="*", layout="wide")

BAND_STYLE = {
    "RECOMMEND_APPROVE": ("Recommend approve", "#1a7f37"),
    "MANUAL_REVIEW": ("Manual review required", "#9a6700"),
    "RECOMMEND_DECLINE": ("Recommend decline", "#b42318"),
}


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


@st.cache_data
def get_background():
    return load_dataset()[FEATURES]


st.title("Credit Risk & Loan Decision Support")
st.caption(
    "Decision support for a human underwriter. This tool estimates risk and "
    "explains its reasoning; it does not approve or decline anything."
)

try:
    bundle = get_bundle()
    background = get_background()
except FileNotFoundError:
    st.error("No trained model found. Run `credit-risk train` first, then reload.")
    st.stop()

with st.sidebar:
    st.header("Applicant details")
    st.subheader("Credit profile")
    credit_score = st.slider("Credit score", 300, 850, 690)
    num_past_delinquencies = st.slider("Past delinquencies", 0, 10, 0)
    credit_history_length = st.slider("Credit history (years)", 0.0, 40.0, 10.0, 0.5)
    revolving_utilization = st.slider("Revolving utilisation", 0.0, 1.5, 0.30, 0.01)
    num_open_accounts = st.slider("Open accounts", 1, 20, 5)

    st.subheader("Finances")
    annual_income = st.number_input("Annual income ($)", 15_000, 400_000, 65_000, 1_000)
    debt_to_income = st.slider("Debt-to-income ratio", 0.01, 1.5, 0.32, 0.01)
    employment_length = st.slider("Years employed", 0.0, 40.0, 5.0, 0.5)
    age = st.slider("Age", 18, 80, 38)

    st.subheader("Requested loan")
    loan_amount = st.number_input("Loan amount ($)", 1_000, 60_000, 15_000, 500)
    loan_term_months = st.selectbox("Term (months)", [12, 24, 36, 48, 60], index=2)
    interest_rate = st.slider("Interest rate (%)", 3.0, 30.0, 13.0, 0.1)

    st.subheader("Profile")
    home_ownership = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    employment_status = st.selectbox(
        "Employment status", ["employed", "self_employed", "part_time", "unemployed"]
    )
    loan_purpose = st.selectbox(
        "Loan purpose",
        ["debt_consolidation", "credit_card", "home_improvement", "medical", "car", "other"],
    )
    education = st.selectbox("Education", ["high_school", "bachelors", "masters", "phd"])

application = {
    "age": age,
    "annual_income": annual_income,
    "employment_length": employment_length,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "interest_rate": interest_rate,
    "debt_to_income": debt_to_income,
    "credit_score": credit_score,
    "num_open_accounts": num_open_accounts,
    "num_past_delinquencies": num_past_delinquencies,
    "credit_history_length": credit_history_length,
    "revolving_utilization": revolving_utilization,
    "home_ownership": home_ownership,
    "loan_purpose": loan_purpose,
    "employment_status": employment_status,
    "education": education,
}

result = score_application(bundle, application, background=background, top_n=6)
label, colour = BAND_STYLE[result["recommendation"]]

left, right = st.columns([1, 1.3])

with left:
    st.metric("Estimated default probability", f"{result['default_probability']:.1%}")
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:8px;background:{colour};"
        f"color:white;font-weight:600;text-align:center'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.write(f"**Risk band:** {result['risk_band']}")
    st.write(f"**Model:** {result['model_name']}")
    st.progress(min(result["default_probability"], 1.0))

with right:
    st.subheader("Why the model scored it this way")
    reasons = [r for r in result["reasons"] if "label" in r]
    if reasons:
        table = pd.DataFrame(
            {
                "Factor": [r["label"] for r in reasons],
                "Value": [r["value"] for r in reasons],
                "Effect": [
                    "increases risk" if r["contribution"] > 0 else "decreases risk"
                    for r in reasons
                ],
                "Influence": [abs(r["contribution"]) for r in reasons],
            }
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.bar_chart(table.set_index("Factor")["Influence"])
    else:
        st.info("Explanation unavailable for this model.")

st.divider()
st.warning(
    "**Decision support only.** This is a statistical estimate produced from "
    "historical patterns, not an approval or a denial. A human reviewer makes the "
    "final lending determination and should weigh information the model cannot "
    "see. The model is trained on synthetic data and must not be used for real "
    "lending decisions."
)
