"""Human-in-the-loop policy layer. The model estimates default risk; a reviewer decides."""

from __future__ import annotations


def risk_band(pd_default: float) -> str:
    if pd_default < 0.08:
        return "low"
    if pd_default < 0.18:
        return "moderate"
    if pd_default < 0.32:
        return "elevated"
    return "severe"


def recommend(
    pd_default: float, credit_score: float, debt_to_income: float
) -> tuple[str, str]:
    """Map model output + hard policy gates to a *recommendation*, not a decision.

    Returns (action, rationale) where action is approve | refer | decline.
    """
    band = risk_band(pd_default)

    if credit_score < 580:
        return (
            "decline",
            "Credit score is below the 580 minimum in the current credit policy.",
        )
    if pd_default >= 0.40:
        return (
            "decline",
            "Estimated default probability is above the 40% automatic-refer-out threshold.",
        )
    if debt_to_income > 0.50 and pd_default >= 0.22:
        return (
            "decline",
            "Debt-to-income exceeds 50% with elevated modelled default risk.",
        )

    if (
        pd_default <= 0.10
        and credit_score >= 700
        and debt_to_income <= 0.36
        and band == "low"
    ):
        return (
            "approve",
            "Low modelled default risk, credit score and DTI sit inside the standard approve band.",
        )

    if pd_default <= 0.14 and credit_score >= 660 and debt_to_income <= 0.40:
        return (
            "approve",
            "Risk, score, and leverage are within the standard eligibility window.",
        )

    return (
        "refer",
        "The file sits outside automatic approve/decline bands and needs an underwriter.",
    )


def eligibility_label(recommendation: str) -> str:
    return {
        "approve": "Eligible — recommend approve",
        "refer": "Conditional — refer to underwriter",
        "decline": "Not eligible under current policy",
    }[recommendation]
