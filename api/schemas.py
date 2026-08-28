from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from credit_risk.config import (
    EMPLOYMENT_STATUS,
    HOME_OWNERSHIP,
    LOAN_PURPOSE,
    VERIFICATION_STATUS,
)


class BorrowerFeatures(BaseModel):
    annual_income: float = Field(ge=0)
    loan_amount: float = Field(ge=0)
    debt_to_income: float = Field(ge=0, le=1.5)
    credit_score: int = Field(ge=300, le=850)
    employment_length_years: float = Field(ge=0, le=50)
    credit_history_length_years: float = Field(ge=0, le=60)
    revolving_utilization: float = Field(ge=0, le=2)
    num_open_accounts: int = Field(ge=0, le=80)
    num_delinquencies_24m: int = Field(ge=0, le=20)
    num_inquiries_6m: int = Field(ge=0, le=20)
    public_records: int = Field(ge=0, le=10)
    monthly_debt: float = Field(ge=0)
    age: int = Field(ge=18, le=100)
    home_ownership: str
    loan_purpose: str
    employment_status: str
    verification_status: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "annual_income": 72000,
                "loan_amount": 14000,
                "debt_to_income": 0.31,
                "credit_score": 694,
                "employment_length_years": 4.5,
                "credit_history_length_years": 9.0,
                "revolving_utilization": 0.41,
                "num_open_accounts": 8,
                "num_delinquencies_24m": 0,
                "num_inquiries_6m": 1,
                "public_records": 0,
                "monthly_debt": 1480,
                "age": 34,
                "home_ownership": "RENT",
                "loan_purpose": "debt_consolidation",
                "employment_status": "employed",
                "verification_status": "source_verified",
            }
        }
    }


class DecisionRequest(BaseModel):
    decision: Literal["approve", "refer", "decline"]
    notes: str = Field(min_length=3, max_length=2000)
    underwriter: str = "A. Chen"


class ScoreResponse(BaseModel):
    application_id: Optional[str] = None
    borrower_name: Optional[str] = None
    features: dict[str, Any]
    pd_default: float
    risk_band: str
    policy_recommendation: str
    policy_rationale: str
    eligibility: str
    shap: list[dict[str, Any]]
    top_drivers: list[dict[str, Any]]
    expected_pd: float
    status: str = "scored"
    decision: Optional[str] = None
    decision_notes: Optional[str] = None


def reference_values() -> dict[str, list[str]]:
    return {
        "home_ownership": HOME_OWNERSHIP,
        "loan_purpose": LOAN_PURPOSE,
        "employment_status": EMPLOYMENT_STATUS,
        "verification_status": VERIFICATION_STATUS,
    }
