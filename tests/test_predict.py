"""Scoring, reason codes, and the reviewer-facing summary."""

import pytest

from credit_risk.config import FEATURES, DecisionPolicy
from credit_risk.explain import explain_application
from credit_risk.predict import (
    DISCLAIMER,
    PROBABILITY_BOUNDS,
    render_adjudication,
    risk_band,
    score_application,
    score_batch,
)


@pytest.fixture(scope="module")
def bundle(fitted_pipeline):
    return {
        "model": fitted_pipeline,
        "uncalibrated_model": fitted_pipeline,
        "model_name": "random_forest",
        "threshold": 0.3,
        "policy": DecisionPolicy(),
        "features": FEATURES,
    }


def test_risk_bands_are_ordered():
    assert risk_band(0.02) == "Very low"
    assert risk_band(0.30) == "Moderate"
    assert risk_band(0.95) == "High"


def test_scoring_returns_the_expected_shape(bundle, split):
    _, X_test, _, _ = split
    result = score_application(bundle, X_test.iloc[[0]], background=X_test)
    assert 0 <= result["default_probability"] <= 1
    assert result["recommendation"] in {
        "RECOMMEND_APPROVE", "MANUAL_REVIEW", "RECOMMEND_DECLINE"
    }
    assert isinstance(result["reasons"], list)


def test_probability_is_never_reported_as_certainty(bundle, split):
    """A calibrated model must not tell a loan officer '100%'."""
    _, X_test, _, _ = split
    for i in range(min(25, len(X_test))):
        p = score_application(bundle, X_test.iloc[[i]], background=X_test)["default_probability"]
        assert PROBABILITY_BOUNDS[0] <= p <= PROBABILITY_BOUNDS[1]


def test_a_missing_field_is_rejected_rather_than_guessed(bundle, split):
    _, X_test, _, _ = split
    incomplete = X_test.iloc[[0]].drop(columns=["credit_score"])
    with pytest.raises(ValueError, match="missing required fields"):
        score_application(bundle, incomplete)


def test_a_clearly_worse_applicant_scores_higher_risk(bundle, split):
    _, X_test, _, _ = split
    base = X_test.iloc[[0]].copy()

    good = base.copy()
    good.loc[:, "credit_score"] = 800
    good.loc[:, "debt_to_income"] = 0.10
    good.loc[:, "num_past_delinquencies"] = 0
    good.loc[:, "revolving_utilization"] = 0.05

    bad = base.copy()
    bad.loc[:, "credit_score"] = 520
    bad.loc[:, "debt_to_income"] = 0.75
    bad.loc[:, "num_past_delinquencies"] = 4
    bad.loc[:, "revolving_utilization"] = 0.98

    good_p = score_application(bundle, good, background=X_test)["default_probability"]
    bad_p = score_application(bundle, bad, background=X_test)["default_probability"]
    assert bad_p > good_p


def test_reason_codes_are_ordered_by_influence(fitted_pipeline, split):
    _, X_test, _, _ = split
    reasons = explain_application(fitted_pipeline, X_test.iloc[[0]], X_test, top_n=5)
    assert len(reasons) == 5
    magnitudes = [abs(r["contribution"]) for r in reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for reason in reasons:
        assert reason["direction"] in {"increases risk", "decreases risk"}
        assert reason["feature"] in FEATURES


def test_explaining_more_than_one_row_is_an_error(fitted_pipeline, split):
    _, X_test, _, _ = split
    with pytest.raises(ValueError, match="exactly one row"):
        explain_application(fitted_pipeline, X_test.iloc[:2], X_test)


def test_batch_scoring_covers_every_row(bundle, split):
    _, X_test, _, _ = split
    scored = score_batch(bundle, X_test.head(50))
    assert len(scored) == 50
    assert {"default_probability", "risk_band", "recommendation"} <= set(scored.columns)


def test_adjudication_summary_always_states_it_is_not_a_decision(bundle, split):
    _, X_test, _, _ = split
    result = score_application(bundle, X_test.iloc[[0]], background=X_test)
    text = render_adjudication(result)
    assert "DECISION SUPPORT ONLY" in text
    assert "reviewer" in text.lower()
    assert result["risk_band"] in text


def test_disclaimer_never_promises_an_automatic_decision():
    lowered = DISCLAIMER.lower()
    assert "not an approval" in lowered
    assert "human reviewer" in lowered
