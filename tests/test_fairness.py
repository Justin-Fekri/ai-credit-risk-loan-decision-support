"""The audit has to be right, or it provides false assurance."""

import numpy as np
import pandas as pd

from credit_risk.fairness import FOUR_FIFTHS, add_audit_segments, audit, format_report


def test_age_groups_are_derived_correctly():
    df = pd.DataFrame({"age": [22, 35, 50, 70]})
    groups = add_audit_segments(df)["age_group"].tolist()
    assert groups == ["under_30", "30_to_44", "45_to_59", "60_plus"]


def test_audit_segments_are_never_model_features():
    from credit_risk.config import FEATURES

    assert "age_group" not in FEATURES


def test_a_blatant_disparity_is_flagged():
    """Construct data where one group is approved far less often."""
    n = 200
    df = pd.DataFrame(
        {
            "age": [25] * n + [50] * n,
            "employment_status": ["employed"] * 2 * n,
            "home_ownership": ["RENT"] * 2 * n,
        }
    )
    # Young applicants all score high risk; older applicants all score low.
    y_prob = np.array([0.9] * n + [0.1] * n)
    y_true = np.array([1] * n + [0] * n)
    report = audit(df, y_true, y_prob, threshold=0.5)

    young = report["segments"]["age_group"]["under_30"]
    assert young["approval_rate"] == 0.0
    assert young["disparate_impact_ratio"] < FOUR_FIFTHS
    assert "age_group=under_30" in report["flagged_segments"]


def test_equal_treatment_is_not_flagged():
    n = 200
    df = pd.DataFrame(
        {
            "age": [25] * n + [50] * n,
            "employment_status": ["employed"] * 2 * n,
            "home_ownership": ["RENT"] * 2 * n,
        }
    )
    y_prob = np.tile([0.1, 0.9], n)  # identical distribution in both groups
    y_true = np.tile([0, 1], n)
    report = audit(df, y_true, y_prob, threshold=0.5)
    assert report["flagged_segments"] == []


def test_tiny_segments_are_reported_but_not_flagged():
    """Five applications cannot support a disparate-impact claim."""
    df = pd.DataFrame(
        {
            "age": [25] * 5 + [50] * 100,
            "employment_status": ["employed"] * 105,
            "home_ownership": ["RENT"] * 105,
        }
    )
    y_prob = np.array([0.9] * 5 + [0.1] * 100)
    y_true = np.array([1] * 5 + [0] * 100)
    report = audit(df, y_true, y_prob, threshold=0.5, min_segment_size=30)
    young = report["segments"]["age_group"]["under_30"]
    assert young["applications"] == 5
    assert young["flagged_for_review"] is False


def test_report_formats_without_error(split, probabilities):
    X_train, X_test, y_train, y_test = split
    report = audit(X_test, y_test, probabilities, threshold=0.5)
    text = format_report(report)
    assert "FAIRNESS AUDIT" in text
    assert "age_group" in text
