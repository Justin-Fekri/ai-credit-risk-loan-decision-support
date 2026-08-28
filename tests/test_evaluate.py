"""Metrics are checked against hand-computed values, not against themselves."""

import numpy as np
import pytest

from credit_risk.config import DecisionPolicy
from credit_risk.evaluate import (
    best_f1_threshold,
    classification_metrics,
    cost_analysis,
    review_workload,
)


def test_metrics_match_a_hand_built_confusion_matrix():
    #                     TN     FP     FN     TP
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8])
    # At threshold 0.5: predicted positive = indices 2,3,6,7
    # TP = 2 (indices 6,7), FP = 2 (indices 2,3), FN = 2, TN = 2
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    cm = metrics["confusion_matrix"]
    assert (cm["true_positive"], cm["false_positive"]) == (2, 2)
    assert (cm["false_negative"], cm["true_negative"]) == (2, 2)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_a_perfect_ranking_scores_auc_one():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    assert classification_metrics(y_true, y_prob)["roc_auc"] == pytest.approx(1.0)


def test_threshold_changes_the_precision_recall_trade_off(split, probabilities):
    _, _, _, y_test = split
    low = classification_metrics(y_test, probabilities, 0.2)
    high = classification_metrics(y_test, probabilities, 0.8)
    # A stricter cut-off catches fewer defaulters but is right more often.
    assert low["recall"] >= high["recall"]


def test_cost_analysis_penalises_missed_defaults_more_heavily():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])  # exactly wrong: 2 FN and 2 FP
    result = cost_analysis(y_true, y_prob, threshold=0.5,
                           cost_false_negative=5.0, cost_false_positive=1.0)
    assert result["missed_defaults"] == 2
    assert result["rejected_good_borrowers"] == 2
    assert result["total_relative_cost"] == pytest.approx(2 * 5.0 + 2 * 1.0)


def test_best_f1_threshold_is_a_probability(split, probabilities):
    _, _, _, y_test = split
    assert 0.0 <= best_f1_threshold(y_test, probabilities) <= 1.0


def test_decision_policy_bands_at_their_boundaries():
    policy = DecisionPolicy(approve_below=0.20, decline_above=0.55)
    assert policy.band(0.05) == "RECOMMEND_APPROVE"
    assert policy.band(0.199) == "RECOMMEND_APPROVE"
    assert policy.band(0.20) == "MANUAL_REVIEW"   # boundary is inclusive upward
    assert policy.band(0.55) == "MANUAL_REVIEW"
    assert policy.band(0.551) == "RECOMMEND_DECLINE"
    assert policy.band(0.99) == "RECOMMEND_DECLINE"


def test_review_workload_covers_the_whole_portfolio(split, probabilities):
    _, _, _, y_test = split
    workload = review_workload(y_test, probabilities)
    assert sum(b["applications"] for b in workload.values()) == len(y_test)
    assert sum(b["share_of_portfolio"] for b in workload.values()) == pytest.approx(1.0, abs=0.01)


def test_riskier_bands_really_do_default_more(split, probabilities):
    """The core claim of the product: the bands separate real risk."""
    _, _, _, y_test = split
    workload = review_workload(y_test, probabilities)
    approve = workload["RECOMMEND_APPROVE"]["actual_default_rate"]
    decline = workload["RECOMMEND_DECLINE"]["actual_default_rate"]
    if approve is not None and decline is not None:
        assert approve < decline
