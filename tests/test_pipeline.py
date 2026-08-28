"""Preprocessing must survive the messy inputs that real applications carry."""

import numpy as np
import pytest

from credit_risk.config import FEATURES
from credit_risk.pipeline import build_model, build_preprocessor


def test_rejects_an_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("magic_beans")


def test_preprocessor_imputes_missing_numbers(small_dataset):
    pre = build_preprocessor()
    transformed = pre.fit_transform(small_dataset[FEATURES])
    assert not np.isnan(transformed).any()


def test_preprocessor_survives_a_category_it_never_saw(small_dataset):
    """`handle_unknown="ignore"` matters: production sends new categories."""
    pre = build_preprocessor()
    pre.fit(small_dataset[FEATURES])

    row = small_dataset[FEATURES].iloc[[0]].copy()
    row["loan_purpose"] = "wedding"  # not in the training vocabulary
    transformed = pre.transform(row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_an_all_nan_column_at_predict_time(small_dataset):
    pre = build_preprocessor()
    pre.fit(small_dataset[FEATURES])
    row = small_dataset[FEATURES].iloc[[0]].copy()
    row["employment_length"] = np.nan
    assert not np.isnan(pre.transform(row)).any()


def test_scaling_is_applied_only_when_asked(small_dataset):
    unscaled = build_preprocessor(scale_numeric=False).fit_transform(small_dataset[FEATURES])
    scaled = build_preprocessor(scale_numeric=True).fit_transform(small_dataset[FEATURES])
    # Income in raw dollars has a huge spread; standardised it does not.
    assert unscaled[:, 1].std() > 100
    assert scaled[:, 1].std() == pytest.approx(1.0, abs=0.1)


def test_a_fitted_model_beats_random_guessing(fitted_pipeline, split, probabilities):
    from sklearn.metrics import roc_auc_score

    _, _, _, y_test = split
    assert roc_auc_score(y_test, probabilities) > 0.65


def test_probabilities_are_valid(probabilities):
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
