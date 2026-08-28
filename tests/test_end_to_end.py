"""One slow test that runs the real path: train -> save -> load -> score."""

import pytest

from credit_risk.config import FEATURES
from credit_risk.data import generate_dataset, save_dataset
from credit_risk.predict import render_adjudication, score_application
from credit_risk.train import load_bundle, train


@pytest.mark.slow
def test_train_then_score_a_real_application(tmp_path):
    data_path = tmp_path / "loans.csv"
    model_path = tmp_path / "model.joblib"
    save_dataset(generate_dataset(2500, seed=11), data_path)

    summary = train(
        data_path=data_path,
        model_path=model_path,
        candidates=("logistic_regression", "random_forest"),
    )
    assert summary["metrics"]["roc_auc"] > 0.65
    assert 0.0 < summary["threshold"] < 1.0
    assert model_path.exists()

    bundle = load_bundle(model_path)
    assert bundle["features"] == FEATURES
    assert "trained_at" in bundle["training_metadata"]

    application = generate_dataset(1, seed=5).iloc[[0]]
    result = score_application(bundle, application[FEATURES], background=application[FEATURES])
    assert "DECISION SUPPORT ONLY" in render_adjudication(result)


def test_load_bundle_gives_a_useful_error_when_untrained(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        load_bundle(tmp_path / "nope.joblib")
