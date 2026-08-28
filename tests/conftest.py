"""Shared fixtures. A small dataset keeps the suite fast; a session-scoped
trained model means the expensive fit happens once."""

import numpy as np
import pytest

from credit_risk.data import generate_dataset
from credit_risk.pipeline import build_model
from credit_risk.train import split_data


@pytest.fixture(scope="session")
def small_dataset():
    return generate_dataset(n_rows=1200, seed=7)


@pytest.fixture(scope="session")
def split(small_dataset):
    return split_data(small_dataset)


@pytest.fixture(scope="session")
def fitted_pipeline(split):
    X_train, X_test, y_train, y_test = split
    pipeline = build_model("random_forest")
    pipeline.fit(X_train, y_train)
    return pipeline


@pytest.fixture(scope="session")
def probabilities(fitted_pipeline, split):
    X_train, X_test, y_train, y_test = split
    return np.asarray(fitted_pipeline.predict_proba(X_test)[:, 1])
