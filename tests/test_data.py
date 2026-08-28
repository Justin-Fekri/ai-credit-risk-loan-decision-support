"""The generator is the foundation of every other test -- if it drifts, the
model results stop meaning anything."""

import pandas as pd
import pytest

from credit_risk.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from credit_risk.data import generate_dataset, load_dataset, save_dataset


def test_has_every_declared_column(small_dataset):
    for column in FEATURES + [TARGET]:
        assert column in small_dataset.columns


def test_is_deterministic_for_a_given_seed():
    a = generate_dataset(300, seed=99)
    b = generate_dataset(300, seed=99)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_data():
    a = generate_dataset(300, seed=1)
    b = generate_dataset(300, seed=2)
    assert not a[TARGET].equals(b[TARGET])


def test_target_is_binary(small_dataset):
    assert set(small_dataset[TARGET].unique()) <= {0, 1}


@pytest.mark.parametrize("target_rate", [0.10, 0.18, 0.30])
def test_default_rate_tracks_its_target(target_rate):
    df = generate_dataset(6000, seed=3, target_default_rate=target_rate)
    assert df[TARGET].mean() == pytest.approx(target_rate, abs=0.03)


def test_missing_values_are_present_for_imputation_to_handle(small_dataset):
    assert small_dataset["employment_length"].isna().any()
    assert small_dataset["revolving_utilization"].isna().any()


def test_numeric_columns_are_numeric(small_dataset):
    for column in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(small_dataset[column])


def test_credit_score_stays_in_the_real_fico_range(small_dataset):
    assert small_dataset["credit_score"].between(300, 850).all()


def test_risk_relationships_point_the_right_way(small_dataset):
    """Defaulters should look worse on the drivers we deliberately encoded."""
    defaulters = small_dataset[small_dataset[TARGET] == 1]
    repayers = small_dataset[small_dataset[TARGET] == 0]
    assert defaulters["credit_score"].mean() < repayers["credit_score"].mean()
    assert defaulters["debt_to_income"].mean() > repayers["debt_to_income"].mean()
    assert (
        defaulters["num_past_delinquencies"].mean()
        > repayers["num_past_delinquencies"].mean()
    )


def test_round_trip_through_csv(tmp_path, small_dataset):
    path = tmp_path / "loans.csv"
    save_dataset(small_dataset, path)
    loaded = load_dataset(path)
    assert len(loaded) == len(small_dataset)
    for column in CATEGORICAL_FEATURES:
        assert loaded[column].dtype == "string"


def test_load_rejects_a_file_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [30], "default": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(path)
