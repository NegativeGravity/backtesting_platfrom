import pandas as pd
import pytest

from backend.ml.time_policy import (
    ModelTrainingMode,
    assert_backtest_2025_only,
    assert_no_2025,
    split_raw_for_model_training,
)


def make_data(start: str, end: str) -> pd.DataFrame:
    ts = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )


def test_tuned_split_never_contains_2025():
    data = make_data("2020-01-01", "2025-12-31")
    split = split_raw_for_model_training(data, ModelTrainingMode.TUNED)

    assert split.train["timestamp"].max() < pd.Timestamp("2024-01-01T00:00:00Z")
    assert split.validation is not None
    assert split.validation["timestamp"].min() >= pd.Timestamp("2024-01-01T00:00:00Z")
    assert split.validation["timestamp"].max() < pd.Timestamp("2025-01-01T00:00:00Z")

    assert_no_2025(split.train, "train")
    assert_no_2025(split.validation, "validation")


def test_train_only_split_uses_2020_to_2024_only():
    data = make_data("2020-01-01", "2025-12-31")
    split = split_raw_for_model_training(data, ModelTrainingMode.TRAIN_ONLY)

    assert split.validation is None
    assert split.train["timestamp"].min() >= pd.Timestamp("2020-01-01T00:00:00Z")
    assert split.train["timestamp"].max() < pd.Timestamp("2025-01-01T00:00:00Z")


def test_backtest_must_be_2025_only():
    data = make_data("2025-01-01", "2025-12-31")
    assert_backtest_2025_only(data)


def test_backtest_rejects_2024_data():
    data = make_data("2024-12-30", "2025-12-31")

    with pytest.raises(ValueError):
        assert_backtest_2025_only(data)