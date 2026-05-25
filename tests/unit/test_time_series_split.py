import pandas as pd
import pytest

from backend.ml.splits import time_based_train_validation_test_split


def test_time_based_split_preserves_order() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="h", tz="UTC"),
            "value": range(100),
        }
    )

    split = time_based_train_validation_test_split(
        data=data,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    assert split.train["value"].max() < split.validation["value"].min()
    assert split.validation["value"].max() < split.test["value"].min()


def test_time_based_split_rejects_invalid_ratios() -> None:
    data = pd.DataFrame({"value": range(10)})

    with pytest.raises(ValueError):
        time_based_train_validation_test_split(
            data=data,
            train_ratio=0.8,
            validation_ratio=0.3,
        )