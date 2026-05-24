import pandas as pd
import pytest

from trading_system.core.exceptions import DataValidationError
from trading_system.data.validator import validate_ohlcv


def test_validator_rejects_duplicate_timestamps() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01"], utc=True),
            "open": [1.0, 1.0],
            "high": [2.0, 2.0],
            "low": [0.5, 0.5],
            "close": [1.5, 1.5],
            "volume": [100.0, 100.0],
        }
    )

    with pytest.raises(DataValidationError):
        validate_ohlcv(data)


def test_validator_accepts_valid_data() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"], utc=True),
            "open": [1.0, 1.5],
            "high": [2.0, 2.0],
            "low": [0.5, 1.0],
            "close": [1.5, 1.8],
            "volume": [100.0, 120.0],
        }
    )

    validate_ohlcv(data)