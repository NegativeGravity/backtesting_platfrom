import pandas as pd

from trading_system.core.exceptions import DataValidationError


def validate_ohlcv(data: pd.DataFrame) -> None:
    if data.empty:
        raise DataValidationError("Market data is empty.")

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    if data["timestamp"].isna().any():
        raise DataValidationError("Dataset contains null timestamps.")

    if data["timestamp"].duplicated().any():
        raise DataValidationError("Dataset contains duplicated timestamps.")

    if not data["timestamp"].is_monotonic_increasing:
        raise DataValidationError("Timestamps must be sorted in ascending order.")

    price_columns = ["open", "high", "low", "close"]
    if data[price_columns].isna().any().any():
        raise DataValidationError("Dataset contains null or non-numeric prices.")

    if (data[price_columns] <= 0).any().any():
        raise DataValidationError("Prices must be strictly positive.")

    if data["volume"].isna().any():
        raise DataValidationError("Dataset contains null or non-numeric volume.")

    if (data["volume"] < 0).any():
        raise DataValidationError("Volume cannot be negative.")

    invalid_high = data["high"] < data[["open", "close", "low"]].max(axis=1)
    if invalid_high.any():
        raise DataValidationError("Invalid OHLC rows: high is below open/close/low.")

    invalid_low = data["low"] > data[["open", "close", "high"]].min(axis=1)
    if invalid_low.any():
        raise DataValidationError("Invalid OHLC rows: low is above open/close/high.")