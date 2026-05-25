from pathlib import Path

import pandas as pd

from backend.core.exceptions import DataValidationError

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def load_ohlcv_csv(path: Path, timestamp_column: str = "timestamp") -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Market data file not found: {path}")

    data = pd.read_csv(path)

    if timestamp_column not in data.columns:
        raise DataValidationError(f"Timestamp column not found: {timestamp_column}")

    data = data.rename(columns={timestamp_column: "timestamp"})

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data