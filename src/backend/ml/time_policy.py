from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
TUNED_TRAIN_END = pd.Timestamp("2024-01-01T00:00:00Z")
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
VALIDATION_END = pd.Timestamp("2025-01-01T00:00:00Z")

TRAIN_ONLY_END = pd.Timestamp("2025-01-01T00:00:00Z")

BACKTEST_START = pd.Timestamp("2025-01-01T00:00:00Z")
BACKTEST_END = pd.Timestamp("2026-01-01T00:00:00Z")


TRAIN_START_STR = "2020-01-01T00:00:00Z"
TUNED_TRAIN_END_STR = "2024-01-01T00:00:00Z"
VALIDATION_START_STR = "2024-01-01T00:00:00Z"
VALIDATION_END_STR = "2025-01-01T00:00:00Z"
TRAIN_ONLY_END_STR = "2025-01-01T00:00:00Z"
BACKTEST_START_STR = "2025-01-01T00:00:00Z"
BACKTEST_END_STR = "2026-01-01T00:00:00Z"


class ModelTrainingMode(str, Enum):
    TRAIN_ONLY = "train_only"
    TUNED = "tuned"


@dataclass(frozen=True)
class CalendarSplit:
    train: pd.DataFrame
    validation: pd.DataFrame | None = None


def normalize_timestamp_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot use empty market data.")

    frame = data.copy()

    if "timestamp" not in frame.columns and "ts" in frame.columns:
        frame = frame.rename(columns={"ts": "timestamp"})

    if "timestamp" not in frame.columns:
        raise ValueError("Data must contain timestamp or ts column.")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    return frame


def slice_half_open(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = normalize_timestamp_frame(data)
    return frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)].copy()


def split_raw_for_model_training(
    data: pd.DataFrame,
    mode: ModelTrainingMode,
) -> CalendarSplit:
    frame = normalize_timestamp_frame(data)

    if mode == ModelTrainingMode.TRAIN_ONLY:
        train = slice_half_open(frame, TRAIN_START, TRAIN_ONLY_END)
        _require_non_empty(train, "train_only")
        assert_no_2025(train, "train_only")
        return CalendarSplit(train=train.reset_index(drop=True))

    if mode == ModelTrainingMode.TUNED:
        train = slice_half_open(frame, TRAIN_START, TUNED_TRAIN_END)
        validation = slice_half_open(frame, VALIDATION_START, VALIDATION_END)

        _require_non_empty(train, "train")
        _require_non_empty(validation, "validation")

        assert_no_2025(train, "train")
        assert_no_2025(validation, "validation")

        return CalendarSplit(
            train=train.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
        )

    raise ValueError(f"Unsupported model training mode: {mode}")


def assert_no_2025(data: pd.DataFrame, label: str) -> None:
    frame = normalize_timestamp_frame(data)

    if frame["timestamp"].max() >= BACKTEST_START:
        raise ValueError(
            f"Data leakage detected in {label}: model data contains 2025. "
            f"max_timestamp={frame['timestamp'].max()}"
        )


def assert_backtest_2025_only(data: pd.DataFrame) -> None:
    frame = normalize_timestamp_frame(data)

    min_ts = frame["timestamp"].min()
    max_ts = frame["timestamp"].max()

    if min_ts < BACKTEST_START or max_ts >= BACKTEST_END:
        raise ValueError(
            "Backtest data must be strictly inside 2025. "
            f"Expected [{BACKTEST_START}, {BACKTEST_END}), "
            f"got min={min_ts}, max={max_ts}."
        )


def _require_non_empty(data: pd.DataFrame, label: str) -> None:
    if data.empty:
        raise ValueError(f"{label} split is empty.")