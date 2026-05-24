from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSeriesSplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def time_based_train_validation_test_split(
    data: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
) -> TimeSeriesSplitResult:
    if data.empty:
        raise ValueError("Cannot split empty dataset.")

    total_rows = len(data)
    train_end = int(total_rows * train_ratio)
    validation_end = int(total_rows * (train_ratio + validation_ratio))

    if train_end <= 0 or validation_end <= train_end or validation_end >= total_rows:
        raise ValueError("Invalid split sizes for time series dataset.")

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    return TimeSeriesSplitResult(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )