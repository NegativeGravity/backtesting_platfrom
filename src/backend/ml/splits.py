from __future__ import annotations

import pandas as pd

from backend.ml.time_policy import (
    CalendarSplit,
    ModelTrainingMode,
    split_raw_for_model_training,
)


def calendar_split_for_model_training(
    data: pd.DataFrame,
    mode: ModelTrainingMode,
) -> CalendarSplit:
    return split_raw_for_model_training(data=data, mode=mode)
