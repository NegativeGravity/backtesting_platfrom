import pandas as pd


def filter_date_range(
    data: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = data.copy()
    filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)

    if start_date is not None:
        filtered = filtered[filtered["timestamp"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        filtered = filtered[filtered["timestamp"] < pd.Timestamp(end_date)]

    return filtered.reset_index(drop=True)
