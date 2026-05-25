import pandas as pd


def filter_date_range(
    data: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    filtered = data.copy()

    if start_date is not None:
        filtered = filtered[filtered["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]

    if end_date is not None:
        filtered = filtered[filtered["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

    return filtered.reset_index(drop=True)