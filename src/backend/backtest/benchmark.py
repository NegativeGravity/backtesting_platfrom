import pandas as pd


def build_buy_and_hold_benchmark(
    data: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot build benchmark from empty data.")

    first_open = float(data.iloc[0]["open"])
    quantity = initial_capital / first_open

    benchmark = data[["timestamp", "close"]].copy()
    benchmark["benchmark_equity"] = quantity * benchmark["close"]
    benchmark["benchmark_return"] = benchmark["benchmark_equity"] / initial_capital - 1.0

    return benchmark[["timestamp", "benchmark_equity", "benchmark_return"]]
