from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    output_path = Path("data/sample/btc_usdt_1h.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=42)

    periods = 1200
    timestamps = pd.date_range(
        start="2023-01-01 00:00:00",
        periods=periods,
        freq="h",
        tz="UTC",
    )

    returns = rng.normal(loc=0.00015, scale=0.015, size=periods)
    close = 20_000 * np.exp(np.cumsum(returns))

    open_prices = np.empty(periods)
    open_prices[0] = close[0]
    open_prices[1:] = close[:-1]

    intrabar_noise = rng.uniform(0.001, 0.01, size=periods)
    high = np.maximum(open_prices, close) * (1.0 + intrabar_noise)
    low = np.minimum(open_prices, close) * (1.0 - intrabar_noise)
    volume = rng.lognormal(mean=8.0, sigma=0.5, size=periods)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    data.to_csv(output_path, index=False)
    print(f"Sample dataset generated: {output_path}")


if __name__ == "__main__":
    main()