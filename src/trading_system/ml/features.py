from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "ma_distance_12",
    "ma_distance_24",
    "volatility_12",
    "volatility_24",
    "volume_z_24",
]


@dataclass(frozen=True)
class FeatureConfig:
    horizon_bars: int
    min_return_threshold: float
    fee_rate: float
    slippage_bps: float


def build_feature_frame(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    frame = _build_causal_features(data)

    future_return = frame["close"].shift(-config.horizon_bars) / frame["close"] - 1.0

    estimated_round_trip_cost = 2.0 * (
        config.fee_rate + config.slippage_bps / 10_000.0
    )

    net_future_return = future_return - estimated_round_trip_cost

    frame["target"] = (net_future_return > config.min_return_threshold).astype(int)
    frame["future_return"] = future_return
    frame["net_future_return"] = net_future_return

    frame = frame.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)

    return frame


def build_latest_inference_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = _build_causal_features(data)
    latest = frame.tail(1)

    if latest[FEATURE_COLUMNS].isna().any(axis=None):
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    return latest[FEATURE_COLUMNS].copy()


def _build_causal_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()

    close = frame["close"]
    volume = frame["volume"]

    frame["return_1"] = close.pct_change(1)
    frame["return_3"] = close.pct_change(3)
    frame["return_6"] = close.pct_change(6)
    frame["return_12"] = close.pct_change(12)
    frame["return_24"] = close.pct_change(24)

    ma_12 = close.rolling(window=12, min_periods=12).mean()
    ma_24 = close.rolling(window=24, min_periods=24).mean()

    frame["ma_distance_12"] = close / ma_12 - 1.0
    frame["ma_distance_24"] = close / ma_24 - 1.0

    frame["volatility_12"] = frame["return_1"].rolling(window=12, min_periods=12).std()
    frame["volatility_24"] = frame["return_1"].rolling(window=24, min_periods=24).std()

    volume_mean = volume.rolling(window=24, min_periods=24).mean()
    volume_std = volume.rolling(window=24, min_periods=24).std()

    frame["volume_z_24"] = (volume - volume_mean) / volume_std.replace(0.0, np.nan)

    return frame