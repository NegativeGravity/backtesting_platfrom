
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from backend.strategy.signals import Signal


DEFAULT_FALLBACK_LOOKBACK = 1024


class IndexedStrategy(Protocol):
    def generate_signal_at(self, market: "MarketDataView", index: int, portfolio: Any) -> Signal: ...


@dataclass(frozen=True, slots=True)
class MarketArrays:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    extra: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(slots=True)
class MarketDataView:
    """Immutable, array-backed market view used inside low-latency loops.

    Engines pass only a bar index to strategies. Expensive rolling/ewm feature arrays
    are computed once per data set on first use and then read in O(1) per bar.
    The legacy DataFrame window path is kept only as a bounded fallback for custom
    strategies that have not yet implemented generate_signal_at(...).
    """

    arrays: MarketArrays
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "MarketDataView":
        arrays = frame_to_market_arrays(frame)
        return cls(
            MarketArrays(
                ts=arrays["ts"],
                open=arrays["open"],
                high=arrays["high"],
                low=arrays["low"],
                close=arrays["close"],
                volume=arrays["volume"],
                extra={key: value for key, value in arrays.items() if key not in {"ts", "open", "high", "low", "close", "volume"}},
            )
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        ts: np.ndarray,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        extra: dict[str, np.ndarray] | None = None,
    ) -> "MarketDataView":
        return cls(MarketArrays(ts=ts, open=open_, high=high, low=low, close=close, volume=volume, extra=extra or {}))

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MarketDataView":
        return cls.from_arrays(
            ts=np.asarray(payload["ts"], dtype=np.int64),
            open_=np.asarray(payload["open"], dtype=np.float64),
            high=np.asarray(payload["high"], dtype=np.float64),
            low=np.asarray(payload["low"], dtype=np.float64),
            close=np.asarray(payload["close"], dtype=np.float64),
            volume=np.asarray(payload.get("volume", []), dtype=np.float64),
            extra={key: np.asarray(value) for key, value in dict(payload.get("extra", {})).items()},
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "ts": self.arrays.ts.tolist(),
            "open": self.arrays.open.tolist(),
            "high": self.arrays.high.tolist(),
            "low": self.arrays.low.tolist(),
            "close": self.arrays.close.tolist(),
            "volume": self.arrays.volume.tolist(),
            "extra": {key: value.tolist() for key, value in self.arrays.extra.items()},
        }

    def __len__(self) -> int:
        return int(self.arrays.close.shape[0])

    def timestamp(self, index: int) -> pd.Timestamp:
        if len(self) == 0:
            return pd.Timestamp.utcnow()
        safe_index = min(max(int(index), 0), len(self) - 1)
        return pd.Timestamp(int(self.arrays.ts[safe_index]), unit="ns", tz="UTC")

    def window(self, index: int, lookback: int | None = None) -> pd.DataFrame:
        """Bounded legacy window. This intentionally never returns data.iloc[:i+1]."""
        if len(self) == 0:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        safe_index = min(max(int(index), 0), len(self) - 1)
        span = int(lookback or DEFAULT_FALLBACK_LOOKBACK)
        start = max(0, safe_index + 1 - span)
        stop = safe_index + 1
        data = {
            "timestamp": pd.to_datetime(self.arrays.ts[start:stop], utc=True),
            "open": self.arrays.open[start:stop],
            "high": self.arrays.high[start:stop],
            "low": self.arrays.low[start:stop],
            "close": self.arrays.close[start:stop],
            "volume": self.arrays.volume[start:stop],
        }
        for column, values in self.arrays.extra.items():
            data[column] = values[start:stop]
        return pd.DataFrame(data)

    def value(self, column: str, index: int) -> float:
        return float(getattr(self.arrays, column)[index])

    def close_at(self, index: int) -> float:
        return float(self.arrays.close[index])

    def open_at(self, index: int) -> float:
        return float(self.arrays.open[index])

    def high_at(self, index: int) -> float:
        return float(self.arrays.high[index])

    def low_at(self, index: int) -> float:
        return float(self.arrays.low[index])

    def volume_at(self, index: int) -> float:
        return float(self.arrays.volume[index])

    def pct_change(self, column: str, periods: int = 1) -> np.ndarray:
        key = f"pct:{column}:{periods}"
        if key not in self._cache:
            series = pd.Series(getattr(self.arrays, column), copy=False)
            self._cache[key] = series.pct_change(periods).to_numpy(dtype=np.float64)
        return self._cache[key]

    def log_diff(self, column: str = "close", periods: int = 1) -> np.ndarray:
        key = f"logdiff:{column}:{periods}"
        if key not in self._cache:
            values = np.log(np.asarray(getattr(self.arrays, column), dtype=np.float64))
            self._cache[key] = pd.Series(values, copy=False).diff(periods).to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_mean(self, column: str, window: int, min_periods: int | None = None) -> np.ndarray:
        minp = window if min_periods is None else int(min_periods)
        key = f"rmean:{column}:{window}:{minp}"
        if key not in self._cache:
            self._cache[key] = pd.Series(getattr(self.arrays, column), copy=False).rolling(window, min_periods=minp).mean().to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_std(self, column: str, window: int, ddof: int = 1, min_periods: int | None = None) -> np.ndarray:
        minp = window if min_periods is None else int(min_periods)
        key = f"rstd:{column}:{window}:{ddof}:{minp}"
        if key not in self._cache:
            self._cache[key] = pd.Series(getattr(self.arrays, column), copy=False).rolling(window, min_periods=minp).std(ddof=ddof).to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_skew(self, column: str, window: int) -> np.ndarray:
        key = f"rskew:{column}:{window}"
        if key not in self._cache:
            self._cache[key] = pd.Series(getattr(self.arrays, column), copy=False).rolling(window).skew().to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_kurt(self, column: str, window: int) -> np.ndarray:
        key = f"rkurt:{column}:{window}"
        if key not in self._cache:
            self._cache[key] = pd.Series(getattr(self.arrays, column), copy=False).rolling(window).kurt().to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_max(self, column: str, window: int, shift: int = 0) -> np.ndarray:
        key = f"rmax:{column}:{window}:{shift}"
        if key not in self._cache:
            values = pd.Series(getattr(self.arrays, column), copy=False).rolling(window).max()
            if shift:
                values = values.shift(shift)
            self._cache[key] = values.to_numpy(dtype=np.float64)
        return self._cache[key]

    def rolling_min(self, column: str, window: int, shift: int = 0) -> np.ndarray:
        key = f"rmin:{column}:{window}:{shift}"
        if key not in self._cache:
            values = pd.Series(getattr(self.arrays, column), copy=False).rolling(window).min()
            if shift:
                values = values.shift(shift)
            self._cache[key] = values.to_numpy(dtype=np.float64)
        return self._cache[key]

    def ema(self, column: str, span: int) -> np.ndarray:
        key = f"ema:{column}:{span}"
        if key not in self._cache:
            self._cache[key] = pd.Series(getattr(self.arrays, column), copy=False).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)
        return self._cache[key]

    def atr(self, window: int) -> np.ndarray:
        key = f"atr:{window}"
        if key not in self._cache:
            high = self.arrays.high
            low = self.arrays.low
            close = self.arrays.close
            previous_close = np.empty_like(close, dtype=np.float64)
            previous_close[0] = np.nan
            previous_close[1:] = close[:-1]
            true_range = np.maximum.reduce(
                [
                    high - low,
                    np.abs(high - previous_close),
                    np.abs(low - previous_close),
                ]
            )
            self._cache[key] = pd.Series(true_range).ewm(alpha=1.0 / window, adjust=False).mean().to_numpy(dtype=np.float64)
        return self._cache[key]


    def cached_rolling_array(self, name: str, values: np.ndarray, window: int, op: str) -> np.ndarray:
        key = f"derived:{name}:{window}:{op}"
        if key not in self._cache:
            self._cache[key] = _rolling_from_array(values, window, op)
        return self._cache[key]

    def downside_vol(self, window: int) -> np.ndarray:
        key = f"downvol:{window}"
        if key not in self._cache:
            returns = self.pct_change("close", 1)
            values = np.where(returns < 0.0, returns, 0.0)
            self._cache[key] = pd.Series(values).rolling(window).std().to_numpy(dtype=np.float64)
        return self._cache[key]

    def upside_vol(self, window: int) -> np.ndarray:
        key = f"upvol:{window}"
        if key not in self._cache:
            returns = self.pct_change("close", 1)
            values = np.where(returns > 0.0, returns, 0.0)
            self._cache[key] = pd.Series(values).rolling(window).std().to_numpy(dtype=np.float64)
        return self._cache[key]

    def ml_momentum_features_at(self, index: int) -> pd.DataFrame:
        from backend.ml.features import FEATURE_COLUMNS

        if index < 25:
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        close = self.arrays.close
        volume = self.arrays.volume
        row = {
            "return_1": self.pct_change("close", 1)[index],
            "return_3": self.pct_change("close", 3)[index],
            "return_6": self.pct_change("close", 6)[index],
            "return_12": self.pct_change("close", 12)[index],
            "return_24": self.pct_change("close", 24)[index],
            "ma_distance_12": close[index] / self.rolling_mean("close", 12)[index] - 1.0,
            "ma_distance_24": close[index] / self.rolling_mean("close", 24)[index] - 1.0,
            "volatility_12": self.cached_rolling_array("return_1", self.pct_change("close", 1), 12, "std")[index],
            "volatility_24": self.cached_rolling_array("return_1", self.pct_change("close", 1), 24, "std")[index],
            "volume_z_24": (volume[index] - self.rolling_mean("volume", 24)[index]) / _safe_denominator(self.rolling_std("volume", 24)[index]),
        }
        if any(not np.isfinite(float(row[column])) for column in FEATURE_COLUMNS):
            return pd.DataFrame(columns=FEATURE_COLUMNS)
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def ml_regime_features_at(self, index: int, *, trend_window: int, volatility_window: int) -> pd.DataFrame | None:
        required = max(trend_window, volatility_window) + 5
        if index + 1 < required:
            return None

        close = self.arrays.close
        high = self.arrays.high
        low = self.arrays.low
        volume = self.arrays.volume
        returns_1 = self.pct_change("close", 1)
        log_returns = self.log_diff("close", 1)
        ema_fast = self.ema("close", 16)
        ema_slow = self.ema("close", trend_window)
        realized_vol = self.cached_rolling_array("log_returns", log_returns, volatility_window, "std")
        vol_of_vol = self.cached_rolling_array(f"realized_vol_{volatility_window}", realized_vol, volatility_window, "std")
        range_atr = self.atr(14)
        channel_high = self.rolling_max("high", trend_window, shift=1)
        channel_low = self.rolling_min("low", trend_window, shift=1)
        channel_width = channel_high[index] - channel_low[index]
        volume_std = self.rolling_std("volume", volatility_window)
        volume_mean = self.rolling_mean("volume", volatility_window)

        row: dict[str, float] = {}
        for horizon in (1, 3, 6, 12, 24, 48):
            row[f"ret_{horizon}"] = self.pct_change("close", horizon)[index]
            row[f"logret_{horizon}"] = self.log_diff("close", horizon)[index]

        row.update(
            {
                "ema_fast_distance": close[index] / _safe_denominator(ema_fast[index]) - 1.0,
                "ema_slow_distance": close[index] / _safe_denominator(ema_slow[index]) - 1.0,
                "trend_slope": ema_slow[index] / _safe_denominator(ema_slow[index - 8]) - 1.0 if index >= 8 else np.nan,
                "realized_vol": realized_vol[index],
                "vol_of_vol": vol_of_vol[index],
                "atr_pct": range_atr[index] / _safe_denominator(close[index]),
                "channel_position": (close[index] - channel_low[index]) / _safe_denominator(channel_width),
                "breakout_pressure": (close[index] - channel_high[index]) / _safe_denominator(range_atr[index]),
                "breakdown_pressure": (channel_low[index] - close[index]) / _safe_denominator(range_atr[index]),
                "volume_z": (volume[index] - volume_mean[index]) / _safe_denominator(volume_std[index]),
                "downside_vol": self.downside_vol(volatility_window)[index],
                "upside_vol": self.upside_vol(volatility_window)[index],
                "return_skew": self.cached_rolling_array("return_1", returns_1, volatility_window, "skew")[index],
                "return_kurt": self.cached_rolling_array("return_1", returns_1, volatility_window, "kurt")[index],
            }
        )
        clean = {key: (0.0 if not np.isfinite(float(value)) else float(value)) for key, value in row.items()}
        return pd.DataFrame([clean])

    def dl_sequence_at(
        self,
        index: int,
        *,
        feature_columns: list[str],
        sequence_length: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ) -> np.ndarray | None:
        required = sequence_length + 64
        if index + 1 < required:
            return None
        matrix = self._dl_feature_matrix(feature_columns)
        start = index + 1 - sequence_length
        sequence = matrix[start:index + 1].astype(np.float32, copy=True)
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        return (sequence - feature_mean) / feature_std

    def _dl_feature_matrix(self, feature_columns: list[str]) -> np.ndarray:
        key = "dl:" + ",".join(feature_columns)
        if key in self._cache:
            return self._cache[key]

        close = self.arrays.close
        high = self.arrays.high
        low = self.arrays.low
        open_ = self.arrays.open
        volume = self.arrays.volume
        atr = self.atr(14)
        ema16 = self.ema("close", 16)
        ema64 = self.ema("close", 64)
        channel_high = self.rolling_max("high", 64, shift=1)
        channel_low = self.rolling_min("low", 64, shift=1)
        channel_width = channel_high - channel_low
        volume_z = (volume - self.rolling_mean("volume", 48)) / _safe_denominator_array(self.rolling_std("volume", 48))

        feature_map = {
            "log_return_1": self.log_diff("close", 1),
            "log_return_3": self.log_diff("close", 3),
            "log_return_12": self.log_diff("close", 12),
            "body_atr": (close - open_) / _safe_denominator_array(atr),
            "range_atr": (high - low) / _safe_denominator_array(atr),
            "upper_wick_atr": (high - np.maximum(open_, close)) / _safe_denominator_array(atr),
            "lower_wick_atr": (np.minimum(open_, close) - low) / _safe_denominator_array(atr),
            "ema_16_distance": close / _safe_denominator_array(ema16) - 1.0,
            "ema_64_distance": close / _safe_denominator_array(ema64) - 1.0,
            "ema_spread": ema16 / _safe_denominator_array(ema64) - 1.0,
            "realized_vol_24": self.cached_rolling_array("log_return_1", self.log_diff("close", 1), 24, "std"),
            "realized_vol_64": self.cached_rolling_array("log_return_1", self.log_diff("close", 1), 64, "std"),
            "volume_z": volume_z,
            "channel_position": (close - channel_low) / _safe_denominator_array(channel_width),
            "breakout_distance_atr": (close - channel_high) / _safe_denominator_array(atr),
            "breakdown_distance_atr": (channel_low - close) / _safe_denominator_array(atr),
        }
        matrix = np.column_stack([feature_map.get(column, np.zeros(len(self), dtype=np.float64)) for column in feature_columns])
        self._cache[key] = matrix
        return matrix


def call_strategy_signal(strategy: Any, market: MarketDataView, index: int, portfolio: Any) -> Signal:
    fast_path = getattr(strategy, "generate_signal_at", None)
    if callable(fast_path):
        return fast_path(market=market, index=index, portfolio=portfolio)

    lookback = int(
        getattr(strategy, "max_lookback", None)
        or getattr(strategy, "_min_bars", None)
        or getattr(strategy, "_lookback", None)
        or DEFAULT_FALLBACK_LOOKBACK
    )
    lookback = max(1, min(lookback + 16, DEFAULT_FALLBACK_LOOKBACK))
    return strategy.generate_signal(market_window=market.window(index, lookback), portfolio=portfolio)


def frame_to_market_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if frame.empty:
        raise ValueError("Market data frame is empty.")
    normalized = frame.reset_index(drop=True).copy()
    if "timestamp" not in normalized.columns and "ts" in normalized.columns:
        normalized = normalized.rename(columns={"ts": "timestamp"})
    timestamps = pd.to_datetime(normalized["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
    volume = normalized["volume"].to_numpy(dtype=np.float64, copy=True) if "volume" in normalized.columns else np.zeros(len(normalized), dtype=np.float64)
    arrays = {
        "ts": timestamps,
        "open": normalized["open"].to_numpy(dtype=np.float64, copy=True),
        "high": normalized["high"].to_numpy(dtype=np.float64, copy=True),
        "low": normalized["low"].to_numpy(dtype=np.float64, copy=True),
        "close": normalized["close"].to_numpy(dtype=np.float64, copy=True),
        "volume": volume,
    }
    base_columns = {"timestamp", "ts", "open", "high", "low", "close", "volume"}
    for column in normalized.columns:
        if column in base_columns:
            continue
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.notna().any():
            arrays[column] = values.fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    return arrays


def _safe_denominator(value: float, fallback: float = np.nan) -> float:
    if not np.isfinite(float(value)) or abs(float(value)) <= 1e-15:
        return fallback
    return float(value)


def _safe_denominator_array(values: np.ndarray) -> np.ndarray:
    safe = np.asarray(values, dtype=np.float64).copy()
    safe[~np.isfinite(safe) | (np.abs(safe) <= 1e-15)] = np.nan
    return safe


def _rolling_from_array(values: np.ndarray, window: int, op: str) -> np.ndarray:
    key_series = pd.Series(values, copy=False)
    rolling = key_series.rolling(window)
    if op == "std":
        return rolling.std().to_numpy(dtype=np.float64)
    if op == "skew":
        return rolling.skew().to_numpy(dtype=np.float64)
    if op == "kurt":
        return rolling.kurt().to_numpy(dtype=np.float64)
    raise ValueError(f"Unsupported rolling operation: {op}")

