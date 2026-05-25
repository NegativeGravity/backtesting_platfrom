from __future__ import annotations

from typing import Any

import pandas as pd

from backend.strategy.signals import Signal, SignalType


class LiquiditySweepReversalStrategy:
    """
    Liquidity sweep / false-breakout reversal strategy.

    It looks for stop-hunt style candles:
    - Price sweeps the previous range high, rejects back below it, and closes with
      a strong upper wick -> SHORT.
    - Price sweeps the previous range low, rejects back above it, and closes with
      a strong lower wick -> LONG.

    Filters:
    - ATR-normalized wick size.
    - Optional volume z-score confirmation.
    - Optional regime filter to avoid fading extremely strong trends.

    No lookahead: previous range high/low are shifted by one bar.
    """

    def __init__(
        self,
        symbol: str,
        sweep_window: int = 30,
        atr_window: int = 14,
        min_wick_atr: float = 0.35,
        min_reclaim_pct: float = 0.15,
        volume_window: int = 40,
        min_volume_z: float = -0.25,
        trend_ema: int = 100,
        max_trend_distance_atr: float = 3.0,
        exit_zscore: float = 0.10,
        max_holding_bars: int = 24,
    ) -> None:
        self._symbol = symbol
        self._sweep_window = int(sweep_window)
        self._atr_window = int(atr_window)
        self._min_wick_atr = float(min_wick_atr)
        self._min_reclaim_pct = float(min_reclaim_pct)
        self._volume_window = int(volume_window)
        self._min_volume_z = float(min_volume_z)
        self._trend_ema = int(trend_ema)
        self._max_trend_distance_atr = float(max_trend_distance_atr)
        self._exit_zscore = float(exit_zscore)
        self._max_holding_bars = int(max_holding_bars)
        self._min_bars = max(self._sweep_window, self._atr_window, self._volume_window, self._trend_ema) + 3

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        timestamp = self._timestamp(market_window)

        if len(market_window) < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        frame = market_window.tail(self._min_bars + self._max_holding_bars + 5).copy()
        open_ = frame["open"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float) if "volume" in frame else pd.Series(0.0, index=frame.index)

        current_open = float(open_.iloc[-1])
        current_high = float(high.iloc[-1])
        current_low = float(low.iloc[-1])
        current_close = float(close.iloc[-1])
        candle_range = max(current_high - current_low, 1e-12)
        body_high = max(current_open, current_close)
        body_low = min(current_open, current_close)
        upper_wick = current_high - body_high
        lower_wick = body_low - current_low

        prev_range_high = float(high.rolling(self._sweep_window).max().shift(1).iloc[-1])
        prev_range_low = float(low.rolling(self._sweep_window).min().shift(1).iloc[-1])
        atr = self._atr(high, low, close, self._atr_window)
        current_atr = float(atr.iloc[-1])
        trend = close.ewm(span=self._trend_ema, adjust=False).mean()
        trend_now = float(trend.iloc[-1])
        trend_distance_atr = abs(current_close - trend_now) / current_atr if current_atr > 0 else 999.0

        volume_mean = volume.rolling(self._volume_window).mean()
        volume_std = volume.rolling(self._volume_window).std(ddof=0)
        volume_z = float(((volume - volume_mean) / volume_std.replace(0.0, pd.NA)).iloc[-1])
        if pd.isna(volume_z):
            volume_z = 0.0

        if not self._finite(prev_range_high, prev_range_low, current_atr, trend_now):
            return self._hold(timestamp, "indicator_not_ready", {})

        position_side = self._position_side(portfolio)
        metadata = {
            "open": current_open,
            "high": current_high,
            "low": current_low,
            "close": current_close,
            "prev_range_high": prev_range_high,
            "prev_range_low": prev_range_low,
            "upper_wick_atr": upper_wick / current_atr if current_atr > 0 else 0.0,
            "lower_wick_atr": lower_wick / current_atr if current_atr > 0 else 0.0,
            "upper_wick_pct_of_range": upper_wick / candle_range,
            "lower_wick_pct_of_range": lower_wick / candle_range,
            "atr": current_atr,
            "volume_z": volume_z,
            "trend_ema": trend_now,
            "trend_distance_atr": trend_distance_atr,
        }

        z = self._rolling_zscore(close, self._sweep_window)
        z_now = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0
        metadata["close_zscore"] = z_now

        if position_side == "LONG":
            if z_now >= -self._exit_zscore or self._bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.65, "liquidity_sweep_long_exit", metadata)
            return self._hold(timestamp, "liquidity_sweep_long_hold", metadata)

        if position_side == "SHORT":
            if z_now <= self._exit_zscore or self._bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.65, "liquidity_sweep_short_exit", metadata)
            return self._hold(timestamp, "liquidity_sweep_short_hold", metadata)

        regime_ok = trend_distance_atr <= self._max_trend_distance_atr
        volume_ok = volume_z >= self._min_volume_z

        swept_high = current_high > prev_range_high
        rejected_high = current_close < prev_range_high
        high_reclaim_depth = (current_high - current_close) / max(current_high - prev_range_high, 1e-12)
        upper_wick_ok = upper_wick / current_atr >= self._min_wick_atr if current_atr > 0 else False

        swept_low = current_low < prev_range_low
        rejected_low = current_close > prev_range_low
        low_reclaim_depth = (current_close - current_low) / max(prev_range_low - current_low, 1e-12)
        lower_wick_ok = lower_wick / current_atr >= self._min_wick_atr if current_atr > 0 else False

        if swept_high and rejected_high and high_reclaim_depth >= self._min_reclaim_pct and upper_wick_ok and volume_ok and regime_ok:
            confidence = min(1.0, 0.55 + 0.10 * min(3.0, upper_wick / current_atr) + 0.05 * max(0.0, volume_z))
            metadata["high_reclaim_depth"] = high_reclaim_depth
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "liquidity_sweep_short", metadata)

        if swept_low and rejected_low and low_reclaim_depth >= self._min_reclaim_pct and lower_wick_ok and volume_ok and regime_ok:
            confidence = min(1.0, 0.55 + 0.10 * min(3.0, lower_wick / current_atr) + 0.05 * max(0.0, volume_z))
            metadata["low_reclaim_depth"] = low_reclaim_depth
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "liquidity_sweep_long", metadata)

        return self._hold(timestamp, "no_liquidity_sweep", metadata)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1.0 / window, adjust=False).mean()

    @staticmethod
    def _rolling_zscore(close: pd.Series, window: int) -> pd.Series:
        mean = close.rolling(window).mean()
        std = close.rolling(window).std(ddof=0)
        return (close - mean) / std.replace(0.0, pd.NA)

    @staticmethod
    def _position_side(portfolio: Any) -> str | None:
        side = getattr(portfolio, "position_side", None)
        if side in {"LONG", "SHORT"}:
            return str(side)

        quantity = float(getattr(portfolio, "position_quantity", 0.0) or 0.0)
        if quantity > 0:
            return "LONG"
        if quantity < 0:
            return "SHORT"
        return None

    @staticmethod
    def _bars_since_entry(portfolio: Any) -> int:
        value = getattr(portfolio, "bars_since_entry", 0)
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _timestamp(market_window: pd.DataFrame) -> pd.Timestamp:
        if market_window.empty:
            return pd.Timestamp.utcnow()
        return pd.Timestamp(market_window.iloc[-1]["timestamp"])

    @staticmethod
    def _finite(*values: float) -> bool:
        return all(pd.notna(value) for value in values)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
