from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class LiquiditySweepReversalStrategy:
    """Liquidity sweep / false-breakout reversal strategy with cached indicators."""

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
        self.max_lookback = self._min_bars + self._max_holding_bars + 8

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        current_open = market.open_at(index)
        current_high = market.high_at(index)
        current_low = market.low_at(index)
        current_close = market.close_at(index)
        candle_range = max(current_high - current_low, 1e-12)
        body_high = max(current_open, current_close)
        body_low = min(current_open, current_close)
        upper_wick = current_high - body_high
        lower_wick = body_low - current_low

        prev_range_high = float(market.rolling_max("high", self._sweep_window, shift=1)[index])
        prev_range_low = float(market.rolling_min("low", self._sweep_window, shift=1)[index])
        atr = market.atr(self._atr_window)
        current_atr = float(atr[index])
        trend_now = float(market.ema("close", self._trend_ema)[index])
        trend_distance_atr = abs(current_close - trend_now) / current_atr if current_atr > 0 else 999.0
        volume_std = float(market.rolling_std("volume", self._volume_window, ddof=0)[index])
        volume_mean = float(market.rolling_mean("volume", self._volume_window)[index])
        volume_z = (market.volume_at(index) - volume_mean) / volume_std if volume_std > 0 else 0.0
        if not np.isfinite(volume_z):
            volume_z = 0.0

        if not self._finite(prev_range_high, prev_range_low, current_atr, trend_now):
            return self._hold(timestamp, "indicator_not_ready", {})

        z_now = self._rolling_zscore_at(market, index)
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
            "close_zscore": z_now,
        }

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

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _rolling_zscore_at(self, market: MarketDataView, index: int) -> float:
        mean = market.rolling_mean("close", self._sweep_window)[index]
        std = market.rolling_std("close", self._sweep_window, ddof=0)[index]
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            return 0.0
        return float((market.close_at(index) - mean) / std)

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
    def _finite(*values: float) -> bool:
        return all(np.isfinite(float(value)) for value in values)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
