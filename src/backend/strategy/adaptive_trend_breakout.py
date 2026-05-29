from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class AdaptiveTrendBreakoutStrategy:

    def __init__(
        self,
        symbol: str,
        fast_ema: int = 20,
        slow_ema: int = 50,
        channel_window: int = 40,
        atr_window: int = 14,
        atr_baseline_window: int = 80,
        min_atr_expansion: float = 1.05,
        exit_ema: int = 20,
    ) -> None:
        self._symbol = symbol
        self._fast_ema = int(fast_ema)
        self._slow_ema = int(slow_ema)
        self._channel_window = int(channel_window)
        self._atr_window = int(atr_window)
        self._atr_baseline_window = int(atr_baseline_window)
        self._min_atr_expansion = float(min_atr_expansion)
        self._exit_ema = int(exit_ema)
        self._min_bars = max(
            self._slow_ema,
            self._channel_window,
            self._atr_window + self._atr_baseline_window,
            self._exit_ema,
        ) + 2
        self.max_lookback = self._min_bars + 8

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        current_close = market.close_at(index)
        fast_now = float(market.ema("close", self._fast_ema)[index])
        slow_now = float(market.ema("close", self._slow_ema)[index])
        exit_ema_now = float(market.ema("close", self._exit_ema)[index])
        prev_upper = float(market.rolling_max("high", self._channel_window, shift=1)[index])
        prev_lower = float(market.rolling_min("low", self._channel_window, shift=1)[index])
        atr = market.atr(self._atr_window)
        current_atr = float(atr[index])

        # The original used a rolling mean of ATR as baseline. Cache it once per data set.
        baseline_key = f"atr_mean:{self._atr_window}:{self._atr_baseline_window}"
        if baseline_key not in market._cache:
            market._cache[baseline_key] = pd.Series(atr).rolling(self._atr_baseline_window).mean().to_numpy(dtype=np.float64)
        atr_baseline = float(market._cache[baseline_key][index])

        if not self._finite(prev_upper, prev_lower, current_atr, atr_baseline, fast_now, slow_now, exit_ema_now):
            return self._hold(timestamp, "indicator_not_ready", {})

        atr_expansion = current_atr / atr_baseline if atr_baseline > 0 else 0.0
        trend_up = fast_now > slow_now
        trend_down = fast_now < slow_now
        volatility_ok = atr_expansion >= self._min_atr_expansion
        position_side = self._position_side(portfolio)

        metadata = {
            "close": current_close,
            "fast_ema": fast_now,
            "slow_ema": slow_now,
            "exit_ema": exit_ema_now,
            "prev_upper_channel": prev_upper,
            "prev_lower_channel": prev_lower,
            "atr": current_atr,
            "atr_baseline": atr_baseline,
            "atr_expansion": atr_expansion,
            "trend_up": trend_up,
            "trend_down": trend_down,
            "volatility_ok": volatility_ok,
        }

        if position_side == "LONG":
            if (not trend_up) or current_close < exit_ema_now or atr_expansion < 0.85:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "trend_long_exit", metadata)
            return self._hold(timestamp, "trend_long_hold", metadata)

        if position_side == "SHORT":
            if (not trend_down) or current_close > exit_ema_now or atr_expansion < 0.85:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "trend_short_exit", metadata)
            return self._hold(timestamp, "trend_short_hold", metadata)

        if current_close > prev_upper and trend_up and volatility_ok:
            confidence = min(1.0, 0.55 + 0.15 * min(3.0, atr_expansion))
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "adaptive_trend_breakout_long", metadata)

        if current_close < prev_lower and trend_down and volatility_ok:
            confidence = min(1.0, 0.55 + 0.15 * min(3.0, atr_expansion))
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "adaptive_trend_breakout_short", metadata)

        return self._hold(timestamp, "no_breakout_alignment", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

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
    def _finite(*values: float) -> bool:
        return all(np.isfinite(float(value)) for value in values)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)