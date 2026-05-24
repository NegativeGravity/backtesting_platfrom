from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.strategy.signals import Signal, SignalType


class AdaptiveTrendBreakoutStrategy:
    """
    Adaptive trend-following breakout strategy.

    Designed for candle-level backtests on one liquid asset.
    It trades only when three independent conditions agree:
    1. Trend regime: fast EMA vs slow EMA.
    2. Breakout: close breaks the previous Donchian channel.
    3. Volatility expansion: current ATR is above its own baseline.

    Long:
        close > previous upper channel AND fast EMA > slow EMA AND ATR expansion.
    Short:
        close < previous lower channel AND fast EMA < slow EMA AND ATR expansion.
    Exit:
        trend flips, price loses trailing EMA, or volatility collapses.

    No lookahead: all channel levels used for entry are shifted by one bar.
    """

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

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        timestamp = self._timestamp(market_window)

        if len(market_window) < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        frame = market_window.tail(self._min_bars + 5).copy()
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)

        current_close = float(close.iloc[-1])
        fast = close.ewm(span=self._fast_ema, adjust=False).mean()
        slow = close.ewm(span=self._slow_ema, adjust=False).mean()
        exit_ema = close.ewm(span=self._exit_ema, adjust=False).mean()

        prev_upper = float(high.rolling(self._channel_window).max().shift(1).iloc[-1])
        prev_lower = float(low.rolling(self._channel_window).min().shift(1).iloc[-1])

        atr = self._atr(high=high, low=low, close=close, window=self._atr_window)
        current_atr = float(atr.iloc[-1])
        atr_baseline = float(atr.rolling(self._atr_baseline_window).mean().iloc[-1])

        if not self._finite(prev_upper, prev_lower, current_atr, atr_baseline):
            return self._hold(timestamp, "indicator_not_ready", {})

        atr_expansion = current_atr / atr_baseline if atr_baseline > 0 else 0.0
        fast_now = float(fast.iloc[-1])
        slow_now = float(slow.iloc[-1])
        exit_ema_now = float(exit_ema.iloc[-1])

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
    def _timestamp(market_window: pd.DataFrame) -> pd.Timestamp:
        if market_window.empty:
            return pd.Timestamp.utcnow()
        return pd.Timestamp(market_window.iloc[-1]["timestamp"])

    @staticmethod
    def _finite(*values: float) -> bool:
        return all(pd.notna(value) for value in values)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
