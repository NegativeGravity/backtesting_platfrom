from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import (
    bars_since_entry,
    finite_at,
    kama,
    position_side,
    regime_snapshot,
    safe_denominator,
    squeeze_release,
    target_fraction_from_risk,
    volume_z,
    realized_vol_percentile,
)
from backend.strategy.signals import Signal, SignalType


class VolatilitySqueezeBreakoutStrategy:
    def __init__(
        self,
        symbol: str,
        bb_window: int = 20,
        kc_window: int = 20,
        kc_atr_mult: float = 1.5,
        min_squeeze_bars: int = 8,
        channel_window: int = 48,
        kama_window: int = 48,
        volume_window: int = 72,
        atr_window: int = 14,
        min_volume_z: float = 1.0,
        hard_stop_atr: float = 1.8,
        risk_per_trade: float = 0.0045,
        max_notional_fraction: float = 1.25,
        max_holding_bars: int = 20,
    ) -> None:
        self._symbol = symbol
        self._bb_window = int(bb_window)
        self._kc_window = int(kc_window)
        self._kc_atr_mult = float(kc_atr_mult)
        self._min_squeeze_bars = int(min_squeeze_bars)
        self._channel_window = int(channel_window)
        self._kama_window = int(kama_window)
        self._volume_window = int(volume_window)
        self._atr_window = int(atr_window)
        self._min_volume_z = float(min_volume_z)
        self._hard_stop_atr = float(hard_stop_atr)
        self._risk_per_trade = float(risk_per_trade)
        self._max_notional_fraction = float(max_notional_fraction)
        self._max_holding_bars = int(max_holding_bars)
        self._min_bars = max(self._channel_window, self._kama_window, self._volume_window, 168) + self._min_squeeze_bars + 4
        self.max_lookback = self._min_bars + self._max_holding_bars + 16

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        close = market.close_at(index)
        atr = market.atr(self._atr_window)
        trend = kama(market, self._kama_window)
        release = squeeze_release(market, self._bb_window, self._kc_window, self._kc_atr_mult, self._min_squeeze_bars)
        upper = market.rolling_max("close", self._channel_window, shift=1)
        lower = market.rolling_min("close", self._channel_window, shift=1)
        vol_z = volume_z(market, self._volume_window)
        slope = self._slope(trend, index, 12)

        if not finite_at(index, atr, trend, upper, lower):
            return self._hold(timestamp, "indicator_not_ready", {})

        regime = regime_snapshot(market, index)
        side = position_side(portfolio)
        metadata = {
            "strategy": "volatility_squeeze_breakout",
            "close": close,
            "squeeze_released": bool(release[index] > 0.0),
            "donchian_high_close": float(upper[index]),
            "donchian_low_close": float(lower[index]),
            "kama": float(trend[index]),
            "kama_slope_12": slope,
            "volume_z": float(vol_z[index]),
            "atr": float(atr[index]),
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
        }

        if side == "LONG":
            if close < trend[index] or bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.66, "squeeze_breakout_long_exit", metadata)
            return self._hold(timestamp, "squeeze_breakout_long_hold", metadata)

        if side == "SHORT":
            if close > trend[index] or bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.66, "squeeze_breakout_short_exit", metadata)
            return self._hold(timestamp, "squeeze_breakout_short_hold", metadata)

        sizing = target_fraction_from_risk(
            close=close,
            atr=float(atr[index]),
            stop_atr=self._hard_stop_atr,
            risk_per_trade=self._risk_per_trade,
            max_notional_fraction=self._max_notional_fraction,
            vol_percentile=float(realized_vol_percentile(market, 24, 168)[index]),
        )
        metadata["target_notional_fraction"] = sizing[0]
        metadata["stop_loss_pct"] = sizing[1]
        metadata["take_profit_pct"] = max(sizing[2], sizing[1] * 1.8)

        if release[index] > 0.0 and close > upper[index] and slope > 0.0 and vol_z[index] > self._min_volume_z:
            confidence = min(1.0, 0.57 + 0.08 * max(0.0, vol_z[index]) + 0.08 * min(3.0, abs(slope) * 500.0))
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "volatility_squeeze_breakout_long", metadata)

        if release[index] > 0.0 and close < lower[index] and slope < 0.0 and vol_z[index] > self._min_volume_z:
            confidence = min(1.0, 0.57 + 0.08 * max(0.0, vol_z[index]) + 0.08 * min(3.0, abs(slope) * 500.0))
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "volatility_squeeze_breakout_short", metadata)

        return self._hold(timestamp, "no_squeeze_breakout", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    @staticmethod
    def _slope(values: np.ndarray, index: int, lookback: int) -> float:
        if index < lookback:
            return 0.0
        previous = float(values[index - lookback])
        current = float(values[index])
        if not np.isfinite(current) or not np.isfinite(previous) or abs(previous) <= 1e-12:
            return 0.0
        return current / previous - 1.0

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
