from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import (
    adx_bundle,
    bars_since_entry,
    choppiness,
    finite_at,
    kama,
    position_side,
    regime_snapshot,
    rolling_vwap,
    safe_denominator,
    target_fraction_from_risk,
    volume_z,
    vortex_bundle,
    realized_vol_percentile,
)
from backend.strategy.signals import Signal, SignalType


class AdaptiveTrendExpansionProStrategy:
    def __init__(
        self,
        symbol: str,
        kama_fast: int = 48,
        kama_slow: int = 144,
        channel_window: int = 72,
        choppiness_window: int = 48,
        adx_window: int = 14,
        vortex_window: int = 14,
        atr_window: int = 14,
        vwap_window: int = 24,
        volume_window: int = 72,
        realized_vol_window: int = 168,
        min_adx: float = 20.0,
        max_entry_choppiness: float = 52.0,
        no_trade_choppiness: float = 58.0,
        min_volume_z: float = 0.75,
        max_entry_range_atr: float = 2.8,
        hard_stop_atr: float = 2.2,
        trailing_stop_atr: float = 3.2,
        risk_per_trade: float = 0.005,
        max_notional_fraction: float = 1.5,
        time_stop_bars: int = 18,
    ) -> None:
        self._symbol = symbol
        self._kama_fast = int(kama_fast)
        self._kama_slow = int(kama_slow)
        self._channel_window = int(channel_window)
        self._choppiness_window = int(choppiness_window)
        self._adx_window = int(adx_window)
        self._vortex_window = int(vortex_window)
        self._atr_window = int(atr_window)
        self._vwap_window = int(vwap_window)
        self._volume_window = int(volume_window)
        self._realized_vol_window = int(realized_vol_window)
        self._min_adx = float(min_adx)
        self._max_entry_choppiness = float(max_entry_choppiness)
        self._no_trade_choppiness = float(no_trade_choppiness)
        self._min_volume_z = float(min_volume_z)
        self._max_entry_range_atr = float(max_entry_range_atr)
        self._hard_stop_atr = float(hard_stop_atr)
        self._trailing_stop_atr = float(trailing_stop_atr)
        self._risk_per_trade = float(risk_per_trade)
        self._max_notional_fraction = float(max_notional_fraction)
        self._time_stop_bars = int(time_stop_bars)
        self._min_bars = max(
            self._kama_slow,
            self._channel_window,
            self._choppiness_window,
            self._realized_vol_window,
            self._volume_window,
        ) + 8
        self.max_lookback = self._min_bars + self._time_stop_bars + 32

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        close = market.close_at(index)
        high = market.high_at(index)
        low = market.low_at(index)
        current_range = high - low
        atr = market.atr(self._atr_window)
        current_atr = float(atr[index])
        fast = kama(market, self._kama_fast)
        slow = kama(market, self._kama_slow)
        adx, _, _ = adx_bundle(market, self._adx_window)
        vortex_plus, vortex_minus = vortex_bundle(market, self._vortex_window)
        chop = choppiness(market, self._choppiness_window)
        vol_z = volume_z(market, self._volume_window)
        vwap = rolling_vwap(market, self._vwap_window)
        vol_percentile = realized_vol_percentile(market, 24, self._realized_vol_window)
        upper = market.rolling_max("close", self._channel_window, shift=1)
        lower = market.rolling_min("close", self._channel_window, shift=1)
        slope = self._slope(fast, index, 12)

        if not finite_at(index, atr, fast, slow, adx, vortex_plus, vortex_minus, chop, vwap, upper, lower):
            return self._hold(timestamp, "indicator_not_ready", {})

        regime = regime_snapshot(market, index)
        side = position_side(portfolio)
        metadata = {
            "strategy": "adaptive_trend_expansion_pro",
            "close": close,
            "kama_fast": float(fast[index]),
            "kama_slow": float(slow[index]),
            "kama_slope_12": slope,
            "donchian_high_close": float(upper[index]),
            "donchian_low_close": float(lower[index]),
            "choppiness": float(chop[index]),
            "adx": float(adx[index]),
            "vortex_plus": float(vortex_plus[index]),
            "vortex_minus": float(vortex_minus[index]),
            "atr": current_atr,
            "natr": current_atr / safe_denominator(close, 1.0),
            "rolling_vwap": float(vwap[index]),
            "volume_z": float(vol_z[index]),
            "realized_vol_percentile": float(vol_percentile[index]),
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
        }

        if side == "LONG":
            exit_signal = self._long_exit(market, index, portfolio, close, fast, current_atr)
            if exit_signal is not None:
                metadata.update(exit_signal[1])
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.72, exit_signal[0], metadata)
            return self._hold(timestamp, "trend_expansion_long_hold", metadata)

        if side == "SHORT":
            exit_signal = self._short_exit(market, index, portfolio, close, fast, current_atr)
            if exit_signal is not None:
                metadata.update(exit_signal[1])
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.72, exit_signal[0], metadata)
            return self._hold(timestamp, "trend_expansion_short_hold", metadata)

        if chop[index] > self._no_trade_choppiness:
            return self._hold(timestamp, "choppy_regime_block", metadata)

        candle_range_atr = current_range / safe_denominator(current_atr, 1.0)
        if candle_range_atr > self._max_entry_range_atr:
            metadata["candle_range_atr"] = float(candle_range_atr)
            return self._hold(timestamp, "late_breakout_range_block", metadata)

        trend_up = fast[index] > slow[index]
        trend_down = fast[index] < slow[index]
        volatility_size = target_fraction_from_risk(
            close=close,
            atr=current_atr,
            stop_atr=self._hard_stop_atr,
            risk_per_trade=self._risk_per_trade,
            max_notional_fraction=self._max_notional_fraction,
            vol_percentile=float(vol_percentile[index]),
        )

        metadata["target_notional_fraction"] = volatility_size[0]
        metadata["stop_loss_pct"] = volatility_size[1]
        metadata["take_profit_pct"] = max(volatility_size[2], volatility_size[1] * 1.8)

        long_regime_ok = regime["regime"] in {"steady_uptrend", "rapid_uptrend"} or regime["probabilities"].get("steady_uptrend", 0.0) + regime["probabilities"].get("rapid_uptrend", 0.0) > 0.45
        short_regime_ok = regime["regime"] in {"steady_downtrend", "rapid_downtrend"} or regime["probabilities"].get("steady_downtrend", 0.0) + regime["probabilities"].get("rapid_downtrend", 0.0) > 0.45

        if (
            close > upper[index]
            and trend_up
            and slope > 0.0
            and adx[index] > self._min_adx
            and vortex_plus[index] > vortex_minus[index]
            and chop[index] < self._max_entry_choppiness
            and vol_z[index] > self._min_volume_z
            and close > vwap[index]
            and long_regime_ok
        ):
            confidence = min(1.0, 0.58 + 0.08 * min(3.0, (adx[index] - self._min_adx) / 10.0) + 0.05 * max(0.0, vol_z[index]))
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "adaptive_trend_expansion_pro_long", metadata)

        if (
            close < lower[index]
            and trend_down
            and slope < 0.0
            and adx[index] > self._min_adx
            and vortex_minus[index] > vortex_plus[index]
            and chop[index] < self._max_entry_choppiness
            and vol_z[index] > self._min_volume_z
            and close < vwap[index]
            and short_regime_ok
        ):
            confidence = min(1.0, 0.58 + 0.08 * min(3.0, (adx[index] - self._min_adx) / 10.0) + 0.05 * max(0.0, vol_z[index]))
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "adaptive_trend_expansion_pro_short", metadata)

        return self._hold(timestamp, "no_trend_expansion_alignment", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _long_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, fast: np.ndarray, atr: float) -> tuple[str, dict[str, float]] | None:
        if close < fast[index]:
            return "trend_expansion_long_kama_invalidation", {}
        bars = bars_since_entry(portfolio)
        if bars > 0:
            start = max(0, index + 1 - bars)
            high_close = float(np.nanmax(market.arrays.close[start:index + 1]))
            chandelier = high_close - self._trailing_stop_atr * atr
            if close < chandelier:
                return "trend_expansion_long_chandelier_exit", {"chandelier_stop": chandelier}
            entry_price = float(getattr(portfolio, "entry_price", 0.0) or 0.0)
            if bars >= self._time_stop_bars and entry_price > 0.0:
                stop_distance = self._hard_stop_atr * atr
                achieved_r = (close - entry_price) / safe_denominator(stop_distance, 1.0)
                if achieved_r < 0.8:
                    return "trend_expansion_long_time_stop", {"achieved_r": float(achieved_r)}
        return None

    def _short_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, fast: np.ndarray, atr: float) -> tuple[str, dict[str, float]] | None:
        if close > fast[index]:
            return "trend_expansion_short_kama_invalidation", {}
        bars = bars_since_entry(portfolio)
        if bars > 0:
            start = max(0, index + 1 - bars)
            low_close = float(np.nanmin(market.arrays.close[start:index + 1]))
            chandelier = low_close + self._trailing_stop_atr * atr
            if close > chandelier:
                return "trend_expansion_short_chandelier_exit", {"chandelier_stop": chandelier}
            entry_price = float(getattr(portfolio, "entry_price", 0.0) or 0.0)
            if bars >= self._time_stop_bars and entry_price > 0.0:
                stop_distance = self._hard_stop_atr * atr
                achieved_r = (entry_price - close) / safe_denominator(stop_distance, 1.0)
                if achieved_r < 0.8:
                    return "trend_expansion_short_time_stop", {"achieved_r": float(achieved_r)}
        return None

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
