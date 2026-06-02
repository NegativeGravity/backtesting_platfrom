from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import (
    adx_bundle,
    bars_since_entry,
    finite_at,
    position_side,
    regime_snapshot,
    safe_denominator,
    target_fraction_from_risk,
    volume_z,
    realized_vol_percentile,
)
from backend.strategy.signals import Signal, SignalType


class RegimeAdaptiveBtcTrendBreakoutStrategy:
    def __init__(
        self,
        symbol: str,
        fast_ema: int = 50,
        slow_ema: int = 200,
        channel_window: int = 48,
        adx_window: int = 14,
        atr_window: int = 14,
        realized_vol_window: int = 24,
        realized_vol_baseline_window: int = 720,
        volume_window: int = 72,
        min_adx: float = 22.0,
        flat_adx: float = 18.0,
        min_volume_z: float = 0.5,
        max_ema_spread_for_flat: float = 0.004,
        low_bandwidth_percentile: float = 30.0,
        trailing_stop_atr: float = 3.0,
        drawdown_stop_atr: float = 2.5,
        risk_per_trade: float = 0.006,
        max_notional_fraction: float = 1.5,
        max_holding_bars: int = 72,
    ) -> None:
        self._symbol = symbol
        self._fast_ema = int(fast_ema)
        self._slow_ema = int(slow_ema)
        self._channel_window = int(channel_window)
        self._adx_window = int(adx_window)
        self._atr_window = int(atr_window)
        self._realized_vol_window = int(realized_vol_window)
        self._realized_vol_baseline_window = int(realized_vol_baseline_window)
        self._volume_window = int(volume_window)
        self._min_adx = float(min_adx)
        self._flat_adx = float(flat_adx)
        self._min_volume_z = float(min_volume_z)
        self._max_ema_spread_for_flat = float(max_ema_spread_for_flat)
        self._low_bandwidth_percentile = float(low_bandwidth_percentile)
        self._trailing_stop_atr = float(trailing_stop_atr)
        self._drawdown_stop_atr = float(drawdown_stop_atr)
        self._risk_per_trade = float(risk_per_trade)
        self._max_notional_fraction = float(max_notional_fraction)
        self._max_holding_bars = int(max_holding_bars)
        self._min_bars = max(
            self._slow_ema,
            self._channel_window,
            self._realized_vol_baseline_window,
            self._volume_window,
        ) + 8
        self.max_lookback = self._min_bars + self._max_holding_bars + 16

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        close = market.close_at(index)
        ema_fast = market.ema("close", self._fast_ema)
        ema_slow = market.ema("close", self._slow_ema)
        atr = market.atr(self._atr_window)
        adx, _, _ = adx_bundle(market, self._adx_window)
        channel_high = market.rolling_max("high", self._channel_window, shift=1)
        channel_low = market.rolling_min("low", self._channel_window, shift=1)
        vol_z = volume_z(market, self._volume_window)
        realized_vol = self._realized_vol(market)
        baseline_vol = self._realized_vol_baseline(market, realized_vol)
        band_percentile = self._bollinger_bandwidth_percentile(market)

        if not finite_at(index, ema_fast, ema_slow, atr, adx, channel_high, channel_low, realized_vol, baseline_vol, band_percentile):
            return self._hold(timestamp, "indicator_not_ready", {})

        regime = regime_snapshot(market, index)
        metadata = {
            "strategy": "regime_adaptive_btc_trend_breakout",
            "close": close,
            "ema_fast": float(ema_fast[index]),
            "ema_slow": float(ema_slow[index]),
            "donchian_high": float(channel_high[index]),
            "donchian_low": float(channel_low[index]),
            "adx": float(adx[index]),
            "atr": float(atr[index]),
            "realized_vol_24h": float(realized_vol[index]),
            "median_realized_vol_30d": float(baseline_vol[index]),
            "volume_z": float(vol_z[index]),
            "bollinger_bandwidth_percentile": float(band_percentile[index]),
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
        }

        side = position_side(portfolio)
        if side == "LONG":
            exit_reason = self._long_exit(market, index, portfolio, close, ema_fast[index], atr[index])
            if exit_reason is not None:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, exit_reason, metadata)
            return self._hold(timestamp, "regime_trend_long_hold", metadata)

        if side == "SHORT":
            exit_reason = self._short_exit(market, index, portfolio, close, ema_fast[index], atr[index])
            if exit_reason is not None:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, exit_reason, metadata)
            return self._hold(timestamp, "regime_trend_short_hold", metadata)

        if self._flat_regime(adx[index], ema_fast[index], ema_slow[index], close, band_percentile[index]):
            return self._hold(timestamp, "flat_regime_block", metadata)

        sizing = target_fraction_from_risk(
            close=close,
            atr=float(atr[index]),
            stop_atr=self._trailing_stop_atr,
            risk_per_trade=self._risk_per_trade,
            max_notional_fraction=self._max_notional_fraction,
            vol_percentile=float(realized_vol_percentile(market, 24, self._realized_vol_baseline_window)[index]),
        )
        metadata["target_notional_fraction"] = sizing[0]
        metadata["stop_loss_pct"] = sizing[1]
        metadata["take_profit_pct"] = max(sizing[2], sizing[1] * 2.0)

        long_ok = (
            ema_fast[index] > ema_slow[index]
            and close > channel_high[index]
            and adx[index] > self._min_adx
            and realized_vol[index] > baseline_vol[index]
            and vol_z[index] > self._min_volume_z
        )
        short_ok = (
            ema_fast[index] < ema_slow[index]
            and close < channel_low[index]
            and adx[index] > self._min_adx
            and realized_vol[index] > baseline_vol[index]
            and vol_z[index] > self._min_volume_z
        )

        if long_ok:
            confidence = min(1.0, 0.55 + 0.07 * min(4.0, (adx[index] - self._min_adx) / 8.0) + 0.04 * max(0.0, vol_z[index]))
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "regime_adaptive_btc_trend_breakout_long", metadata)

        if short_ok:
            confidence = min(1.0, 0.55 + 0.07 * min(4.0, (adx[index] - self._min_adx) / 8.0) + 0.04 * max(0.0, vol_z[index]))
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "regime_adaptive_btc_trend_breakout_short", metadata)

        return self._hold(timestamp, "no_regime_trend_breakout", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _long_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, ema_fast: float, atr: float) -> str | None:
        if close < ema_fast:
            return "regime_trend_long_ema_invalidation"
        bars = bars_since_entry(portfolio)
        if bars >= self._max_holding_bars:
            return "regime_trend_long_time_stop"
        if bars > 0:
            start = max(0, index + 1 - bars)
            peak_close = float(np.nanmax(market.arrays.close[start:index + 1]))
            if close < peak_close - self._trailing_stop_atr * atr:
                return "regime_trend_long_atr_trailing_stop"
            entry_price = float(getattr(portfolio, "entry_price", 0.0) or 0.0)
            if entry_price > 0.0 and peak_close - close > self._drawdown_stop_atr * atr:
                return "regime_trend_long_peak_giveback_stop"
        return None

    def _short_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, ema_fast: float, atr: float) -> str | None:
        if close > ema_fast:
            return "regime_trend_short_ema_invalidation"
        bars = bars_since_entry(portfolio)
        if bars >= self._max_holding_bars:
            return "regime_trend_short_time_stop"
        if bars > 0:
            start = max(0, index + 1 - bars)
            trough_close = float(np.nanmin(market.arrays.close[start:index + 1]))
            if close > trough_close + self._trailing_stop_atr * atr:
                return "regime_trend_short_atr_trailing_stop"
            entry_price = float(getattr(portfolio, "entry_price", 0.0) or 0.0)
            if entry_price > 0.0 and close - trough_close > self._drawdown_stop_atr * atr:
                return "regime_trend_short_peak_giveback_stop"
        return None

    def _flat_regime(self, adx: float, ema_fast: float, ema_slow: float, close: float, bandwidth_percentile: float) -> bool:
        ema_spread = abs(ema_fast / safe_denominator(ema_slow, close) - 1.0)
        return bool(adx < self._flat_adx and bandwidth_percentile <= self._low_bandwidth_percentile and ema_spread < self._max_ema_spread_for_flat)

    def _realized_vol(self, market: MarketDataView) -> np.ndarray:
        key = f"regime_trend:rv:{self._realized_vol_window}"
        if key not in market._cache:
            market._cache[key] = pd.Series(market.log_diff("close", 1)).rolling(self._realized_vol_window).std().to_numpy(dtype=np.float64)
        return market._cache[key]

    def _realized_vol_baseline(self, market: MarketDataView, realized_vol: np.ndarray) -> np.ndarray:
        key = f"regime_trend:rv_median:{self._realized_vol_baseline_window}"
        if key not in market._cache:
            market._cache[key] = pd.Series(realized_vol).rolling(self._realized_vol_baseline_window).median().to_numpy(dtype=np.float64)
        return market._cache[key]

    def _bollinger_bandwidth_percentile(self, market: MarketDataView) -> np.ndarray:
        key = "regime_trend:bb_width_percentile:20:240"
        if key not in market._cache:
            middle = market.rolling_mean("close", 20, min_periods=20)
            std = market.rolling_std("close", 20, ddof=0, min_periods=20)
            width = (4.0 * std) / np.where(np.abs(middle) <= 1e-12, np.nan, middle)
            market._cache[key] = pd.Series(width).rolling(240).apply(self._last_percent_rank, raw=True).to_numpy(dtype=np.float64)
        return market._cache[key]

    @staticmethod
    def _last_percent_rank(values: np.ndarray) -> float:
        clean = values[np.isfinite(values)]
        if clean.size <= 1:
            return np.nan
        return float(100.0 * np.sum(clean[:-1] <= clean[-1]) / (clean.size - 1))

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
