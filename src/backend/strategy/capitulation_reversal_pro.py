from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import (
    adx_bundle,
    bars_since_entry,
    bollinger_middle,
    connors_rsi,
    finite_at,
    kama,
    position_side,
    range_z,
    regime_snapshot,
    relative_volume,
    robust_return_z,
    rolling_vwap,
    safe_denominator,
    target_fraction_from_risk,
    realized_vol_percentile,
)
from backend.strategy.signals import Signal, SignalType


class CapitulationReversalProStrategy:
    def __init__(
        self,
        symbol: str,
        shock_window: int = 168,
        vwap_window: int = 24,
        atr_window: int = 14,
        kama_fast: int = 48,
        kama_slow: int = 144,
        donchian_window: int = 72,
        robust_z_threshold: float = 2.3,
        min_range_z: float = 1.6,
        min_rvol: float = 1.8,
        min_vwap_dev_atr: float = 1.8,
        long_connors_max: float = 15.0,
        short_connors_min: float = 85.0,
        min_wick_ratio: float = 0.35,
        long_clv_min: float = 0.45,
        short_clv_max: float = 0.55,
        stop_atr_buffer: float = 0.35,
        hard_stop_atr: float = 1.25,
        risk_per_trade: float = 0.0035,
        max_notional_fraction: float = 1.0,
        max_holding_bars: int = 8,
        require_confirmation: bool = True,
    ) -> None:
        self._symbol = symbol
        self._shock_window = int(shock_window)
        self._vwap_window = int(vwap_window)
        self._atr_window = int(atr_window)
        self._kama_fast = int(kama_fast)
        self._kama_slow = int(kama_slow)
        self._donchian_window = int(donchian_window)
        self._robust_z_threshold = float(robust_z_threshold)
        self._min_range_z = float(min_range_z)
        self._min_rvol = float(min_rvol)
        self._min_vwap_dev_atr = float(min_vwap_dev_atr)
        self._long_connors_max = float(long_connors_max)
        self._short_connors_min = float(short_connors_min)
        self._min_wick_ratio = float(min_wick_ratio)
        self._long_clv_min = float(long_clv_min)
        self._short_clv_max = float(short_clv_max)
        self._stop_atr_buffer = float(stop_atr_buffer)
        self._hard_stop_atr = float(hard_stop_atr)
        self._risk_per_trade = float(risk_per_trade)
        self._max_notional_fraction = float(max_notional_fraction)
        self._max_holding_bars = int(max_holding_bars)
        self._require_confirmation = bool(require_confirmation)
        self._min_bars = max(self._shock_window, self._kama_slow, self._donchian_window, 100) + 4
        self.max_lookback = self._min_bars + self._max_holding_bars + 16

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        side = position_side(portfolio)
        atr = market.atr(self._atr_window)
        fast = kama(market, self._kama_fast)
        middle = bollinger_middle(market, 20)
        vwap = rolling_vwap(market, self._vwap_window)

        if not finite_at(index, atr, fast, middle, vwap):
            return self._hold(timestamp, "indicator_not_ready", {})

        close = market.close_at(index)
        metadata = self._metadata_at(market, index)

        if side == "LONG":
            if self._long_exit(market, index, portfolio, close, vwap[index], fast[index], middle[index]):
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "capitulation_reversal_long_exit", metadata)
            return self._hold(timestamp, "capitulation_reversal_long_hold", metadata)

        if side == "SHORT":
            if self._short_exit(market, index, portfolio, close, vwap[index], fast[index], middle[index]):
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "capitulation_reversal_short_exit", metadata)
            return self._hold(timestamp, "capitulation_reversal_short_hold", metadata)

        signal_index = index - 1 if self._require_confirmation else index
        if signal_index < self._min_bars:
            return self._hold(timestamp, "waiting_for_confirmation_context", metadata)

        long_candidate = self._long_shock(market, signal_index)
        short_candidate = self._short_shock(market, signal_index)
        if self._require_confirmation:
            long_candidate = long_candidate and self._long_confirmation(market, index, signal_index)
            short_candidate = short_candidate and self._short_confirmation(market, index, signal_index)

        sizing = target_fraction_from_risk(
            close=close,
            atr=float(atr[index]),
            stop_atr=self._hard_stop_atr,
            risk_per_trade=self._risk_per_trade,
            max_notional_fraction=self._max_notional_fraction,
            vol_percentile=float(realized_vol_percentile(market, 24, self._shock_window)[index]),
        )
        metadata["target_notional_fraction"] = sizing[0]
        metadata["stop_loss_pct"] = sizing[1]
        metadata["take_profit_pct"] = max(sizing[2], sizing[1] * 1.6)

        if long_candidate and not self._long_trend_blocked(market, index):
            shock_low = market.low_at(signal_index)
            stop_price = shock_low - self._stop_atr_buffer * float(atr[signal_index])
            metadata["shock_candle_index"] = int(signal_index)
            metadata["shock_low"] = float(shock_low)
            metadata["structural_stop_price"] = float(stop_price)
            metadata["stop_loss_pct"] = max(metadata["stop_loss_pct"], (close - stop_price) / safe_denominator(close, 1.0))
            confidence = self._confidence(metadata, long=True)
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "capitulation_reversal_pro_long", metadata)

        if short_candidate and not self._short_trend_blocked(market, index):
            shock_high = market.high_at(signal_index)
            stop_price = shock_high + self._stop_atr_buffer * float(atr[signal_index])
            metadata["shock_candle_index"] = int(signal_index)
            metadata["shock_high"] = float(shock_high)
            metadata["structural_stop_price"] = float(stop_price)
            metadata["stop_loss_pct"] = max(metadata["stop_loss_pct"], (stop_price - close) / safe_denominator(close, 1.0))
            confidence = self._confidence(metadata, long=False)
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "capitulation_reversal_pro_short", metadata)

        return self._hold(timestamp, "no_capitulation_reversal_edge", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _metadata_at(self, market: MarketDataView, index: int) -> dict[str, Any]:
        candle_range = max(market.high_at(index) - market.low_at(index), 1e-12)
        body_high = max(market.open_at(index), market.close_at(index))
        body_low = min(market.open_at(index), market.close_at(index))
        upper_wick_ratio = (market.high_at(index) - body_high) / candle_range
        lower_wick_ratio = (body_low - market.low_at(index)) / candle_range
        clv = (market.close_at(index) - market.low_at(index)) / candle_range
        atr = market.atr(self._atr_window)
        vwap = rolling_vwap(market, self._vwap_window)
        regime = regime_snapshot(market, index)
        return {
            "strategy": "capitulation_reversal_pro",
            "close": market.close_at(index),
            "robust_ret_z": float(robust_return_z(market, self._shock_window)[index]),
            "range_z": float(range_z(market, self._shock_window)[index]),
            "rvol": float(relative_volume(market, self._shock_window)[index]),
            "vwap_dev_atr": float((market.close_at(index) - vwap[index]) / safe_denominator(atr[index], np.nan)),
            "connors_rsi": float(connors_rsi(market)[index]),
            "lower_wick_ratio": float(lower_wick_ratio),
            "upper_wick_ratio": float(upper_wick_ratio),
            "clv": float(clv),
            "natr_percentile": float(realized_vol_percentile(market, 24, self._shock_window)[index]),
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
        }

    def _long_shock(self, market: MarketDataView, index: int) -> bool:
        values = self._metadata_at(market, index)
        return (
            values["robust_ret_z"] < -self._robust_z_threshold
            and values["range_z"] > self._min_range_z
            and values["rvol"] > self._min_rvol
            and values["vwap_dev_atr"] < -self._min_vwap_dev_atr
            and values["connors_rsi"] < self._long_connors_max
            and values["lower_wick_ratio"] > self._min_wick_ratio
            and values["clv"] > self._long_clv_min
            and values["regime"] != "rapid_downtrend"
        )

    def _short_shock(self, market: MarketDataView, index: int) -> bool:
        values = self._metadata_at(market, index)
        return (
            values["robust_ret_z"] > self._robust_z_threshold
            and values["range_z"] > self._min_range_z
            and values["rvol"] > self._min_rvol
            and values["vwap_dev_atr"] > self._min_vwap_dev_atr
            and values["connors_rsi"] > self._short_connors_min
            and values["upper_wick_ratio"] > self._min_wick_ratio
            and values["clv"] < self._short_clv_max
            and values["regime"] != "rapid_uptrend"
        )

    def _long_confirmation(self, market: MarketDataView, index: int, shock_index: int) -> bool:
        shock_high = market.high_at(shock_index)
        shock_mid = (market.high_at(shock_index) + market.low_at(shock_index)) * 0.5
        return market.high_at(index) > shock_high or market.close_at(index) > shock_mid

    def _short_confirmation(self, market: MarketDataView, index: int, shock_index: int) -> bool:
        shock_low = market.low_at(shock_index)
        shock_mid = (market.high_at(shock_index) + market.low_at(shock_index)) * 0.5
        return market.low_at(index) < shock_low or market.close_at(index) < shock_mid

    def _long_trend_blocked(self, market: MarketDataView, index: int) -> bool:
        fast = kama(market, self._kama_fast)
        slow = kama(market, self._kama_slow)
        adx, _, _ = adx_bundle(market, 14)
        low_channel = market.rolling_min("close", self._donchian_window, shift=1)
        return bool(fast[index] < slow[index] and adx[index] > 30.0 and market.close_at(index) < low_channel[index])

    def _short_trend_blocked(self, market: MarketDataView, index: int) -> bool:
        fast = kama(market, self._kama_fast)
        slow = kama(market, self._kama_slow)
        adx, _, _ = adx_bundle(market, 14)
        high_channel = market.rolling_max("close", self._donchian_window, shift=1)
        return bool(fast[index] > slow[index] and adx[index] > 30.0 and market.close_at(index) > high_channel[index])

    def _long_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, vwap: float, fast: float, middle: float) -> bool:
        if close >= vwap or close >= fast or close >= middle:
            return True
        return bars_since_entry(portfolio) >= self._max_holding_bars

    def _short_exit(self, market: MarketDataView, index: int, portfolio: Any, close: float, vwap: float, fast: float, middle: float) -> bool:
        if close <= vwap or close <= fast or close <= middle:
            return True
        return bars_since_entry(portfolio) >= self._max_holding_bars

    @staticmethod
    def _confidence(metadata: dict[str, Any], long: bool) -> float:
        shock = abs(float(metadata.get("robust_ret_z", 0.0)))
        stretch = abs(float(metadata.get("vwap_dev_atr", 0.0)))
        wick_key = "lower_wick_ratio" if long else "upper_wick_ratio"
        wick = float(metadata.get(wick_key, 0.0))
        return min(1.0, 0.52 + 0.06 * min(4.0, shock) + 0.05 * min(3.0, stretch) + 0.20 * wick)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
