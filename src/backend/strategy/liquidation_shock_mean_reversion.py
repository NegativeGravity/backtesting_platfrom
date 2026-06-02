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
    rolling_vwap,
    safe_denominator,
    target_fraction_from_risk,
    volume_z,
    realized_vol_percentile,
)
from backend.strategy.signals import Signal, SignalType


class LiquidationShockMeanReversionStrategy:
    def __init__(
        self,
        symbol: str,
        return_window: int = 72,
        volume_window: int = 72,
        atr_window: int = 14,
        ema_fast: int = 20,
        ema_mid: int = 50,
        ema_slow: int = 200,
        vwap_window: int = 24,
        shock_sigma: float = 2.5,
        min_volume_z: float = 2.0,
        min_ema_distance_atr: float = 1.25,
        long_rsi_max: float = 25.0,
        short_rsi_min: float = 75.0,
        long_clv_min: float = 0.35,
        short_clv_max: float = 0.65,
        hard_stop_atr: float = 1.35,
        wick_stop_buffer_atr: float = 0.25,
        risk_per_trade: float = 0.004,
        max_notional_fraction: float = 1.0,
        max_holding_bars: int = 6,
    ) -> None:
        self._symbol = symbol
        self._return_window = int(return_window)
        self._volume_window = int(volume_window)
        self._atr_window = int(atr_window)
        self._ema_fast = int(ema_fast)
        self._ema_mid = int(ema_mid)
        self._ema_slow = int(ema_slow)
        self._vwap_window = int(vwap_window)
        self._shock_sigma = float(shock_sigma)
        self._min_volume_z = float(min_volume_z)
        self._min_ema_distance_atr = float(min_ema_distance_atr)
        self._long_rsi_max = float(long_rsi_max)
        self._short_rsi_min = float(short_rsi_min)
        self._long_clv_min = float(long_clv_min)
        self._short_clv_max = float(short_clv_max)
        self._hard_stop_atr = float(hard_stop_atr)
        self._wick_stop_buffer_atr = float(wick_stop_buffer_atr)
        self._risk_per_trade = float(risk_per_trade)
        self._max_notional_fraction = float(max_notional_fraction)
        self._max_holding_bars = int(max_holding_bars)
        self._min_bars = max(self._return_window, self._volume_window, self._ema_slow, 168) + 8
        self.max_lookback = self._min_bars + self._max_holding_bars + 16

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        close = market.close_at(index)
        atr = market.atr(self._atr_window)
        ema20 = market.ema("close", self._ema_fast)
        ema50 = market.ema("close", self._ema_mid)
        ema200 = market.ema("close", self._ema_slow)
        vwap = rolling_vwap(market, self._vwap_window)
        returns = market.log_diff("close", 1)
        return_std = self._return_std(market)
        rsi = self._rsi(market)
        vol_z = volume_z(market, self._volume_window)

        if not finite_at(index, atr, ema20, ema50, ema200, vwap, returns, return_std, rsi):
            return self._hold(timestamp, "indicator_not_ready", {})

        metadata = self._metadata_at(market, index, returns, return_std, rsi, vol_z, ema50, ema200, atr, vwap)
        side = position_side(portfolio)

        if side == "LONG":
            if self._long_exit(close, ema20[index], ema50[index], vwap[index], portfolio):
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "liquidation_shock_long_exit", metadata)
            return self._hold(timestamp, "liquidation_shock_long_hold", metadata)

        if side == "SHORT":
            if self._short_exit(close, ema20[index], ema50[index], vwap[index], portfolio):
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.70, "liquidation_shock_short_exit", metadata)
            return self._hold(timestamp, "liquidation_shock_short_hold", metadata)

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
        metadata["take_profit_pct"] = max(sizing[2], sizing[1] * 1.5)

        if self._long_entry(market, index, returns, return_std, vol_z, rsi, ema50, ema200, atr):
            shock_low = market.low_at(index)
            stop_price = shock_low - self._wick_stop_buffer_atr * float(atr[index])
            metadata["structural_stop_price"] = float(stop_price)
            metadata["stop_loss_pct"] = max(metadata["stop_loss_pct"], (close - stop_price) / safe_denominator(close, 1.0))
            confidence = self._confidence(metadata, long=True)
            return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "liquidation_shock_mean_reversion_long", metadata)

        if self._short_entry(market, index, returns, return_std, vol_z, rsi, ema50, ema200, atr):
            shock_high = market.high_at(index)
            stop_price = shock_high + self._wick_stop_buffer_atr * float(atr[index])
            metadata["structural_stop_price"] = float(stop_price)
            metadata["stop_loss_pct"] = max(metadata["stop_loss_pct"], (stop_price - close) / safe_denominator(close, 1.0))
            confidence = self._confidence(metadata, long=False)
            return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "liquidation_shock_mean_reversion_short", metadata)

        return self._hold(timestamp, "no_liquidation_shock_reversion", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _long_entry(
        self,
        market: MarketDataView,
        index: int,
        returns: np.ndarray,
        return_std: np.ndarray,
        vol_z: np.ndarray,
        rsi: np.ndarray,
        ema50: np.ndarray,
        ema200: np.ndarray,
        atr: np.ndarray,
    ) -> bool:
        clv = self._clv(market, index)
        ema_distance = (ema50[index] - market.low_at(index)) / safe_denominator(atr[index], np.nan)
        return bool(
            returns[index] < -self._shock_sigma * return_std[index]
            and vol_z[index] > self._min_volume_z
            and ema_distance > self._min_ema_distance_atr
            and rsi[index] < self._long_rsi_max
            and clv > self._long_clv_min
            and not self._long_trend_blocked(market, index, ema50, ema200)
        )

    def _short_entry(
        self,
        market: MarketDataView,
        index: int,
        returns: np.ndarray,
        return_std: np.ndarray,
        vol_z: np.ndarray,
        rsi: np.ndarray,
        ema50: np.ndarray,
        ema200: np.ndarray,
        atr: np.ndarray,
    ) -> bool:
        clv = self._clv(market, index)
        ema_distance = (market.high_at(index) - ema50[index]) / safe_denominator(atr[index], np.nan)
        return bool(
            returns[index] > self._shock_sigma * return_std[index]
            and vol_z[index] > self._min_volume_z
            and ema_distance > self._min_ema_distance_atr
            and rsi[index] > self._short_rsi_min
            and clv < self._short_clv_max
            and not self._short_trend_blocked(market, index, ema50, ema200)
        )

    def _long_exit(self, close: float, ema20: float, ema50: float, vwap: float, portfolio: Any) -> bool:
        return bool(close >= ema20 or close >= vwap or close >= ema50 or bars_since_entry(portfolio) >= self._max_holding_bars)

    def _short_exit(self, close: float, ema20: float, ema50: float, vwap: float, portfolio: Any) -> bool:
        return bool(close <= ema20 or close <= vwap or close <= ema50 or bars_since_entry(portfolio) >= self._max_holding_bars)

    def _long_trend_blocked(self, market: MarketDataView, index: int, ema50: np.ndarray, ema200: np.ndarray) -> bool:
        adx, _, _ = adx_bundle(market, 14)
        channel_low = market.rolling_min("low", 48, shift=1)
        return bool(ema50[index] < ema200[index] and adx[index] > 30.0 and market.close_at(index) < channel_low[index])

    def _short_trend_blocked(self, market: MarketDataView, index: int, ema50: np.ndarray, ema200: np.ndarray) -> bool:
        adx, _, _ = adx_bundle(market, 14)
        channel_high = market.rolling_max("high", 48, shift=1)
        return bool(ema50[index] > ema200[index] and adx[index] > 30.0 and market.close_at(index) > channel_high[index])

    def _metadata_at(
        self,
        market: MarketDataView,
        index: int,
        returns: np.ndarray,
        return_std: np.ndarray,
        rsi: np.ndarray,
        vol_z: np.ndarray,
        ema50: np.ndarray,
        ema200: np.ndarray,
        atr: np.ndarray,
        vwap: np.ndarray,
    ) -> dict[str, Any]:
        candle_range = max(market.high_at(index) - market.low_at(index), 1e-12)
        body_high = max(market.open_at(index), market.close_at(index))
        body_low = min(market.open_at(index), market.close_at(index))
        regime = regime_snapshot(market, index)
        return {
            "strategy": "liquidation_shock_mean_reversion",
            "close": market.close_at(index),
            "log_return_1h": float(returns[index]),
            "rolling_return_std": float(return_std[index]),
            "shock_sigma": float(returns[index] / safe_denominator(return_std[index], np.nan)),
            "volume_z": float(vol_z[index]),
            "rsi": float(rsi[index]),
            "ema50": float(ema50[index]),
            "ema200": float(ema200[index]),
            "atr": float(atr[index]),
            "vwap": float(vwap[index]),
            "vwap_dev_atr": float((market.close_at(index) - vwap[index]) / safe_denominator(atr[index], np.nan)),
            "clv": float(self._clv(market, index)),
            "upper_wick_ratio": float((market.high_at(index) - body_high) / candle_range),
            "lower_wick_ratio": float((body_low - market.low_at(index)) / candle_range),
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
        }

    def _return_std(self, market: MarketDataView) -> np.ndarray:
        key = f"liquidation:ret_std:{self._return_window}"
        if key not in market._cache:
            market._cache[key] = pd.Series(market.log_diff("close", 1)).rolling(self._return_window).std().to_numpy(dtype=np.float64)
        return market._cache[key]

    def _rsi(self, market: MarketDataView) -> np.ndarray:
        key = "liquidation:rsi:14"
        if key in market._cache:
            return market._cache[key]
        close = market.arrays.close
        delta = np.diff(close, prepend=np.nan)
        gain = np.where(delta > 0.0, delta, 0.0)
        loss = np.where(delta < 0.0, -delta, 0.0)
        average_gain = pd.Series(gain).ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy(dtype=np.float64)
        average_loss = pd.Series(loss).ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy(dtype=np.float64)
        rs = average_gain / np.where(np.abs(average_loss) <= 1e-12, np.nan, average_loss)
        output = 100.0 - 100.0 / (1.0 + rs)
        market._cache[key] = np.nan_to_num(output, nan=50.0, posinf=100.0, neginf=0.0)
        return market._cache[key]

    @staticmethod
    def _clv(market: MarketDataView, index: int) -> float:
        candle_range = market.high_at(index) - market.low_at(index)
        return float((market.close_at(index) - market.low_at(index)) / safe_denominator(candle_range, 1.0))

    @staticmethod
    def _confidence(metadata: dict[str, Any], long: bool) -> float:
        shock = abs(float(metadata.get("shock_sigma", 0.0) or 0.0))
        volume = max(0.0, float(metadata.get("volume_z", 0.0) or 0.0))
        wick = float(metadata.get("lower_wick_ratio" if long else "upper_wick_ratio", 0.0) or 0.0)
        return min(1.0, 0.50 + 0.06 * min(4.0, shock) + 0.03 * min(4.0, volume) + 0.18 * wick)

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
