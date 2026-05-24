from __future__ import annotations

from typing import Any

import pandas as pd

from trading_system.strategy.signals import Signal, SignalType


class MeanReversionStrategy:
    """
    Mean reversion strategy with long and short support.

    Rules:
    - z <= -entry_z_score -> LONG
    - z >=  entry_z_score -> SHORT
    - if position exists and abs(z) <= exit_z_score -> EXIT
    - otherwise HOLD
    """

    def __init__(
        self,
        symbol: str,
        lookback: int,
        entry_z_score: float,
        exit_z_score: float,
    ) -> None:
        self._symbol = symbol
        self._lookback = int(lookback)
        self._entry_z_score = float(entry_z_score)
        self._exit_z_score = float(exit_z_score)

    def generate_signal(
        self,
        market_window: pd.DataFrame,
        portfolio: Any,
    ) -> Signal:
        if len(market_window) < self._lookback:
            timestamp = (
                pd.Timestamp(market_window.iloc[-1]["timestamp"])
                if not market_window.empty
                else pd.Timestamp.utcnow()
            )
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="not_enough_history",
                metadata={},
            )

        current_bar = market_window.iloc[-1]
        timestamp = pd.Timestamp(current_bar["timestamp"])
        current_close = float(current_bar["close"])

        closes = market_window["close"].tail(self._lookback).astype(float)
        rolling_mean = float(closes.mean())
        rolling_std = float(closes.std(ddof=0))

        if rolling_std <= 0:
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="zero_volatility",
                metadata={
                    "close": current_close,
                    "rolling_mean": rolling_mean,
                    "rolling_std": rolling_std,
                    "z_score": 0.0,
                },
            )

        z_score = (current_close - rolling_mean) / rolling_std

        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))

        metadata = {
            "close": current_close,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "z_score": z_score,
        }

        if position_quantity != 0 and abs(z_score) <= self._exit_z_score:
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.EXIT,
                confidence=abs(z_score),
                reason="mean_reversion_exit",
                metadata=metadata,
            )

        if position_quantity == 0 and z_score <= -self._entry_z_score:
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.LONG,
                confidence=abs(z_score),
                reason="long_entry_z_score_reached",
                metadata=metadata,
            )

        if position_quantity == 0 and z_score >= self._entry_z_score:
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.SHORT,
                confidence=abs(z_score),
                reason="short_entry_z_score_reached",
                metadata=metadata,
            )

        return Signal(
            timestamp=timestamp,
            symbol=self._symbol,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            reason="no_edge",
            metadata=metadata,
        )