from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class MeanReversionStrategy:
    """
    Mean reversion strategy with long and short support.

    Fast path uses MarketDataView + bar index. No per-bar data.iloc[:i+1]
    slices and no repeated rolling calculations inside the hot loop.
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
        self.max_lookback = self._lookback

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._lookback:
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "not_enough_history", {})

        current_close = market.close_at(index)
        rolling_mean = float(market.rolling_mean("close", self._lookback, min_periods=self._lookback)[index])
        rolling_std = float(market.rolling_std("close", self._lookback, ddof=0, min_periods=self._lookback)[index])

        if not np.isfinite(rolling_mean) or not np.isfinite(rolling_std) or rolling_std <= 0:
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
            return Signal(timestamp, self._symbol, SignalType.EXIT, abs(z_score), "mean_reversion_exit", metadata)
        if position_quantity == 0 and z_score <= -self._entry_z_score:
            return Signal(timestamp, self._symbol, SignalType.LONG, abs(z_score), "long_entry_z_score_reached", metadata)
        if position_quantity == 0 and z_score >= self._entry_z_score:
            return Signal(timestamp, self._symbol, SignalType.SHORT, abs(z_score), "short_entry_z_score_reached", metadata)
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "no_edge", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)