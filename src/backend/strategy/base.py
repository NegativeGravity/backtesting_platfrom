from abc import ABC, abstractmethod
from typing import Any, Protocol

import pandas as pd

from backend.strategy.signals import Signal


class PortfolioView(Protocol):
    symbol: str
    position_quantity: float
    equity: float
    cash: float
    position_side: str | None


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(
        self,
        market_window: pd.DataFrame,
        portfolio: PortfolioView,
    ) -> Signal:
        """Generate a signal using a bounded legacy DataFrame window."""

    def generate_signal_at(
        self,
        market: Any,
        index: int,
        portfolio: PortfolioView,
    ) -> Signal:
        """Fast-path hook for array/index based engines.

        Subclasses that do not override this still avoid the old O(n²) full-window
        pattern because MarketDataView.window(...) returns a bounded tail window.
        """
        lookback = int(getattr(self, "max_lookback", getattr(self, "_min_bars", 1024)))
        return self.generate_signal(market_window=market.window(index, lookback + 16), portfolio=portfolio)