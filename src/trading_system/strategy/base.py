from abc import ABC, abstractmethod
from typing import Protocol

import pandas as pd

from trading_system.strategy.signals import Signal


class PortfolioView(Protocol):
    symbol: str
    position_quantity: float
    equity: float
    cash: float


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(
        self,
        market_window: pd.DataFrame,
        portfolio: PortfolioView,
    ) -> Signal:
        """Generate a trading signal using only market data available up to current bar."""