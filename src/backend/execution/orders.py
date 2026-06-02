from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class OrderSide(str, Enum):
    """
    Backward-compatible side for the old backtest engine.

    BUY  = old long entry
    SELL = old long exit
    """

    BUY = "BUY"
    SELL = "SELL"


class OrderAction(str, Enum):
    """
    Explicit live-trading order action.

    BUY_TO_OPEN:
        Open a long position.

    SELL_TO_CLOSE:
        Close a long position.

    SELL_TO_OPEN:
        Open a short position.

    BUY_TO_CLOSE:
        Close a short position.
    """

    BUY_TO_OPEN = "BUY_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"


@dataclass(frozen=True)
class OrderIntent:
    """
    Backward-compatible order intent for the old backtest engine.
    """

    symbol: str
    side: OrderSide
    quantity: float
    signal_time: pd.Timestamp
    reason: str


@dataclass(frozen=True)
class LiveOrderIntent:
    symbol: str
    action: OrderAction
    quantity: float
    signal_time: pd.Timestamp
    reason: str
