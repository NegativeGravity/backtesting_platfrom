from dataclasses import dataclass

import pandas as pd

from backend.execution.orders import OrderSide


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: OrderSide
    quantity: float
    fill_time: pd.Timestamp
    fill_price: float
    fee: float
    slippage_cost: float
    reason: str

    @property
    def notional(self) -> float:
        return self.quantity * self.fill_price