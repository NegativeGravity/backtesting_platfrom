import pandas as pd

from backend.core.config import ExecutionConfig
from backend.execution.fills import Fill
from backend.execution.orders import OrderIntent, OrderSide


class ExecutionSimulator:
    def __init__(self, config: ExecutionConfig) -> None:
        if config.execution_timing != "next_bar_open":
            raise ValueError("Only next_bar_open execution is supported in Phase 1.")
        self._config = config

    def execute_at_open(
        self,
        order: OrderIntent,
        execution_bar: pd.Series,
        available_cash: float,
    ) -> Fill | None:
        open_price = float(execution_bar["open"])
        fill_time = execution_bar["timestamp"]

        slippage_per_unit = open_price * self._config.slippage_bps / 10_000.0

        if order.side == OrderSide.BUY:
            fill_price = open_price + slippage_per_unit
            quantity = order.quantity
            estimated_total_cost = quantity * fill_price * (1.0 + self._config.fee_rate)

            if estimated_total_cost > available_cash:
                quantity = available_cash / (fill_price * (1.0 + self._config.fee_rate))

            if quantity <= 0:
                return None

            fee = quantity * fill_price * self._config.fee_rate
            slippage_cost = quantity * slippage_per_unit

        else:
            fill_price = max(0.0, open_price - slippage_per_unit)
            quantity = order.quantity

            if quantity <= 0:
                return None

            fee = quantity * fill_price * self._config.fee_rate
            slippage_cost = quantity * slippage_per_unit

        return Fill(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            fill_time=fill_time,
            fill_price=fill_price,
            fee=fee,
            slippage_cost=slippage_cost,
            reason=order.reason,
        )