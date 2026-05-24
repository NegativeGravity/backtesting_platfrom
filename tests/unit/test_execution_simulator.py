import pandas as pd

from trading_system.core.config import ExecutionConfig
from trading_system.execution.orders import OrderIntent, OrderSide
from trading_system.execution.simulator import ExecutionSimulator


def test_buy_executes_at_next_bar_open_plus_slippage() -> None:
    config = ExecutionConfig(
        execution_timing="next_bar_open",
        fee_rate=0.001,
        slippage_bps=10.0,
        allow_fractional_quantity=True,
        pessimistic_intrabar_policy=True,
    )

    simulator = ExecutionSimulator(config)

    order = OrderIntent(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=1.0,
        signal_time=pd.Timestamp("2023-01-01", tz="UTC"),
        reason="test",
    )

    bar = pd.Series(
        {
            "timestamp": pd.Timestamp("2023-01-01 01:00:00", tz="UTC"),
            "open": 100.0,
        }
    )

    fill = simulator.execute_at_open(order, bar, available_cash=1000.0)

    assert fill is not None
    assert fill.fill_price == 100.1
    assert fill.fee == fill.quantity * fill.fill_price * 0.001