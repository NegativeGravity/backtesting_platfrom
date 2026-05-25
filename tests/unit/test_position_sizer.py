import pandas as pd

from backend.core.config import RiskConfig
from backend.risk.position_sizer import PositionSizer
from backend.strategy.signals import Signal, SignalType


def test_position_sizer_respects_max_notional() -> None:
    config = RiskConfig(
        risk_per_trade_pct=0.01,
        max_position_notional_pct=0.5,
        min_cash_pct=0.05,
        stop_distance_pct=0.02,
        min_order_notional=10.0,
    )

    sizer = PositionSizer(config)

    signal = Signal(
        timestamp=pd.Timestamp("2023-01-01", tz="UTC"),
        symbol="BTCUSDT",
        signal_type=SignalType.LONG,
        confidence=1.0,
        reason="test",
    )

    order = sizer.size_order(
        signal=signal,
        current_price=100.0,
        equity=10_000.0,
        cash=10_000.0,
        current_position_quantity=0.0,
    )

    assert order is not None
    assert order.quantity * 100.0 <= 5_000.0