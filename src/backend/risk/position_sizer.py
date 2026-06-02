from backend.core.config import RiskConfig
from backend.execution.orders import OrderIntent, OrderSide
from backend.strategy.signals import Signal, SignalType


class PositionSizer:
    def __init__(self, config: RiskConfig) -> None:
        self._config = config

    def size_order(
        self,
        signal: Signal,
        current_price: float,
        equity: float,
        cash: float,
        current_position_quantity: float,
    ) -> OrderIntent | None:
        if signal.signal_type == SignalType.HOLD:
            return None

        if signal.signal_type == SignalType.EXIT:
            if current_position_quantity <= 0:
                return None

            return OrderIntent(
                symbol=signal.symbol,
                side=OrderSide.SELL,
                quantity=current_position_quantity,
                signal_time=signal.timestamp,
                reason=signal.reason,
            )

        if signal.signal_type == SignalType.LONG:
            if current_position_quantity > 0:
                return None

            if current_price <= 0:
                return None

            risk_amount = equity * self._config.risk_per_trade_pct
            stop_distance = current_price * self._config.stop_distance_pct

            if stop_distance <= 0:
                return None

            raw_quantity = risk_amount / stop_distance
            raw_notional = raw_quantity * current_price

            max_position_notional = equity * self._config.max_position_notional_pct
            cash_buffer = equity * self._config.min_cash_pct
            available_cash_after_buffer = max(0.0, cash - cash_buffer)

            final_notional = min(
                raw_notional,
                max_position_notional,
                available_cash_after_buffer,
            )

            if final_notional < self._config.min_order_notional:
                return None

            quantity = final_notional / current_price

            return OrderIntent(
                symbol=signal.symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                signal_time=signal.timestamp,
                reason=signal.reason,
            )

        return None
