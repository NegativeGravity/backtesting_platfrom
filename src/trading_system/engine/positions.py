from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class OpenPosition:
    position_id: str
    robot_id: str
    robot_name: str
    worker_id: str
    strategy: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_time: pd.Timestamp
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_fee: float
    entry_slippage_cost: float
    reason: str

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.entry_price)

    @property
    def current_notional(self) -> float:
        return abs(self.quantity * self.current_price)

    @property
    def gross_unrealized_pnl(self) -> float:
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity

        return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl(self) -> float:
        return self.gross_unrealized_pnl - self.entry_fee - self.entry_slippage_cost

    @property
    def unrealized_return_pct(self) -> float:
        if self.notional <= 0:
            return 0.0

        return self.unrealized_pnl / self.notional

    def update_price(self, price: float) -> None:
        self.current_price = float(price)

    def to_event_payload(self) -> dict:
        payload = asdict(self)
        payload["side"] = self.side.value
        payload["entry_time"] = self.entry_time.isoformat()
        payload["notional"] = self.notional
        payload["current_notional"] = self.current_notional
        payload["gross_unrealized_pnl"] = self.gross_unrealized_pnl
        payload["unrealized_pnl"] = self.unrealized_pnl
        payload["unrealized_return_pct"] = self.unrealized_return_pct
        return payload


@dataclass
class ClosedPosition:
    position_id: str
    robot_id: str
    robot_name: str
    worker_id: str
    strategy: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    entry_fee: float
    exit_fee: float
    entry_slippage_cost: float
    exit_slippage_cost: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str

    def to_event_payload(self) -> dict:
        payload = asdict(self)
        payload["side"] = self.side.value
        payload["entry_time"] = self.entry_time.isoformat()
        payload["exit_time"] = self.exit_time.isoformat()
        return payload

    def to_trade_record(self) -> dict:
        return {
            "trade_id": self.position_id,
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "fees": self.entry_fee + self.exit_fee,
            "slippage_cost": self.entry_slippage_cost + self.exit_slippage_cost,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "worker_id": self.worker_id,
            "strategy": self.strategy,
            "side": self.side.value,
        }


def position_key(worker_id: str, symbol: str) -> str:
    return f"{worker_id}:{symbol}"


def calculate_gross_pnl(
    side: PositionSide,
    entry_price: float,
    exit_price: float,
    quantity: float,
) -> float:
    if side == PositionSide.LONG:
        return (exit_price - entry_price) * quantity

    return (entry_price - exit_price) * quantity


def build_closed_position(
    position: OpenPosition,
    exit_time: pd.Timestamp,
    exit_price: float,
    exit_fee: float,
    exit_slippage_cost: float,
    exit_reason: str,
) -> ClosedPosition:
    gross_pnl = calculate_gross_pnl(
        side=position.side,
        entry_price=position.entry_price,
        exit_price=exit_price,
        quantity=position.quantity,
    )
    total_costs = (
        position.entry_fee
        + exit_fee
        + position.entry_slippage_cost
        + exit_slippage_cost
    )
    net_pnl = gross_pnl - total_costs
    return_pct = 0.0 if position.notional <= 0 else net_pnl / position.notional

    return ClosedPosition(
        position_id=position.position_id,
        robot_id=position.robot_id,
        robot_name=position.robot_name,
        worker_id=position.worker_id,
        strategy=position.strategy,
        symbol=position.symbol,
        side=position.side,
        quantity=position.quantity,
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry_price=position.entry_price,
        exit_price=exit_price,
        stop_loss=position.stop_loss,
        take_profit=position.take_profit,
        entry_fee=position.entry_fee,
        exit_fee=exit_fee,
        entry_slippage_cost=position.entry_slippage_cost,
        exit_slippage_cost=exit_slippage_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_pct=return_pct,
        exit_reason=exit_reason,
    )