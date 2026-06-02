from dataclasses import dataclass, field

import pandas as pd

from backend.execution.fills import Fill
from backend.execution.orders import OrderSide
from backend.portfolio.position import Position
from backend.utils.ids import new_id


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    net_pnl: float
    fees: float
    slippage_cost: float
    return_pct: float
    exit_reason: str


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    cash: float
    position_quantity: float
    position_market_value: float
    equity: float
    drawdown: float


@dataclass
class Portfolio:
    symbol: str
    initial_capital: float
    cash: float = field(init=False)
    position: Position = field(init=False)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    _open_trade_entry_time: pd.Timestamp | None = None
    _open_trade_entry_price: float = 0.0
    _open_trade_entry_fee: float = 0.0
    _open_trade_entry_slippage: float = 0.0
    _peak_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self.cash = self.initial_capital
        self.position = Position(symbol=self.symbol)
        self._peak_equity = self.initial_capital

    @property
    def position_quantity(self) -> float:
        return self.position.quantity

    @property
    def equity(self) -> float:
        if not self.equity_curve:
            return self.initial_capital
        return self.equity_curve[-1].equity

    def apply_fill(self, fill: Fill) -> None:
        self.fees_paid += fill.fee
        self.slippage_cost += fill.slippage_cost

        if fill.side == OrderSide.BUY:
            total_cost = fill.notional + fill.fee
            if total_cost > self.cash:
                return

            self.cash -= total_cost
            self.position.quantity += fill.quantity
            self.position.average_entry_price = fill.fill_price

            self._open_trade_entry_time = fill.fill_time
            self._open_trade_entry_price = fill.fill_price
            self._open_trade_entry_fee = fill.fee
            self._open_trade_entry_slippage = fill.slippage_cost
            return

        if fill.side == OrderSide.SELL:
            sell_quantity = min(fill.quantity, self.position.quantity)
            if sell_quantity <= 0:
                return

            gross_pnl = sell_quantity * (fill.fill_price - self.position.average_entry_price)
            total_fees = self._open_trade_entry_fee + fill.fee
            total_slippage = self._open_trade_entry_slippage + fill.slippage_cost
            net_pnl = gross_pnl - total_fees

            self.cash += sell_quantity * fill.fill_price - fill.fee
            self.realized_pnl += net_pnl

            entry_price = self._open_trade_entry_price
            return_pct = (fill.fill_price / entry_price - 1.0) if entry_price > 0 else 0.0

            if self._open_trade_entry_time is not None:
                self.trades.append(
                    TradeRecord(
                        trade_id=new_id("trade"),
                        symbol=fill.symbol,
                        entry_time=self._open_trade_entry_time,
                        exit_time=fill.fill_time,
                        entry_price=entry_price,
                        exit_price=fill.fill_price,
                        quantity=sell_quantity,
                        gross_pnl=gross_pnl,
                        net_pnl=net_pnl,
                        fees=total_fees,
                        slippage_cost=total_slippage,
                        return_pct=return_pct,
                        exit_reason=fill.reason,
                    )
                )

            self.position.quantity -= sell_quantity

            if self.position.quantity <= 1e-12:
                self.position.quantity = 0.0
                self.position.average_entry_price = 0.0
                self._open_trade_entry_time = None
                self._open_trade_entry_price = 0.0
                self._open_trade_entry_fee = 0.0
                self._open_trade_entry_slippage = 0.0

    def mark_to_market(self, timestamp: pd.Timestamp, close_price: float) -> None:
        market_value = self.position.quantity * close_price
        equity = self.cash + market_value

        self._peak_equity = max(self._peak_equity, equity)
        drawdown = equity / self._peak_equity - 1.0 if self._peak_equity > 0 else 0.0

        self.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                cash=self.cash,
                position_quantity=self.position.quantity,
                position_market_value=market_value,
                equity=equity,
                drawdown=drawdown,
            )
        )
