from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from backend.backtest.benchmark import build_buy_and_hold_benchmark
from backend.backtest.metrics import calculate_metrics
from backend.backtest.report import BacktestReportWriter
from backend.core.config import AppConfig
from backend.core.time import timestamp_for_run_id
from backend.engine.positions import (
    ClosedPosition,
    OpenPosition,
    PositionSide,
    build_closed_position,
    position_key,
)
from backend.execution.orders import OrderAction
from backend.strategy.base import BaseStrategy
from backend.strategy.market_view import MarketDataView, call_strategy_signal
from backend.strategy.signals import Signal, SignalType
from backend.utils.ids import new_id

logger = logging.getLogger(__name__)
EventCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class BacktestWorkerSpec:
    worker_id: str
    strategy_name: str
    strategy: BaseStrategy
    model_artifact_path: str | None = None
    helformer_artifact_path: str | None = None
    use_helformer_forecast: bool = False


@dataclass(frozen=True)
class BacktestRobotSpec:
    robot_id: str
    display_name: str
    strategy_workers: list[BacktestWorkerSpec]


@dataclass(frozen=True)
class PendingOrder:
    robot_id: str
    robot_name: str
    worker_id: str
    strategy_name: str
    symbol: str
    action: OrderAction
    quantity: float
    signal_time: pd.Timestamp
    reason: str
    target_notional_fraction: float
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    cash: float
    position_quantity: float
    position_market_value: float
    equity: float
    drawdown: float
    open_position_count: int
    long_market_value: float
    short_market_value: float
    reserved_margin: float
    gross_unrealized_pnl: float
    accounting_model: str


@dataclass
class RobotRuntime:
    robot_id: str
    display_name: str
    workers: dict[str, BacktestWorkerSpec]
    cash: float
    peak_equity: float
    pending_orders: list[PendingOrder] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    closed_positions: list[ClosedPosition] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    signal_log: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    run_dir: Path
    summary: dict[str, Any]


class StrategyPortfolioView:
    """Small immutable-ish view passed to one strategy worker.

    Each strategy sees only its own position on the asset, while the robot equity and cash
    are shared. This prevents one strategy from accidentally closing another strategy's trade.
    """

    def __init__(
        self,
        symbol: str,
        position: OpenPosition | None,
        equity: float,
        cash: float,
        current_price: float | None = None,
        bars_since_entry: int = 0,
    ) -> None:
        self.symbol = symbol
        self.position_quantity = 0.0 if position is None else position.quantity
        self.position_side = None if position is None else position.side.value
        self.equity = equity
        self.cash = cash
        self.has_position = position is not None
        self.current_price = current_price
        self.entry_price = None if position is None else position.entry_price
        self.stop_loss = None if position is None else position.stop_loss
        self.take_profit = None if position is None else position.take_profit
        self.unrealized_pnl = None if position is None else position.unrealized_pnl
        self.unrealized_return_pct = None if position is None else position.unrealized_return_pct
        self.bars_since_entry = int(bars_since_entry)


class RobotBacktestEngine:

    def __init__(
        self,
        config: AppConfig,
        data: pd.DataFrame,
        robot_spec: BacktestRobotSpec,
        event_callback: EventCallback | None = None,
        capital_per_trade_fraction: float | None = None,
    ) -> None:
        if data.empty:
            raise ValueError("Backtest data is empty.")
        if not robot_spec.strategy_workers:
            raise ValueError("Robot must contain at least one strategy worker.")

        worker_ids = [worker.worker_id for worker in robot_spec.strategy_workers]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("Worker IDs must be unique inside a robot.")

        self._config = config
        self._data = data.reset_index(drop=True)
        self._market = MarketDataView.from_frame(self._data)
        self._timestamp_index = {
            pd.Timestamp(row["timestamp"]): int(index)
            for index, row in self._data[["timestamp"]].iterrows()
        }
        self._robot_spec = robot_spec
        self._event_callback = event_callback
        configured_fraction = float(capital_per_trade_fraction or config.risk.max_position_notional_pct)
        self._capital_fraction = min(max(configured_fraction, 0.0), 1.0)
        self._stop_loss_pct = max(float(config.risk.stop_distance_pct), 0.0)
        self._take_profit_pct = max(self._stop_loss_pct * 2.0, self._stop_loss_pct)
        self._report_writer = BacktestReportWriter(config.reporting.output_dir)
        self._runtime = RobotRuntime(
            robot_id=robot_spec.robot_id,
            display_name=robot_spec.display_name,
            workers={worker.worker_id: worker for worker in robot_spec.strategy_workers},
            cash=float(config.backtest.initial_capital),
            peak_equity=float(config.backtest.initial_capital),
        )

    def run(self) -> BacktestResult:
        run_id = f"run_{timestamp_for_run_id()}_{self._robot_spec.robot_id}"
        logger.info("Starting robot backtest: %s", run_id)

        benchmark_curve = build_buy_and_hold_benchmark(
            data=self._data,
            initial_capital=self._config.backtest.initial_capital,
        )

        self._emit("BACKTEST_STARTED", {"run_id": run_id, "robot_id": self._runtime.robot_id})

        for index in range(len(self._data)):
            current_bar = self._data.iloc[index]
            timestamp = pd.Timestamp(current_bar["timestamp"])

            self._mark_positions(float(current_bar["open"]))
            self._execute_pending_orders(current_bar)
            self._check_intrabar_exits(current_bar)
            self._mark_positions(float(current_bar["close"]))
            self._record_equity(timestamp)

            is_last_bar = index + 1 >= len(self._data)

            for worker in self._runtime.workers.values():
                self._process_worker_signal(worker, index, current_bar, is_last_bar)

        self._force_close_open_positions(self._data.iloc[-1])
        self._record_equity(pd.Timestamp(self._data.iloc[-1]["timestamp"]))

        trades = [position.to_trade_record() for position in self._runtime.closed_positions]
        equity_frame = pd.DataFrame([asdict(point) for point in self._runtime.equity_curve])

        fees_paid = sum(float(trade.get("fees", 0.0)) for trade in trades)
        slippage_cost = sum(float(trade.get("slippage_cost", 0.0)) for trade in trades)

        metrics = calculate_metrics(
            equity_curve=equity_frame,
            benchmark_curve=benchmark_curve,
            trades=trades,
            initial_capital=self._config.backtest.initial_capital,
            fees_paid=fees_paid,
            slippage_cost=slippage_cost,
            periods_per_year=self._config.backtest.periods_per_year,
        )

        backtest_start, backtest_end = self._infer_backtest_window()

        summary = {
            "run_id": run_id,

            "dataset": None,
            "dataset_source": "questdb",

            "symbol": self._config.data.symbol,
            "timeframe": self._config.data.timeframe,
            "interval": self._config.data.timeframe,

            "backtest_start": backtest_start,
            "backtest_end": backtest_end,

            "market_data": {
                "source": "questdb",
                "symbol": self._config.data.symbol,
                "interval": self._config.data.timeframe,
                "start": backtest_start,
                "end": backtest_end,
            },

            "strategy": "+".join(w.strategy_name for w in self._runtime.workers.values()),
            "robot": {
                "robot_id": self._runtime.robot_id,
                "display_name": self._runtime.display_name,
                "workers": [
                    {
                        "worker_id": worker.worker_id,
                        "strategy": worker.strategy_name,
                        "model_artifact_path": worker.model_artifact_path,
                        "helformer_artifact_path": worker.helformer_artifact_path,
                        "use_helformer_forecast": worker.use_helformer_forecast,
                    }
                    for worker in self._runtime.workers.values()
                ],
            },
            "execution_model": "signal_on_bar_close_fill_on_next_bar_open",
            "accounting_model": "collateral_based_free_cash_plus_reserved_margin_plus_gross_unrealized_pnl",
            "slippage_model": "embedded_in_fill_price_not_subtracted_from_pnl",
            "position_model": "one_independent_position_per_strategy_worker_per_symbol",
            "metrics": metrics,
        }

        run_dir = self._report_writer.write(
            run_id=run_id,
            config_snapshot=self._config.raw,
            summary=summary,
            trades=trades,
            equity_curve=equity_frame,
            benchmark_curve=benchmark_curve,
            execution_log=self._runtime.execution_log,
        )

        self._emit("BACKTEST_COMPLETED", {"run_id": run_id, "run_dir": str(run_dir)})
        logger.info("Robot backtest completed: %s", run_dir)
        return BacktestResult(run_id=run_id, run_dir=run_dir, summary=summary)

    def _process_worker_signal(
        self,
        worker: BacktestWorkerSpec,
        bar_index: int,
        current_bar: pd.Series,
        is_last_bar: bool,
    ) -> None:
        position = self._get_worker_position(worker.worker_id, self._config.data.symbol)
        portfolio_view = StrategyPortfolioView(
            symbol=self._config.data.symbol,
            position=position,
            equity=self._calculate_equity(),
            cash=self._runtime.cash,
            current_price=float(current_bar["close"]),
            bars_since_entry=self._bars_since_entry(position, bar_index),
        )
        signal = call_strategy_signal(worker.strategy, self._market, bar_index, portfolio_view)
        self._record_signal(worker, signal)

        if not signal.is_actionable:
            return

        if is_last_bar:
            self._record_order_status(
                timestamp=pd.Timestamp(current_bar["timestamp"]),
                worker=worker,
                symbol=signal.symbol,
                action=signal.signal_type.value,
                status="EXPIRED",
                reason="no_next_bar_available",
                requested_bar_time=signal.timestamp,
            )
            return

        self._schedule_order(worker=worker, signal=signal)

    def _infer_backtest_window(self) -> tuple[str, str]:
        if self._data.empty:
            raise ValueError("Cannot infer backtest window from empty data.")

        frame = self._data.copy()

        if "timestamp" not in frame.columns and "ts" in frame.columns:
            frame = frame.rename(columns={"ts": "timestamp"})

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

        start = frame["timestamp"].min()
        last = frame["timestamp"].max()
        end = last + self._interval_to_timedelta(self._config.data.timeframe)

        return start.isoformat(), end.isoformat()

    @staticmethod
    def _interval_to_timedelta(interval: str) -> pd.Timedelta:
        normalized = str(interval).strip().lower()

        if normalized.endswith("m"):
            return pd.Timedelta(minutes=int(normalized[:-1]))

        if normalized.endswith("h"):
            return pd.Timedelta(hours=int(normalized[:-1]))

        if normalized.endswith("d"):
            return pd.Timedelta(days=int(normalized[:-1]))

        raise ValueError(f"Unsupported interval for backtest window inference: {interval}")

    def _schedule_order(self, worker: BacktestWorkerSpec, signal: Signal) -> None:
        position = self._get_worker_position(worker.worker_id, signal.symbol)

        if signal.signal_type in {SignalType.LONG, SignalType.SHORT}:
            if position is not None:
                self._record_order_status(
                    timestamp=signal.timestamp,
                    worker=worker,
                    symbol=signal.symbol,
                    action=signal.signal_type.value,
                    status="REJECTED",
                    reason="worker_already_has_open_position",
                    requested_bar_time=signal.timestamp,
                )
                return
            action = OrderAction.BUY_TO_OPEN if signal.signal_type == SignalType.LONG else OrderAction.SELL_TO_OPEN
            raw_target_fraction = signal.metadata.get("target_notional_fraction", self._capital_fraction) if isinstance(signal.metadata, dict) else self._capital_fraction
            target_fraction = min(max(float(raw_target_fraction), 0.0), self._capital_fraction)
            stop_loss_pct = _positive_metadata_float(signal.metadata, "stop_loss_pct")
            take_profit_pct = _positive_metadata_float(signal.metadata, "take_profit_pct")
            quantity = 0.0
        else:
            if position is None:
                self._record_order_status(
                    timestamp=signal.timestamp,
                    worker=worker,
                    symbol=signal.symbol,
                    action=SignalType.EXIT.value,
                    status="REJECTED",
                    reason="exit_signal_without_open_position",
                    requested_bar_time=signal.timestamp,
                )
                return
            action = OrderAction.SELL_TO_CLOSE if position.side == PositionSide.LONG else OrderAction.BUY_TO_CLOSE
            target_fraction = 0.0
            stop_loss_pct = None
            take_profit_pct = None
            quantity = position.quantity

        if any(order.worker_id == worker.worker_id and order.symbol == signal.symbol for order in self._runtime.pending_orders):
            self._record_order_status(
                timestamp=signal.timestamp,
                worker=worker,
                symbol=signal.symbol,
                action=action.value,
                status="REJECTED",
                reason="worker_already_has_pending_order",
                requested_bar_time=signal.timestamp,
            )
            return

        self._runtime.pending_orders.append(
            PendingOrder(
                robot_id=self._runtime.robot_id,
                robot_name=self._runtime.display_name,
                worker_id=worker.worker_id,
                strategy_name=worker.strategy_name,
                symbol=signal.symbol,
                action=action,
                quantity=quantity,
                signal_time=signal.timestamp,
                reason=signal.reason,
                target_notional_fraction=target_fraction,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
            )
        )

    def _execute_pending_orders(self, current_bar: pd.Series) -> None:
        if not self._runtime.pending_orders:
            return

        timestamp = pd.Timestamp(current_bar["timestamp"])
        open_price = float(current_bar["open"])
        next_pending: list[PendingOrder] = []

        for order in self._runtime.pending_orders:
            try:
                if order.action in {OrderAction.BUY_TO_OPEN, OrderAction.SELL_TO_OPEN}:
                    quantity = self._size_entry_quantity(open_price, order.target_notional_fraction)
                    if quantity <= 0:
                        self._record_order_status(timestamp, order, status="REJECTED", reason="quantity_below_minimum")
                        continue
                    fill_price, fee, slippage_cost = self._simulate_fill(order.action, open_price, quantity)
                    required_cash = self._entry_required_cash(quantity, fill_price, fee)
                    if required_cash > self._runtime.cash:
                        self._record_order_status(timestamp, order, status="REJECTED", reason="insufficient_cash")
                        continue
                    self._runtime.cash -= required_cash
                    side = PositionSide.LONG if order.action == OrderAction.BUY_TO_OPEN else PositionSide.SHORT
                    self._open_position(order, timestamp, side, quantity, fill_price, fee, slippage_cost)
                    self._record_fill(timestamp, order, quantity, fill_price, fee, slippage_cost)
                else:
                    position = self._get_worker_position(order.worker_id, order.symbol)
                    if position is None:
                        self._record_order_status(timestamp, order, status="CANCELLED", reason="position_not_found")
                        continue
                    quantity = position.quantity
                    fill_price, fee, slippage_cost = self._simulate_fill(order.action, open_price, quantity)
                    self._close_position(position, timestamp, fill_price, fee, slippage_cost, order.reason)
                    self._record_fill(timestamp, order, quantity, fill_price, fee, slippage_cost)
            except Exception as exc:
                logger.exception("Order execution failed.")
                self._record_order_status(timestamp, order, status="REJECTED", reason=str(exc))

        self._runtime.pending_orders = next_pending

    def _check_intrabar_exits(self, current_bar: pd.Series) -> None:
        timestamp = pd.Timestamp(current_bar["timestamp"])
        high = float(current_bar["high"])
        low = float(current_bar["low"])

        for position in list(self._runtime.open_positions.values()):
            exit_price: float | None = None
            reason: str | None = None

            if position.side == PositionSide.LONG:
                if low <= position.stop_loss:
                    exit_price = position.stop_loss
                    reason = "stop_loss"
                elif high >= position.take_profit:
                    exit_price = position.take_profit
                    reason = "take_profit"
            else:
                if high >= position.stop_loss:
                    exit_price = position.stop_loss
                    reason = "stop_loss"
                elif low <= position.take_profit:
                    exit_price = position.take_profit
                    reason = "take_profit"

            if exit_price is None or reason is None:
                continue

            action = OrderAction.SELL_TO_CLOSE if position.side == PositionSide.LONG else OrderAction.BUY_TO_CLOSE
            fill_price, fee, slippage_cost = self._simulate_fill(action, exit_price, position.quantity)
            self._close_position(position, timestamp, fill_price, fee, slippage_cost, reason)
            self._record_fill(
                timestamp=timestamp,
                order=PendingOrder(
                    robot_id=self._runtime.robot_id,
                    robot_name=self._runtime.display_name,
                    worker_id=position.worker_id,
                    strategy_name=position.strategy,
                    symbol=position.symbol,
                    action=action,
                    quantity=position.quantity,
                    signal_time=timestamp,
                    reason=reason,
                    target_notional_fraction=0.0,
                ),
                quantity=position.quantity,
                fill_price=fill_price,
                fee=fee,
                slippage_cost=slippage_cost,
            )

    def _force_close_open_positions(self, last_bar: pd.Series) -> None:
        timestamp = pd.Timestamp(last_bar["timestamp"])
        close_price = float(last_bar["close"])
        for position in list(self._runtime.open_positions.values()):
            action = OrderAction.SELL_TO_CLOSE if position.side == PositionSide.LONG else OrderAction.BUY_TO_CLOSE
            fill_price, fee, slippage_cost = self._simulate_fill(action, close_price, position.quantity)
            self._close_position(position, timestamp, fill_price, fee, slippage_cost, "end_of_backtest")
            self._record_fill(
                timestamp=timestamp,
                order=PendingOrder(
                    robot_id=self._runtime.robot_id,
                    robot_name=self._runtime.display_name,
                    worker_id=position.worker_id,
                    strategy_name=position.strategy,
                    symbol=position.symbol,
                    action=action,
                    quantity=position.quantity,
                    signal_time=timestamp,
                    reason="end_of_backtest",
                    target_notional_fraction=0.0,
                ),
                quantity=position.quantity,
                fill_price=fill_price,
                fee=fee,
                slippage_cost=slippage_cost,
            )

    def _size_entry_quantity(self, price: float, target_notional_fraction: float | None = None) -> float:
        equity = self._calculate_equity()
        risk_amount = equity * self._config.risk.risk_per_trade_pct
        stop_distance_pct = max(self._config.risk.stop_distance_pct, 1e-9)
        stop_based_notional = risk_amount / stop_distance_pct
        risk_notional = equity * self._config.risk.max_position_notional_pct
        min_cash_reserve = equity * self._config.risk.min_cash_pct
        deployable_cash = max(0.0, self._runtime.cash - min_cash_reserve)
        fraction = self._capital_fraction if target_notional_fraction is None else min(max(float(target_notional_fraction), 0.0), self._capital_fraction)
        target_notional = min(equity * fraction, stop_based_notional, risk_notional, deployable_cash)
        if target_notional < self._config.risk.min_order_notional or price <= 0:
            return 0.0
        quantity = target_notional / price
        if not self._config.execution.allow_fractional_quantity:
            quantity = float(int(quantity))
        return quantity

    def _simulate_fill(self, action: OrderAction, raw_price: float, quantity: float) -> tuple[float, float, float]:
        if quantity <= 0:
            raise ValueError("Order quantity must be positive.")
        slippage_per_unit = raw_price * self._config.execution.slippage_bps / 10_000.0
        if action in {OrderAction.BUY_TO_OPEN, OrderAction.BUY_TO_CLOSE}:
            fill_price = raw_price + slippage_per_unit
        else:
            fill_price = max(0.0, raw_price - slippage_per_unit)
        fee = quantity * fill_price * self._config.execution.fee_rate
        slippage_cost = quantity * slippage_per_unit
        return fill_price, fee, slippage_cost

    @staticmethod
    def _entry_required_cash(quantity: float, fill_price: float, fee: float) -> float:
        # Collateral-based model for both longs and shorts:
        # reserve entry notional/margin and pay fees from free cash.
        # Slippage is already in fill_price and must not be deducted again.
        return quantity * fill_price + fee

    def _open_position(
        self,
        order: PendingOrder,
        timestamp: pd.Timestamp,
        side: PositionSide,
        quantity: float,
        fill_price: float,
        fee: float,
        slippage_cost: float,
    ) -> None:
        stop_loss_pct = order.stop_loss_pct if order.stop_loss_pct is not None and order.stop_loss_pct > 0.0 else self._stop_loss_pct
        take_profit_pct = order.take_profit_pct if order.take_profit_pct is not None and order.take_profit_pct > 0.0 else self._take_profit_pct

        if side == PositionSide.LONG:
            stop_loss = fill_price * (1.0 - stop_loss_pct)
            take_profit = fill_price * (1.0 + take_profit_pct)
        else:
            stop_loss = fill_price * (1.0 + stop_loss_pct)
            take_profit = fill_price * (1.0 - take_profit_pct)

        position = OpenPosition(
            position_id=new_id("pos"),
            robot_id=order.robot_id,
            robot_name=order.robot_name,
            worker_id=order.worker_id,
            strategy=order.strategy_name,
            symbol=order.symbol,
            side=side,
            quantity=quantity,
            entry_time=timestamp,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_fee=fee,
            entry_slippage_cost=slippage_cost,
            reason=order.reason,
        )
        self._runtime.open_positions[position_key(order.worker_id, order.symbol)] = position

    def _close_position(
        self,
        position: OpenPosition,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_fee: float,
        exit_slippage_cost: float,
        exit_reason: str,
    ) -> None:
        closed = build_closed_position(position, exit_time, exit_price, exit_fee, exit_slippage_cost, exit_reason)
        # Release reserved entry margin and realize PnL. Entry fees were paid
        # from free cash when the position was opened, and slippage is already
        # embedded in entry/exit prices.
        self._runtime.cash += position.notional + closed.gross_pnl - exit_fee
        self._runtime.closed_positions.append(closed)
        self._runtime.open_positions.pop(position_key(position.worker_id, position.symbol), None)

    def _bars_since_entry(self, position: OpenPosition | None, current_index: int) -> int:
        if position is None:
            return 0
        timestamp = pd.Timestamp(position.entry_time)
        entry_index = self._timestamp_index.get(timestamp)
        if entry_index is None:
            entry_index = self._timestamp_index.get(timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC"))
        if entry_index is None:
            return 0
        return max(0, int(current_index) - int(entry_index))

    def _get_worker_position(self, worker_id: str, symbol: str) -> OpenPosition | None:
        return self._runtime.open_positions.get(position_key(worker_id, symbol))

    def _mark_positions(self, price: float) -> None:
        for position in self._runtime.open_positions.values():
            position.update_price(price)

    def _calculate_reserved_margin(self) -> float:
        return sum(position.notional for position in self._runtime.open_positions.values())

    def _calculate_gross_unrealized_pnl(self) -> float:
        return sum(position.gross_unrealized_pnl for position in self._runtime.open_positions.values())

    def _calculate_equity(self) -> float:
        return (
            self._runtime.cash
            + self._calculate_reserved_margin()
            + self._calculate_gross_unrealized_pnl()
        )

    def _record_equity(self, timestamp: pd.Timestamp) -> None:
        long_mv = sum(p.current_notional for p in self._runtime.open_positions.values() if p.side == PositionSide.LONG)
        short_mv = sum(p.current_notional for p in self._runtime.open_positions.values() if p.side == PositionSide.SHORT)
        position_mv = long_mv + short_mv
        reserved_margin = self._calculate_reserved_margin()
        gross_unrealized_pnl = self._calculate_gross_unrealized_pnl()
        equity = self._calculate_equity()
        self._runtime.peak_equity = max(self._runtime.peak_equity, equity)
        drawdown = equity / self._runtime.peak_equity - 1.0 if self._runtime.peak_equity > 0 else 0.0
        signed_quantity = sum(p.quantity if p.side == PositionSide.LONG else -p.quantity for p in self._runtime.open_positions.values())
        self._runtime.equity_curve.append(
            EquityPoint(
                timestamp=timestamp,
                cash=self._runtime.cash,
                position_quantity=signed_quantity,
                position_market_value=position_mv,
                equity=equity,
                drawdown=drawdown,
                open_position_count=len(self._runtime.open_positions),
                long_market_value=long_mv,
                short_market_value=short_mv,
                reserved_margin=reserved_margin,
                gross_unrealized_pnl=gross_unrealized_pnl,
                accounting_model="collateral_based",
            )
        )

    def _record_signal(self, worker: BacktestWorkerSpec, signal: Signal) -> None:
        self._runtime.signal_log.append(
            {
                "timestamp": signal.timestamp,
                "robot_id": self._runtime.robot_id,
                "worker_id": worker.worker_id,
                "strategy": worker.strategy_name,
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "metadata": signal.metadata,
            }
        )

    def _record_fill(self, timestamp: pd.Timestamp, order: PendingOrder, quantity: float, fill_price: float, fee: float, slippage_cost: float) -> None:
        self._runtime.execution_log.append(
            {
                "timestamp": timestamp,
                "robot_id": order.robot_id,
                "robot_name": order.robot_name,
                "worker_id": order.worker_id,
                "strategy": order.strategy_name,
                "symbol": order.symbol,
                "action": order.action.value,
                "side": self._display_side(order.action),
                "quantity": quantity,
                "requested_bar_time": order.signal_time,
                "filled_bar_time": timestamp,
                "fill_price": fill_price,
                "fee": fee,
                "slippage_cost": slippage_cost,
                "status": "FILLED",
                "reason": order.reason,
                "robot_equity": self._calculate_equity(),
                "available_capital_after_fill": self._runtime.cash,
            }
        )

    def _record_order_status(
        self,
        timestamp: pd.Timestamp,
        order: PendingOrder | None = None,
        *,
        worker: BacktestWorkerSpec | None = None,
        symbol: str | None = None,
        action: str | None = None,
        status: str,
        reason: str,
        requested_bar_time: pd.Timestamp | None = None,
    ) -> None:
        self._runtime.execution_log.append(
            {
                "timestamp": timestamp,
                "robot_id": self._runtime.robot_id if order is None else order.robot_id,
                "robot_name": self._runtime.display_name if order is None else order.robot_name,
                "worker_id": worker.worker_id if worker is not None else (order.worker_id if order else None),
                "strategy": worker.strategy_name if worker is not None else (order.strategy_name if order else None),
                "symbol": symbol or (order.symbol if order else None),
                "action": action or (order.action.value if order else None),
                "side": None if order is None else self._display_side(order.action),
                "quantity": None if order is None else order.quantity,
                "requested_bar_time": requested_bar_time or (order.signal_time if order else timestamp),
                "filled_bar_time": None,
                "fill_price": None,
                "fee": None,
                "slippage_cost": None,
                "status": status,
                "reason": reason,
                "robot_equity": self._calculate_equity(),
                "available_capital_after_fill": self._runtime.cash,
            }
        )

    @staticmethod
    def _display_side(action: OrderAction) -> str:
        if action in {OrderAction.BUY_TO_OPEN, OrderAction.BUY_TO_CLOSE}:
            return "BUY"
        return "SELL"

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        self._event_callback({"event_type": event_type, "source": "robot_backtest_engine", "payload": payload})






def _positive_metadata_float(metadata: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(metadata, dict):
        return None
    try:
        value = float(metadata.get(key))
    except (TypeError, ValueError):
        return None
    if value > 0.0:
        return value
    return None
