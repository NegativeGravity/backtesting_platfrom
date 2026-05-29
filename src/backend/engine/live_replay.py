from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import pandas as pd

from backend.backtest.benchmark import build_buy_and_hold_benchmark
from backend.backtest.metrics import calculate_metrics
from backend.backtest.report import BacktestReportWriter
from backend.core.config import load_config
from backend.core.paths import resolve_model_artifact_path
from backend.core.time import timestamp_for_run_id
from backend.data.validator import validate_ohlcv
from backend.engine.bot_worker import run_bot_worker
from backend.engine.messages import BotCommand, BotResponse
from backend.engine.shared_market import SharedMarketDataOwner
from backend.engine.positions import (
    ClosedPosition,
    OpenPosition,
    PositionSide,
    build_closed_position,
    position_key,
)
from backend.data.market_store import MarketDataStore
from backend.ml.time_policy import (
    BACKTEST_START_STR,
    BACKTEST_END_STR,
    assert_backtest_2025_only,
)
from backend.events.envelope import EventEnvelope, EventType
from backend.execution.orders import LiveOrderIntent, OrderAction
from backend.strategy.signals import Signal, SignalType

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]


@dataclass(frozen=True)
class LiveStrategyWorkerSpec:
    worker_id: str
    strategy_name: str
    model_artifact_path: str | None = None


@dataclass(frozen=True)
class LiveRobotSpec:
    robot_id: str
    display_name: str
    strategy_workers: list[LiveStrategyWorkerSpec]


@dataclass
class StrategyWorkerRuntime:
    worker_id: str
    strategy_name: str
    command_queue: mp.Queue
    response_queue: mp.Queue
    process: mp.Process


@dataclass
class PendingLiveOrder:
    order: LiveOrderIntent
    worker_id: str
    strategy_name: str
    target_notional_fraction: float


@dataclass
class EquityPoint:
    timestamp: pd.Timestamp
    cash: float
    position_quantity: float
    position_market_value: float
    equity: float
    drawdown: float
    reserved_margin: float = 0.0
    gross_unrealized_pnl: float = 0.0
    available_capital: float = 0.0
    accounting_model: str = "collateral_based"


@dataclass
class LiveRobotRuntime:
    robot_id: str
    display_name: str
    strategy_workers: dict[str, StrategyWorkerRuntime]
    cash: float
    pending_orders: list[PendingLiveOrder] = field(default_factory=list)
    open_positions: dict[str, OpenPosition] = field(default_factory=dict)
    closed_positions: list[ClosedPosition] = field(default_factory=list)
    execution_log: list[dict] = field(default_factory=list)
    signal_log: list[dict] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)
    last_close: float = 0.0
    peak_equity: float = 0.0


@dataclass(frozen=True)
class LiveReplayResult:
    session_id: str
    output_dirs: dict[str, str]


class LiveReplayEngine:
    def __init__(
        self,
        config_path: str,
        robot_specs: list[LiveRobotSpec],
        replay_delay_seconds: float = 0.0,
        event_callback: EventCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._config_path = config_path
        self._config = load_config(config_path)

        if not robot_specs:
            raise ValueError("At least one robot spec is required.")

        for robot_spec in robot_specs:
            if not robot_spec.robot_id.strip():
                raise ValueError("Robot ID cannot be empty.")

            if not robot_spec.display_name.strip():
                raise ValueError(f"Robot {robot_spec.robot_id} display name cannot be empty.")

            if not robot_spec.strategy_workers:
                raise ValueError(f"Robot {robot_spec.robot_id} has no strategy workers.")

            worker_ids = [worker.worker_id for worker in robot_spec.strategy_workers]

            if len(worker_ids) != len(set(worker_ids)):
                raise ValueError(f"Robot {robot_spec.robot_id} has duplicated worker IDs.")

        self._robot_specs = robot_specs
        self._replay_delay_seconds = replay_delay_seconds
        self._event_callback = event_callback
        self._stop_event = stop_event or threading.Event()
        self._capital_fraction = min(max(float(self._config.risk.max_position_notional_pct), 0.0), 1.0)
        self._stop_loss_pct = max(float(self._config.risk.stop_distance_pct), 0.0)
        self._take_profit_pct = max(self._stop_loss_pct * 2.0, self._stop_loss_pct)
        self._session_id = f"live_{timestamp_for_run_id()}"
        self._robot_runtimes: dict[str, LiveRobotRuntime] = {}
        self._shared_market: SharedMarketDataOwner | None = None

    def run(self) -> LiveReplayResult:
        self._emit(
            EventType.ENGINE_STARTED,
            source="live_replay_engine",
            payload={
                "session_id": self._session_id,
                "execution_model": "signal_on_current_bar_fill_at_next_open",
                "capital_model": "collateral_based_free_cash_plus_reserved_margin_plus_gross_unrealized_pnl",
                "slippage_model": "embedded_in_fill_price_not_subtracted_from_pnl",
                "position_model": "worker_owned_long_short_positions",
                "capital_per_trade_fraction": self._capital_fraction,
                "default_stop_loss_pct": self._stop_loss_pct,
                "default_take_profit_pct": self._take_profit_pct,
                "robots": [
                    {
                        "robot_id": robot.robot_id,
                        "display_name": robot.display_name,
                        "strategy_workers": [
                            {
                                "worker_id": worker.worker_id,
                                "strategy": worker.strategy_name,
                                "model_artifact_path": worker.model_artifact_path,
                            }
                            for worker in robot.strategy_workers
                        ],
                    }
                    for robot in self._robot_specs
                ],
            },
        )

        data = self._load_data()
        benchmark_curve = build_buy_and_hold_benchmark(
            data=data,
            initial_capital=self._config.backtest.initial_capital,
        )

        try:
            self._start_strategy_workers(data)

            for index in range(len(data)):
                if self._stop_event.is_set():
                    break

                current_bar = data.iloc[index]
                current_close = float(current_bar["close"])

                self._process_worker_responses(block=False)
                self._emit_market_bar(current_bar)

                for robot_runtime in self._robot_runtimes.values():
                    robot_runtime.last_close = current_close

                    self._update_open_positions(
                        robot_runtime=robot_runtime,
                        current_price=current_close,
                        timestamp=pd.Timestamp(current_bar["timestamp"]),
                    )

                    self._execute_pending_orders_at_next_open(
                        robot_runtime=robot_runtime,
                        current_bar=current_bar,
                    )

                    self._check_tp_sl_for_open_positions(
                        robot_runtime=robot_runtime,
                        current_bar=current_bar,
                    )

                    self._update_open_positions(
                        robot_runtime=robot_runtime,
                        current_price=current_close,
                        timestamp=pd.Timestamp(current_bar["timestamp"]),
                    )

                    self._record_and_emit_portfolio_update(
                        robot_runtime=robot_runtime,
                        bar=current_bar,
                    )

                    self._send_bar_index_to_robot_workers(
                        robot_runtime=robot_runtime,
                        bar_index=index,
                        current_close=current_close,
                    )

                self._process_worker_responses_for(max_wait_seconds=0.005)

                if self._replay_delay_seconds > 0:
                    time.sleep(self._replay_delay_seconds)

            output_dirs = self._persist_reports(benchmark_curve)

            return LiveReplayResult(
                session_id=self._session_id,
                output_dirs=output_dirs,
            )

        finally:
            self._stop_strategy_workers()
            self._close_shared_market()
            self._emit(
                EventType.ENGINE_STOPPED,
                source="live_replay_engine",
                payload={"session_id": self._session_id},
            )

    def _load_data(self) -> pd.DataFrame:
        data = MarketDataStore.from_yaml().load_ohlcv(
            start=BACKTEST_START_STR,
            end=BACKTEST_END_STR,
            symbol=self._config.data.symbol,
            interval=self._config.data.timeframe,
            use_cache=True,
        )

        assert_backtest_2025_only(data)
        validate_ohlcv(data)

        return data

    def _start_strategy_workers(self, data: pd.DataFrame) -> None:
        self._shared_market = SharedMarketDataOwner.from_frame(data)
        shared_market_descriptor = self._shared_market.descriptor()

        for robot_spec in self._robot_specs:
            worker_runtimes: dict[str, StrategyWorkerRuntime] = {}

            for worker_spec in robot_spec.strategy_workers:
                command_queue: mp.Queue = mp.Queue(maxsize=100)
                response_queue: mp.Queue = mp.Queue(maxsize=100)

                model_path = None

                if worker_spec.strategy_name == "ml_momentum":
                    if worker_spec.model_artifact_path is None:
                        raise ValueError(
                            f"Worker {worker_spec.worker_id} uses ml_momentum "
                            "but model_artifact_path is missing."
                        )

                    model_path = str(resolve_model_artifact_path(worker_spec.model_artifact_path))

                process = mp.Process(
                    target=run_bot_worker,
                    kwargs={
                        "bot_id": worker_spec.worker_id,
                        "strategy_name": worker_spec.strategy_name,
                        "config_path": self._config_path,
                        "model_artifact_path": model_path,
                        "command_queue": command_queue,
                        "response_queue": response_queue,
                        "shared_market_descriptor": shared_market_descriptor,
                    },
                    daemon=True,
                )
                process.start()

                worker_runtimes[worker_spec.worker_id] = StrategyWorkerRuntime(
                    worker_id=worker_spec.worker_id,
                    strategy_name=worker_spec.strategy_name,
                    command_queue=command_queue,
                    response_queue=response_queue,
                    process=process,
                )

            initial_capital = float(self._config.backtest.initial_capital)

            self._robot_runtimes[robot_spec.robot_id] = LiveRobotRuntime(
                robot_id=robot_spec.robot_id,
                display_name=robot_spec.display_name,
                strategy_workers=worker_runtimes,
                cash=initial_capital,
                peak_equity=initial_capital,
            )

    def _stop_strategy_workers(self) -> None:
        for robot_runtime in self._robot_runtimes.values():
            for worker_runtime in robot_runtime.strategy_workers.values():
                try:
                    worker_runtime.command_queue.put_nowait(
                        BotCommand(command_type="STOP", payload={})
                    )
                except Exception:
                    logger.exception(
                        "Failed to send STOP to strategy worker: %s",
                        worker_runtime.worker_id,
                    )

        for robot_runtime in self._robot_runtimes.values():
            for worker_runtime in robot_runtime.strategy_workers.values():
                worker_runtime.process.join(timeout=3)

                if worker_runtime.process.is_alive():
                    worker_runtime.process.terminate()
                    worker_runtime.process.join(timeout=3)

    def _send_bar_index_to_robot_workers(
        self,
        robot_runtime: LiveRobotRuntime,
        bar_index: int,
        current_close: float,
    ) -> None:
        robot_equity = self._calculate_robot_equity(robot_runtime)
        available_capital = self._calculate_available_robot_capital(robot_runtime)

        for worker_runtime in robot_runtime.strategy_workers.values():
            active_position = self._get_worker_position(
                robot_runtime=robot_runtime,
                worker_id=worker_runtime.worker_id,
                symbol=self._config.data.symbol,
            )

            payload = {
                "bar_index": bar_index,
                "portfolio": {
                    "symbol": self._config.data.symbol,
                    "position_quantity": 0.0 if active_position is None else active_position.quantity,
                    "position_side": None if active_position is None else active_position.side.value,
                    "equity": robot_equity,
                    "cash": available_capital,
                    "current_price": current_close,
                    "unrealized_pnl": None if active_position is None else active_position.unrealized_pnl,
                    "robot_equity": robot_equity,
                    "robot_available_capital": available_capital,
                    "bar_index": bar_index,
                },
            }

            try:
                worker_runtime.command_queue.put_nowait(
                    BotCommand(command_type="BAR_INDEX", payload=payload)
                )
            except queue.Full:
                self._emit(
                    EventType.SYSTEM_ALERT,
                    source="live_replay_engine",
                    payload={
                        "robot_id": robot_runtime.robot_id,
                        "robot_name": robot_runtime.display_name,
                        "worker_id": worker_runtime.worker_id,
                        "strategy": worker_runtime.strategy_name,
                        "reason": "worker_command_queue_full_bar_skipped",
                        "bar_index": bar_index,
                    },
                )

    def _process_worker_responses(
        self,
        block: bool = False,
        timeout: float = 0.0,
    ) -> int:
        processed = 0
        for robot_runtime in self._robot_runtimes.values():
            for worker_runtime in robot_runtime.strategy_workers.values():
                while True:
                    try:
                        response: BotResponse = worker_runtime.response_queue.get(
                            block=block,
                            timeout=timeout if block else 0,
                        )
                    except queue.Empty:
                        break

                    self._handle_worker_response(
                        robot_runtime=robot_runtime,
                        worker_runtime=worker_runtime,
                        response=response,
                    )
                    processed += 1

                    if block:
                        break
        return processed

    def _process_worker_responses_for(self, max_wait_seconds: float) -> int:
        deadline = time.perf_counter() + max(0.0, max_wait_seconds)
        processed = self._process_worker_responses(block=False)
        while time.perf_counter() < deadline:
            batch = self._process_worker_responses(block=False)
            processed += batch
            if batch == 0:
                time.sleep(0.0005)
        return processed

    def _close_shared_market(self) -> None:
        if self._shared_market is not None:
            self._shared_market.close()
            self._shared_market = None

    def _handle_worker_response(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_runtime: StrategyWorkerRuntime,
        response: BotResponse,
    ) -> None:
        if response.response_type == "BOT_STARTED":
            self._emit_worker_lifecycle_event(
                event_type=EventType.BOT_STARTED,
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                payload=response.payload,
            )
            return

        if response.response_type == "BOT_STOPPED":
            self._emit_worker_lifecycle_event(
                event_type=EventType.BOT_STOPPED,
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                payload=response.payload,
            )
            return

        if response.response_type == "BOT_HEARTBEAT":
            self._emit_worker_lifecycle_event(
                event_type=EventType.BOT_HEARTBEAT,
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                payload=response.payload,
            )
            return

        if response.response_type == "BOT_ERROR":
            self._emit_worker_lifecycle_event(
                event_type=EventType.BOT_ERROR,
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                payload=response.payload,
            )
            return

        if response.response_type != "SIGNAL_GENERATED":
            return

        signal_payload = {
            **response.payload,
            "robot_id": robot_runtime.robot_id,
            "robot_name": robot_runtime.display_name,
            "worker_id": worker_runtime.worker_id,
            "strategy": worker_runtime.strategy_name,
        }

        robot_runtime.signal_log.append(signal_payload)

        self._emit(
            EventType.SIGNAL_GENERATED,
            source=worker_runtime.worker_id,
            payload=signal_payload,
        )

        signal = self._signal_from_payload(response.payload)

        if not signal.is_actionable:
            return

        current_price = float(signal.metadata.get("close", 0.0)) or robot_runtime.last_close

        if current_price <= 0:
            self._emit(
                EventType.ORDER_REJECTED,
                source="risk_manager",
                payload={
                    "robot_id": robot_runtime.robot_id,
                    "robot_name": robot_runtime.display_name,
                    "worker_id": worker_runtime.worker_id,
                    "strategy": worker_runtime.strategy_name,
                    "reason": "missing_current_price",
                },
            )
            return

        if self._has_pending_order_for_worker(
            robot_runtime=robot_runtime,
            worker_id=worker_runtime.worker_id,
        ):
            self._emit(
                EventType.SYSTEM_ALERT,
                source="order_manager",
                payload={
                    "robot_id": robot_runtime.robot_id,
                    "robot_name": robot_runtime.display_name,
                    "worker_id": worker_runtime.worker_id,
                    "strategy": worker_runtime.strategy_name,
                    "reason": "worker_already_has_pending_order",
                    "signal_type": signal.signal_type.value,
                },
            )
            return

        existing_position = self._get_worker_position(
            robot_runtime=robot_runtime,
            worker_id=worker_runtime.worker_id,
            symbol=signal.symbol,
        )

        if signal.signal_type == SignalType.LONG:
            if existing_position is not None:
                self._emit_position_rejected(
                    robot_runtime=robot_runtime,
                    worker_runtime=worker_runtime,
                    symbol=signal.symbol,
                    reason="worker_already_has_open_position",
                )
                return

            self._schedule_order(
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                order=LiveOrderIntent(
                    symbol=signal.symbol,
                    action=OrderAction.BUY_TO_OPEN,
                    quantity=1.0,
                    signal_time=signal.timestamp,
                    reason=signal.reason,
                ),
                target_notional_fraction=self._capital_fraction,
                current_price=current_price,
            )
            return

        if signal.signal_type == SignalType.SHORT:
            if existing_position is not None:
                self._emit_position_rejected(
                    robot_runtime=robot_runtime,
                    worker_runtime=worker_runtime,
                    symbol=signal.symbol,
                    reason="worker_already_has_open_position",
                )
                return

            self._schedule_order(
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                order=LiveOrderIntent(
                    symbol=signal.symbol,
                    action=OrderAction.SELL_TO_OPEN,
                    quantity=1.0,
                    signal_time=signal.timestamp,
                    reason=signal.reason,
                ),
                target_notional_fraction=self._capital_fraction,
                current_price=current_price,
            )
            return

        if signal.signal_type == SignalType.EXIT:
            if existing_position is None:
                self._emit_position_rejected(
                    robot_runtime=robot_runtime,
                    worker_runtime=worker_runtime,
                    symbol=signal.symbol,
                    reason="exit_signal_without_open_position",
                )
                return

            close_action = (
                OrderAction.SELL_TO_CLOSE
                if existing_position.side == PositionSide.LONG
                else OrderAction.BUY_TO_CLOSE
            )

            self._schedule_order(
                robot_runtime=robot_runtime,
                worker_runtime=worker_runtime,
                order=LiveOrderIntent(
                    symbol=signal.symbol,
                    action=close_action,
                    quantity=existing_position.quantity,
                    signal_time=signal.timestamp,
                    reason=signal.reason,
                ),
                target_notional_fraction=0.0,
                current_price=current_price,
            )
            return

    def _schedule_order(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_runtime: StrategyWorkerRuntime,
        order: LiveOrderIntent,
        target_notional_fraction: float,
        current_price: float,
    ) -> None:
        robot_runtime.pending_orders.append(
            PendingLiveOrder(
                order=order,
                worker_id=worker_runtime.worker_id,
                strategy_name=worker_runtime.strategy_name,
                target_notional_fraction=target_notional_fraction,
            )
        )

        self._emit(
            EventType.ORDER_SCHEDULED,
            source="order_manager",
            payload={
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "worker_id": worker_runtime.worker_id,
                "strategy": worker_runtime.strategy_name,
                "symbol": order.symbol,
                "action": order.action.value,
                "quantity": order.quantity,
                "signal_time": order.signal_time.isoformat(),
                "scheduled_execution": "next_bar_open",
                "capital_per_trade_fraction": target_notional_fraction,
                "robot_equity": self._calculate_robot_equity(robot_runtime),
                "available_capital": self._calculate_available_robot_capital(robot_runtime),
                "reference_price": current_price,
                "reason": order.reason,
            },
        )

    def _execute_pending_orders_at_next_open(
        self,
        robot_runtime: LiveRobotRuntime,
        current_bar: pd.Series,
    ) -> None:
        if not robot_runtime.pending_orders:
            return

        current_open = float(current_bar["open"])
        current_timestamp = pd.Timestamp(current_bar["timestamp"])

        for pending in list(robot_runtime.pending_orders):
            order = pending.order
            is_open_order = order.action in {
                OrderAction.BUY_TO_OPEN,
                OrderAction.SELL_TO_OPEN,
            }

            if is_open_order:
                robot_equity = self._calculate_robot_equity(robot_runtime)
                available_capital = self._calculate_available_robot_capital(robot_runtime)
                risk_amount = robot_equity * self._config.risk.risk_per_trade_pct
                stop_distance_pct = max(self._config.risk.stop_distance_pct, 1e-9)
                stop_based_notional = risk_amount / stop_distance_pct
                risk_notional = robot_equity * self._config.risk.max_position_notional_pct
                min_cash_reserve = robot_equity * self._config.risk.min_cash_pct
                deployable_capital = max(0.0, available_capital - min_cash_reserve)
                target_notional = min(
                    robot_equity * pending.target_notional_fraction,
                    stop_based_notional,
                    risk_notional,
                    deployable_capital,
                )

                if target_notional < self._config.risk.min_order_notional or available_capital < target_notional:
                    self._record_cancelled_order(
                        robot_runtime=robot_runtime,
                        pending=pending,
                        timestamp=current_timestamp,
                        reason="insufficient_free_robot_capital",
                        quantity=0.0,
                        price=current_open,
                        target_notional=target_notional,
                        available_capital=available_capital,
                    )
                    continue

                quantity = target_notional / current_open
            else:
                position = self._get_worker_position(
                    robot_runtime=robot_runtime,
                    worker_id=pending.worker_id,
                    symbol=order.symbol,
                )

                if position is None:
                    self._record_cancelled_order(
                        robot_runtime=robot_runtime,
                        pending=pending,
                        timestamp=current_timestamp,
                        reason="close_order_without_open_position",
                        quantity=0.0,
                        price=current_open,
                    )
                    continue

                quantity = position.quantity

            fill_price, fee, slippage_cost = self._simulate_fill(
                action=order.action,
                raw_price=current_open,
                quantity=quantity,
            )

            execution_record = {
                "timestamp": current_timestamp,
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "worker_id": pending.worker_id,
                "strategy": pending.strategy_name,
                "symbol": order.symbol,
                "action": order.action.value,
                "side": self._display_side(order.action),
                "quantity": quantity,
                "requested_bar_time": order.signal_time,
                "filled_bar_time": current_timestamp,
                "fill_price": fill_price,
                "fee": fee,
                "slippage_cost": slippage_cost,
                "status": "FILLED",
                "reason": order.reason,
                "capital_per_trade_fraction": pending.target_notional_fraction,
            }

            if order.action == OrderAction.BUY_TO_OPEN:
                robot_runtime.cash -= self._entry_required_cash(quantity, fill_price, fee)
                self._open_position(
                    robot_runtime=robot_runtime,
                    pending=pending,
                    timestamp=current_timestamp,
                    side=PositionSide.LONG,
                    quantity=quantity,
                    fill_price=fill_price,
                    fee=fee,
                    slippage_cost=slippage_cost,
                )

            elif order.action == OrderAction.SELL_TO_OPEN:
                robot_runtime.cash -= self._entry_required_cash(quantity, fill_price, fee)
                self._open_position(
                    robot_runtime=robot_runtime,
                    pending=pending,
                    timestamp=current_timestamp,
                    side=PositionSide.SHORT,
                    quantity=quantity,
                    fill_price=fill_price,
                    fee=fee,
                    slippage_cost=slippage_cost,
                )

            elif order.action in {OrderAction.SELL_TO_CLOSE, OrderAction.BUY_TO_CLOSE}:
                self._close_position(
                    robot_runtime=robot_runtime,
                    worker_id=pending.worker_id,
                    symbol=order.symbol,
                    exit_time=current_timestamp,
                    exit_price=fill_price,
                    exit_fee=fee,
                    exit_slippage_cost=slippage_cost,
                    exit_reason=order.reason,
                )

            execution_record["robot_equity"] = self._calculate_robot_equity(robot_runtime)
            execution_record["available_capital_after_fill"] = (
                self._calculate_available_robot_capital(robot_runtime)
            )

            robot_runtime.execution_log.append(execution_record)

            self._emit(
                EventType.ORDER_FILLED,
                source="execution_engine",
                payload=self._json_safe_execution_record(execution_record),
            )

        robot_runtime.pending_orders = []

    def _open_position(
        self,
        robot_runtime: LiveRobotRuntime,
        pending: PendingLiveOrder,
        timestamp: pd.Timestamp,
        side: PositionSide,
        quantity: float,
        fill_price: float,
        fee: float,
        slippage_cost: float,
    ) -> None:
        order = pending.order

        if side == PositionSide.LONG:
            stop_loss = fill_price * (1.0 - self._stop_loss_pct)
            take_profit = fill_price * (1.0 + self._take_profit_pct)
        else:
            stop_loss = fill_price * (1.0 + self._stop_loss_pct)
            take_profit = fill_price * (1.0 - self._take_profit_pct)

        open_position = OpenPosition(
            position_id=f"{robot_runtime.robot_id}_{pending.worker_id}_{order.symbol}_{timestamp.isoformat()}",
            robot_id=robot_runtime.robot_id,
            robot_name=robot_runtime.display_name,
            worker_id=pending.worker_id,
            strategy=pending.strategy_name,
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

        robot_runtime.open_positions[position_key(pending.worker_id, order.symbol)] = open_position

        self._emit(
            EventType.POSITION_OPENED,
            source="position_manager",
            payload=open_position.to_event_payload(),
        )

    def _close_position(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_id: str,
        symbol: str,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_fee: float,
        exit_slippage_cost: float,
        exit_reason: str,
    ) -> None:
        key = position_key(worker_id, symbol)
        position = robot_runtime.open_positions.pop(key, None)

        if position is None:
            return

        closed_position = build_closed_position(
            position=position,
            exit_time=exit_time,
            exit_price=exit_price,
            exit_fee=exit_fee,
            exit_slippage_cost=exit_slippage_cost,
            exit_reason=exit_reason,
        )

        # Release reserved entry margin and realize PnL. Entry fees were paid
        # from free cash at open; slippage is already included in fill prices.
        robot_runtime.cash += position.notional + closed_position.gross_pnl - exit_fee
        robot_runtime.closed_positions.append(closed_position)

        self._emit(
            EventType.POSITION_CLOSED,
            source="position_manager",
            payload=closed_position.to_event_payload(),
        )

    def _check_tp_sl_for_open_positions(
        self,
        robot_runtime: LiveRobotRuntime,
        current_bar: pd.Series,
    ) -> None:
        if not robot_runtime.open_positions:
            return

        timestamp = pd.Timestamp(current_bar["timestamp"])
        high = float(current_bar["high"])
        low = float(current_bar["low"])

        for position in list(robot_runtime.open_positions.values()):
            exit_price: float | None = None
            exit_reason: str | None = None

            if position.side == PositionSide.LONG:
                if low <= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss_hit"
                elif high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit_hit"

            if position.side == PositionSide.SHORT:
                if high >= position.stop_loss:
                    exit_price = position.stop_loss
                    exit_reason = "stop_loss_hit"
                elif low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit_hit"

            if exit_price is None or exit_reason is None:
                continue

            action = (
                OrderAction.SELL_TO_CLOSE
                if position.side == PositionSide.LONG
                else OrderAction.BUY_TO_CLOSE
            )
            fill_price, fee, slippage_cost = self._simulate_fill(
                action=action,
                raw_price=exit_price,
                quantity=position.quantity,
            )

            execution_record = {
                "timestamp": timestamp,
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "worker_id": position.worker_id,
                "strategy": position.strategy,
                "symbol": position.symbol,
                "action": action.value,
                "side": self._display_side(action),
                "quantity": position.quantity,
                "requested_bar_time": timestamp,
                "filled_bar_time": timestamp,
                "fill_price": fill_price,
                "fee": fee,
                "slippage_cost": slippage_cost,
                "status": "FILLED",
                "reason": exit_reason,
            }

            self._close_position(
                robot_runtime=robot_runtime,
                worker_id=position.worker_id,
                symbol=position.symbol,
                exit_time=timestamp,
                exit_price=fill_price,
                exit_fee=fee,
                exit_slippage_cost=slippage_cost,
                exit_reason=exit_reason,
            )

            execution_record["robot_equity"] = self._calculate_robot_equity(robot_runtime)
            execution_record["available_capital_after_fill"] = (
                self._calculate_available_robot_capital(robot_runtime)
            )

            robot_runtime.execution_log.append(execution_record)

            self._emit(
                EventType.ORDER_FILLED,
                source="position_manager",
                payload=self._json_safe_execution_record(execution_record),
            )

    def _update_open_positions(
        self,
        robot_runtime: LiveRobotRuntime,
        current_price: float,
        timestamp: pd.Timestamp,
    ) -> None:
        for position in robot_runtime.open_positions.values():
            position.update_price(current_price)

            payload = position.to_event_payload()
            payload["timestamp"] = timestamp.isoformat()

            self._emit(
                EventType.POSITION_UPDATED,
                source="position_manager",
                payload=payload,
            )

    def _record_and_emit_portfolio_update(
        self,
        robot_runtime: LiveRobotRuntime,
        bar: pd.Series,
    ) -> None:
        timestamp = pd.Timestamp(bar["timestamp"])
        equity = self._calculate_robot_equity(robot_runtime)
        position_market_value = self._calculate_exposure_notional(robot_runtime)
        reserved_margin = self._calculate_reserved_margin(robot_runtime)
        gross_unrealized_pnl = self._calculate_gross_unrealized_pnl(robot_runtime)
        available_capital = self._calculate_available_robot_capital(robot_runtime)

        robot_runtime.peak_equity = max(robot_runtime.peak_equity, equity)
        drawdown = 0.0 if robot_runtime.peak_equity <= 0 else equity / robot_runtime.peak_equity - 1.0

        point = EquityPoint(
            timestamp=timestamp,
            cash=available_capital,
            position_quantity=0.0,
            position_market_value=position_market_value,
            equity=equity,
            drawdown=drawdown,
            reserved_margin=reserved_margin,
            gross_unrealized_pnl=gross_unrealized_pnl,
            available_capital=available_capital,
        )
        robot_runtime.equity_curve.append(point)

        self._emit(
            EventType.PORTFOLIO_UPDATED,
            source=robot_runtime.robot_id,
            payload={
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "timestamp": timestamp.isoformat(),
                "cash": available_capital,
                "equity": equity,
                "drawdown": drawdown,
                "committed_capital": reserved_margin,
                "reserved_margin": reserved_margin,
                "exposure_notional": position_market_value,
                "gross_unrealized_pnl": gross_unrealized_pnl,
                "available_capital": available_capital,
                "open_positions": [
                    position.to_event_payload()
                    for position in robot_runtime.open_positions.values()
                ],
                "closed_positions_count": len(robot_runtime.closed_positions),
            },
        )

    def _persist_reports(self, benchmark_curve: pd.DataFrame) -> dict[str, str]:
        output_dirs: dict[str, str] = {}

        for robot_runtime in self._robot_runtimes.values():
            equity_curve = pd.DataFrame([asdict(point) for point in robot_runtime.equity_curve])
            trades = [
                closed_position.to_trade_record()
                for closed_position in robot_runtime.closed_positions
            ]

            metrics = calculate_metrics(
                equity_curve=equity_curve,
                benchmark_curve=benchmark_curve,
                trades=trades,
                initial_capital=self._config.backtest.initial_capital,
                fees_paid=sum(trade["fees"] for trade in trades),
                slippage_cost=sum(trade["slippage_cost"] for trade in trades),
                periods_per_year=self._config.backtest.periods_per_year,
            )

            run_id = f"{self._session_id}_{robot_runtime.robot_id}"

            summary = {
                "run_id": run_id,
                "session_id": self._session_id,
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "dataset": str(self._config.data.path),
                "symbol": self._config.data.symbol,
                "timeframe": self._config.data.timeframe,
                "strategy": "multi_strategy_robot",
                "engine_mode": "live_replay_multiprocessing_next_bar_open",
                "execution_model": "signal_on_current_bar_fill_at_next_open",
                "capital_model": "collateral_based_free_cash_plus_reserved_margin_plus_gross_unrealized_pnl",
                "slippage_model": "embedded_in_fill_price_not_subtracted_from_pnl",
                "position_model": "worker_owned_long_short_positions",
                "capital_per_trade_fraction": self._capital_fraction,
                "default_stop_loss_pct": self._stop_loss_pct,
                "default_take_profit_pct": self._take_profit_pct,
                "strategy_workers": [
                    {
                        "worker_id": worker.worker_id,
                        "strategy": worker.strategy_name,
                    }
                    for worker in robot_runtime.strategy_workers.values()
                ],
                "metrics": metrics,
            }

            writer = BacktestReportWriter(self._config.reporting.output_dir)
            run_dir = writer.write(
                run_id=run_id,
                config_snapshot=self._config.raw,
                summary=summary,
                trades=trades,
                equity_curve=[],
                benchmark_curve=benchmark_curve,
                execution_log=robot_runtime.execution_log,
            )

            equity_curve.to_csv(run_dir / "equity_curve.csv", index=False)
            pd.DataFrame(trades).to_csv(run_dir / "closed_positions.csv", index=False)

            output_dirs[robot_runtime.robot_id] = str(run_dir)

        return output_dirs

    def _simulate_fill(
        self,
        action: OrderAction,
        raw_price: float,
        quantity: float,
    ) -> tuple[float, float, float]:
        slippage_bps = self._get_execution_bps(
            "slippage_bps",
            "slippage_basis_points",
            "slippage",
            default=5.0,
        )
        fee_bps = self._get_execution_bps(
            "fee_bps",
            "commission_bps",
            "fee_basis_points",
            default=10.0,
        )

        slippage_rate = slippage_bps / 10_000.0
        fee_rate = fee_bps / 10_000.0

        if action in {OrderAction.BUY_TO_OPEN, OrderAction.BUY_TO_CLOSE}:
            fill_price = raw_price * (1.0 + slippage_rate)
        else:
            fill_price = raw_price * (1.0 - slippage_rate)

        notional = abs(quantity * fill_price)
        fee = notional * fee_rate
        slippage_cost = abs(quantity * raw_price * slippage_rate)

        return fill_price, fee, slippage_cost

    @staticmethod
    def _entry_required_cash(quantity: float, fill_price: float, fee: float) -> float:
        # Reserve entry notional/margin and pay fee from free cash.
        # Slippage is represented by fill_price, not charged a second time.
        return abs(quantity * fill_price) + fee

    def _get_execution_bps(
        self,
        *names: str,
        default: float,
    ) -> float:
        for name in names:
            value = getattr(self._config.execution, name, None)
            if value is not None:
                return float(value)

        return default

    def _calculate_robot_equity(self, robot_runtime: LiveRobotRuntime) -> float:
        return (
            robot_runtime.cash
            + self._calculate_reserved_margin(robot_runtime)
            + self._calculate_gross_unrealized_pnl(robot_runtime)
        )

    def _calculate_reserved_margin(self, robot_runtime: LiveRobotRuntime) -> float:
        return sum(
            position.notional
            for position in robot_runtime.open_positions.values()
        )

    def _calculate_gross_unrealized_pnl(self, robot_runtime: LiveRobotRuntime) -> float:
        return sum(
            position.gross_unrealized_pnl
            for position in robot_runtime.open_positions.values()
        )

    def _calculate_exposure_notional(self, robot_runtime: LiveRobotRuntime) -> float:
        return sum(
            position.current_notional
            for position in robot_runtime.open_positions.values()
        )

    def _calculate_committed_capital(self, robot_runtime: LiveRobotRuntime) -> float:
        return self._calculate_reserved_margin(robot_runtime)

    def _calculate_available_robot_capital(self, robot_runtime: LiveRobotRuntime) -> float:
        equity = self._calculate_robot_equity(robot_runtime)
        reserve = equity * self._config.risk.min_cash_pct
        return max(robot_runtime.cash - reserve, 0.0)

    def _get_worker_position(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_id: str,
        symbol: str,
    ) -> OpenPosition | None:
        return robot_runtime.open_positions.get(position_key(worker_id, symbol))

    def _has_pending_order_for_worker(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_id: str,
    ) -> bool:
        return any(
            pending.worker_id == worker_id
            for pending in robot_runtime.pending_orders
        )

    def _record_cancelled_order(
        self,
        robot_runtime: LiveRobotRuntime,
        pending: PendingLiveOrder,
        timestamp: pd.Timestamp,
        reason: str,
        quantity: float,
        price: float,
        target_notional: float | None = None,
        available_capital: float | None = None,
    ) -> None:
        record = {
            "timestamp": timestamp,
            "robot_id": robot_runtime.robot_id,
            "robot_name": robot_runtime.display_name,
            "worker_id": pending.worker_id,
            "strategy": pending.strategy_name,
            "symbol": pending.order.symbol,
            "action": pending.order.action.value,
            "side": self._display_side(pending.order.action),
            "quantity": quantity,
            "requested_bar_time": pending.order.signal_time,
            "filled_bar_time": None,
            "fill_price": price,
            "fee": None,
            "slippage_cost": None,
            "status": "CANCELLED",
            "reason": reason,
            "target_notional": target_notional,
            "available_capital": available_capital,
        }

        robot_runtime.execution_log.append(record)

        self._emit(
            EventType.ORDER_CANCELLED,
            source="risk_manager",
            payload=self._json_safe_execution_record(record),
        )

    def _emit_position_rejected(
        self,
        robot_runtime: LiveRobotRuntime,
        worker_runtime: StrategyWorkerRuntime,
        symbol: str,
        reason: str,
    ) -> None:
        self._emit(
            EventType.ORDER_REJECTED,
            source="position_manager",
            payload={
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "worker_id": worker_runtime.worker_id,
                "strategy": worker_runtime.strategy_name,
                "symbol": symbol,
                "reason": reason,
            },
        )

    def _display_side(self, action: OrderAction) -> str:
        if action == OrderAction.BUY_TO_OPEN:
            return "BUY_OPEN_LONG"
        if action == OrderAction.SELL_TO_CLOSE:
            return "SELL_CLOSE_LONG"
        if action == OrderAction.SELL_TO_OPEN:
            return "SELL_OPEN_SHORT"
        if action == OrderAction.BUY_TO_CLOSE:
            return "BUY_CLOSE_SHORT"
        return action.value

    def _emit_market_bar(self, bar: pd.Series) -> None:
        self._emit(
            EventType.MARKET_BAR,
            source="historical_replay",
            payload={
                "timestamp": bar["timestamp"].isoformat(),
                "symbol": self._config.data.symbol,
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar["volume"]),
            },
        )

    def _emit_worker_lifecycle_event(
        self,
        event_type: EventType,
        robot_runtime: LiveRobotRuntime,
        worker_runtime: StrategyWorkerRuntime,
        payload: dict,
    ) -> None:
        self._emit(
            event_type,
            source=worker_runtime.worker_id,
            payload={
                **payload,
                "robot_id": robot_runtime.robot_id,
                "robot_name": robot_runtime.display_name,
                "worker_id": worker_runtime.worker_id,
                "strategy": worker_runtime.strategy_name,
            },
        )

    def _emit(
        self,
        event_type: EventType,
        source: str,
        payload: dict,
    ) -> None:
        event = EventEnvelope(
            event_type=event_type,
            source=source,
            payload=payload,
            correlation_id=self._session_id,
        ).to_dict()

        if self._event_callback is not None:
            self._event_callback(event)

    def _signal_from_payload(self, payload: dict) -> Signal:
        return Signal(
            timestamp=pd.Timestamp(payload["timestamp"]),
            symbol=str(payload["symbol"]),
            signal_type=SignalType(payload["signal_type"]),
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"]),
            metadata=dict(payload.get("metadata", {})),
        )

    def _serialize_market_window(self, market_window: pd.DataFrame) -> list[dict]:
        frame = market_window.copy()
        frame["timestamp"] = frame["timestamp"].astype(str)
        return frame.to_dict(orient="records")

    def _json_safe_execution_record(self, record: dict) -> dict:
        safe_record = dict(record)

        for key in ("timestamp", "requested_bar_time", "filled_bar_time"):
            value = safe_record.get(key)

            if hasattr(value, "isoformat"):
                safe_record[key] = value.isoformat()
            elif value is None:
                safe_record[key] = None
            else:
                safe_record[key] = str(value)

        return safe_record