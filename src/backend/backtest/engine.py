import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from backend.backtest.benchmark import build_buy_and_hold_benchmark
from backend.backtest.metrics import calculate_metrics
from backend.backtest.report import BacktestReportWriter
from backend.core.config import AppConfig
from backend.core.time import timestamp_for_run_id
from backend.execution.orders import OrderIntent
from backend.execution.simulator import ExecutionSimulator
from backend.portfolio.portfolio import Portfolio
from backend.risk.position_sizer import PositionSizer
from backend.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        config: AppConfig,
        strategy: BaseStrategy,
        data: pd.DataFrame,
    ) -> None:
        self._config = config
        self._strategy = strategy
        self._strategy_name = strategy.__class__.__name__
        self._data = data.reset_index(drop=True)

        self._portfolio = Portfolio(
            symbol=config.data.symbol,
            initial_capital=config.backtest.initial_capital,
        )
        self._position_sizer = PositionSizer(config.risk)
        self._execution_simulator = ExecutionSimulator(config.execution)
        self._report_writer = BacktestReportWriter(config.reporting.output_dir)

        self._pending_order: OrderIntent | None = None
        self._execution_log: list[dict[str, Any]] = []

    def run(self) -> Path:
        run_id = f"run_{timestamp_for_run_id()}"
        logger.info("Starting backtest: %s", run_id)

        benchmark_curve = build_buy_and_hold_benchmark(
            data=self._data,
            initial_capital=self._config.backtest.initial_capital,
        )

        for index in range(len(self._data)):
            current_bar = self._data.iloc[index]

            self._execute_pending_order(index=index, current_bar=current_bar)

            self._portfolio.mark_to_market(
                timestamp=current_bar["timestamp"],
                close_price=float(current_bar["close"]),
            )

            market_window = self._data.iloc[: index + 1]
            signal = self._strategy.generate_signal(
                market_window=market_window,
                portfolio=self._portfolio,
            )

            if signal.is_actionable and index + 1 < len(self._data):
                order = self._position_sizer.size_order(
                    signal=signal,
                    current_price=float(current_bar["close"]),
                    equity=self._portfolio.equity,
                    cash=self._portfolio.cash,
                    current_position_quantity=self._portfolio.position_quantity,
                )
                self._pending_order = order

            elif signal.is_actionable and index + 1 >= len(self._data):
                self._execution_log.append(
                    {
                        "timestamp": current_bar["timestamp"],
                        "symbol": signal.symbol,
                        "side": signal.signal_type.value,
                        "quantity": None,
                        "requested_bar_time": current_bar["timestamp"],
                        "filled_bar_time": None,
                        "fill_price": None,
                        "fee": None,
                        "slippage_cost": None,
                        "status": "EXPIRED",
                        "reason": "no_next_bar_available",
                    }
                )

        equity_curve_df = pd.DataFrame([asdict(point) for point in self._portfolio.equity_curve])

        metrics = calculate_metrics(
            equity_curve=equity_curve_df,
            benchmark_curve=benchmark_curve,
            trades=self._portfolio.trades,
            initial_capital=self._config.backtest.initial_capital,
            fees_paid=self._portfolio.fees_paid,
            slippage_cost=self._portfolio.slippage_cost,
            periods_per_year=self._config.backtest.periods_per_year,
        )

        summary = {
            "run_id": run_id,
            "dataset": str(self._config.data.path),
            "symbol": self._config.data.symbol,
            "timeframe": self._config.data.timeframe,
            "strategy": self._config.strategy.name,
            "metrics": metrics,
        }

        run_dir = self._report_writer.write(
            run_id=run_id,
            config_snapshot=self._config.raw,
            summary=summary,
            trades=self._portfolio.trades,
            equity_curve=self._portfolio.equity_curve,
            benchmark_curve=benchmark_curve,
            execution_log=self._execution_log,
        )

        logger.info("Backtest completed: %s", run_dir)
        return run_dir

    def _execute_pending_order(self, index: int, current_bar: pd.Series) -> None:
        if self._pending_order is None:
            return

        fill = self._execution_simulator.execute_at_open(
            order=self._pending_order,
            execution_bar=current_bar,
            available_cash=self._portfolio.cash,
        )

        if fill is None:
            self._execution_log.append(
                {
                    "timestamp": current_bar["timestamp"],
                    "symbol": self._pending_order.symbol,
                    "side": self._pending_order.side.value,
                    "quantity": self._pending_order.quantity,
                    "requested_bar_time": self._pending_order.signal_time,
                    "filled_bar_time": None,
                    "fill_price": None,
                    "fee": None,
                    "slippage_cost": None,
                    "status": "REJECTED",
                    "reason": "insufficient_cash_or_invalid_quantity",
                }
            )
            self._pending_order = None
            return

        self._portfolio.apply_fill(fill)

        self._execution_log.append(
            {
                "timestamp": current_bar["timestamp"],
                "symbol": fill.symbol,
                "side": fill.side.value,
                "quantity": fill.quantity,
                "requested_bar_time": self._pending_order.signal_time,
                "filled_bar_time": fill.fill_time,
                "fill_price": fill.fill_price,
                "fee": fill.fee,
                "slippage_cost": fill.slippage_cost,
                "status": "FILLED",
                "reason": fill.reason,
            }
        )

        self._pending_order = None