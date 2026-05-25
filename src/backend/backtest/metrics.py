import math
from typing import Any

import numpy as np
import pandas as pd


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def calculate_metrics(
    equity_curve: pd.DataFrame,
    benchmark_curve: pd.DataFrame,
    trades: list[Any],
    initial_capital: float,
    fees_paid: float,
    slippage_cost: float,
    periods_per_year: int,
) -> dict[str, Any]:
    if equity_curve.empty:
        raise ValueError("Cannot calculate metrics from an empty equity curve.")

    final_equity = float(equity_curve.iloc[-1]["equity"])
    benchmark_final_equity = float(benchmark_curve.iloc[-1]["benchmark_equity"])

    total_return = final_equity / initial_capital - 1.0
    benchmark_return = benchmark_final_equity / initial_capital - 1.0

    trade_pnls = [float(_value(trade, "net_pnl", 0.0) or 0.0) for trade in trades]
    winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
    losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

    gross_profit = sum(winning_trades)
    gross_loss = sum(losing_trades)

    profit_factor = None
    if gross_loss < 0:
        profit_factor = gross_profit / abs(gross_loss)
    elif gross_profit > 0:
        profit_factor = math.inf

    returns = equity_curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0

    sharpe_ratio = None
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe_ratio = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))

    downside_returns = returns[returns < 0]
    sortino_ratio = None
    if len(downside_returns) > 1 and downside_returns.std(ddof=1) > 0:
        sortino_ratio = float(returns.mean() / downside_returns.std(ddof=1) * np.sqrt(periods_per_year))

    max_drawdown = float(equity_curve["drawdown"].min()) if "drawdown" in equity_curve else 0.0

    if "open_position_count" in equity_curve:
        exposure_time_pct = float((equity_curve["open_position_count"] > 0).mean())
    elif "position_quantity" in equity_curve:
        exposure_time_pct = float((equity_curve["position_quantity"].abs() > 0).mean())
    else:
        exposure_time_pct = 0.0

    average_win = float(np.mean(winning_trades)) if winning_trades else 0.0
    average_loss = float(np.mean(losing_trades)) if losing_trades else 0.0

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "benchmark_final_equity": benchmark_final_equity,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "max_drawdown": max_drawdown,
        "number_of_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": len(winning_trades) / len(trades) if trades else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": profit_factor,
        "fees_paid": fees_paid,
        "slippage_cost": slippage_cost,
        "exposure_time_pct": exposure_time_pct,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
    }
