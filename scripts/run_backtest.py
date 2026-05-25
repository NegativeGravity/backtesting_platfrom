import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from backend.backtest.engine import BacktestEngine
from backend.core.config import load_config
from backend.core.logging import setup_logging
from backend.core.paths import resolve_project_path
from backend.data.csv_loader import load_ohlcv_csv
from backend.data.market_data import filter_date_range
from backend.data.validator import validate_ohlcv
from backend.strategy.factory import create_strategy

logger = logging.getLogger(__name__)
console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a research-grade backtest.")
    parser.add_argument(
        "--config",
        default="configs/backtest.yaml",
        help="Path to backtest YAML config.",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["mean_reversion", "ml_momentum"],
        help="Strategy override.",
    )
    parser.add_argument(
        "--model-artifact",
        default=None,
        help="Path to saved ML model artifact directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    setup_logging()

    logger.info("Loading data from %s", config.data.path)

    data = load_ohlcv_csv(
        path=config.data.path,
        timestamp_column=config.data.timestamp_column,
    )
    data = filter_date_range(
        data=data,
        start_date=config.backtest.start_date,
        end_date=config.backtest.end_date,
    )
    validate_ohlcv(data)

    model_artifact_path = (
        resolve_project_path(args.model_artifact)
        if args.model_artifact is not None
        else None
    )

    strategy = create_strategy(
        config=config,
        strategy_name=args.strategy,
        model_artifact_path=model_artifact_path,
    )

    engine = BacktestEngine(
        config=config,
        strategy=strategy,
        data=data,
    )

    run_dir = engine.run()
    print_summary(run_dir)


def print_summary(run_dir: Path) -> None:
    summary_path = run_dir / "summary.json"

    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    metrics = summary["metrics"]

    table = Table(title="Backtest Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Run ID", summary["run_id"])
    table.add_row("Symbol", summary["symbol"])
    table.add_row("Strategy", summary["strategy"])
    table.add_row("Final Equity", f"{metrics['final_equity']:,.2f}")
    table.add_row("Buy & Hold Equity", f"{metrics['benchmark_final_equity']:,.2f}")
    table.add_row("Strategy Return", f"{metrics['total_return']:.2%}")
    table.add_row("Buy & Hold Return", f"{metrics['benchmark_return']:.2%}")
    table.add_row("Excess Return", f"{metrics['excess_return']:.2%}")
    table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    table.add_row("Trades", str(metrics["number_of_trades"]))
    table.add_row("Win Rate", f"{metrics['win_rate']:.2%}")
    table.add_row("Profit Factor", str(metrics["profit_factor"]))
    table.add_row("Fees Paid", f"{metrics['fees_paid']:,.2f}")
    table.add_row("Slippage Cost", f"{metrics['slippage_cost']:,.2f}")

    console.print(table)
    console.print(f"\nReport saved to: [bold green]{run_dir}[/bold green]")


if __name__ == "__main__":
    main()