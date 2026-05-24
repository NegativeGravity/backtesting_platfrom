from __future__ import annotations

import argparse
import multiprocessing as mp

from rich.console import Console
from rich.table import Table

from trading_system.core.logging import setup_logging
from trading_system.engine.live_replay import LiveReplayEngine

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiprocessing live-like replay engine.")

    parser.add_argument(
        "--config",
        default="configs/backtest.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["mean_reversion"],
        choices=["mean_reversion", "ml_momentum"],
        help="Strategies to run as isolated bot processes.",
    )
    parser.add_argument(
        "--model-artifact",
        default=None,
        help="Model artifact path required for ml_momentum.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Replay delay per bar in seconds. Use 0 for fastest replay.",
    )

    return parser.parse_args()


def print_event(event: dict) -> None:
    event_type = event["event_type"]

    if event_type in {
        "ENGINE_STARTED",
        "BOT_STARTED",
        "SIGNAL_GENERATED",
        "ORDER_SCHEDULED",
        "ORDER_FILLED",
        "PORTFOLIO_UPDATED",
        "ENGINE_STOPPED",
        "BOT_ERROR",
    }:
        console.print(f"[dim]{event['timestamp']}[/dim] [bold]{event_type}[/bold] {event['payload']}")


def main() -> None:
    setup_logging()
    args = parse_args()

    engine = LiveReplayEngine(
        config_path=args.config,
        strategies=args.strategies,
        model_artifact_path=args.model_artifact,
        replay_delay_seconds=args.delay,
        event_callback=print_event,
    )

    result = engine.run()

    table = Table(title="Live Replay Completed")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Session ID", result.session_id)

    for bot_id, output_dir in result.output_dirs.items():
        table.add_row(bot_id, output_dir)

    console.print(table)


if __name__ == "__main__":
    mp.freeze_support()
    main()