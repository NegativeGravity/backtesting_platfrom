import logging

from rich.console import Console
from rich.table import Table

from trading_system.core.config import load_config
from trading_system.core.logging import setup_logging
from trading_system.data.csv_loader import load_ohlcv_csv
from trading_system.data.market_data import filter_date_range
from trading_system.data.validator import validate_ohlcv
from trading_system.ml.trainer import train_ml_momentum_model

logger = logging.getLogger(__name__)
console = Console()


def main() -> None:
    setup_logging()
    config = load_config("configs/backtest.yaml")

    logger.info("Loading dataset for ML training: %s", config.data.path)

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

    artifact = train_ml_momentum_model(data=data, config=config)

    table = Table(title="ML Training Completed")
    table.add_column("Field")
    table.add_column("Value", justify="right")

    table.add_row("Artifact ID", artifact.artifact_id)
    table.add_row("Artifact Directory", str(artifact.artifact_dir))
    table.add_row("Model Path", str(artifact.model_path))
    table.add_row("Metrics Path", str(artifact.metrics_path))

    console.print(table)
    console.print(
        "\nRun ML backtest with:\n"
        f"[bold green]python scripts/run_backtest.py "
        f"--strategy ml_momentum "
        f"--model-artifact {artifact.artifact_dir}[/bold green]"
    )


if __name__ == "__main__":
    main()