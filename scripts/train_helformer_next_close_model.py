from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from backend.core.config import load_config
from backend.data.market_store import MarketDataStore
from backend.ml.helformer import HelformerTrainingConfig, train_helformer_next_close_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Helformer next-close model artifact.")
    parser.add_argument("--market-config", default="configs/market_data.docker.yaml")
    parser.add_argument("--app-config", default="configs/backtest.yaml")
    parser.add_argument("--split-policy", default="recommended")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--forecast-horizon", type=int, default=1)
    parser.add_argument("--purge-hours", type=int, default=48)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--validation-days", type=int, default=120)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-validation-samples", type=int, default=1800)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--tuning-epochs", type=int, default=8)
    parser.add_argument("--final-epochs-cap", type=int, default=14)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value


def main() -> int:
    args = parse_args()
    app_config = load_config(args.app_config)

    data = MarketDataStore.from_yaml(args.market_config).load_split_from_yaml(
        split="full",
        market_config_path=args.market_config,
        split_policy=args.split_policy,
        use_cache=True,
    )

    training_config = HelformerTrainingConfig(
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        purge_hours=args.purge_hours,
        n_walk_forward_folds=args.folds,
        fold_validation_days=args.validation_days,
        max_train_samples_per_fold=args.max_train_samples,
        max_validation_samples_per_fold=args.max_validation_samples,
        n_trials=args.n_trials,
        tuning_epochs=args.tuning_epochs,
        final_epochs_cap=args.final_epochs_cap,
        prediction_batch_size=args.prediction_batch_size,
        fee_rate=float(app_config.execution.fee_rate),
        allow_short=bool(args.allow_short),
    )

    artifact = train_helformer_next_close_model(
        data=data,
        config=app_config,
        training_config=training_config,
    )

    print(json.dumps(to_jsonable(artifact), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
