#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.core.config import load_config
from backend.data.market_store import MarketDataStore
from backend.ml.artifacts import ModelArtifact
from backend.ml.helformer import HelformerTrainingConfig, train_helformer_next_close_model, read_json, write_json


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
    parser.add_argument("--keep-training-working-dir", action="store_true")
    return parser.parse_args()


def artifact_payload(artifact: Any) -> dict[str, Any]:
    payload = asdict(artifact) if hasattr(artifact, "__dataclass_fields__") else dict(artifact)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def main() -> int:
    args = parse_args()
    config = load_config(args.app_config)
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
        fee_rate=float(config.execution.fee_rate),
        allow_short=bool(args.allow_short),
    )

    trained_artifact = train_helformer_next_close_model(
        data=data,
        config=config,
        training_config=training_config,
    )
    stable_artifacts = publish_stable_helformer_artifacts(
        trained_artifact=trained_artifact,
        keep_training_working_dir=bool(args.keep_training_working_dir),
    )
    print(json.dumps({"trained_working_artifact": artifact_payload(trained_artifact), "published_artifacts": [artifact_payload(item) for item in stable_artifacts]}, indent=2, default=str))
    return 0


def publish_stable_helformer_artifacts(trained_artifact: ModelArtifact, keep_training_working_dir: bool = False) -> list[ModelArtifact]:
    source = trained_artifact.artifact_dir
    base_dir = source.parent
    forecaster = _copy_helformer_artifact(
        source=source,
        destination=base_dir / "helformer_next_close_forecaster",
        artifact_role="helformer_next_close_forecaster",
        strategy="next_close_forecaster",
    )
    strategy = _copy_helformer_artifact(
        source=source,
        destination=base_dir / "helformer_momentum_strategy",
        artifact_role="helformer_momentum_strategy",
        strategy="helformer_momentum",
    )
    if not keep_training_working_dir and source.exists() and source not in {forecaster.artifact_dir, strategy.artifact_dir}:
        shutil.rmtree(source)
    return [forecaster, strategy]


def _copy_helformer_artifact(source: Path, destination: Path, artifact_role: str, strategy: str) -> ModelArtifact:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    metadata_path = destination / "metadata.json"
    metadata = read_json(metadata_path)
    metadata.update({
        "artifact_name": destination.name,
        "artifact_role": artifact_role,
        "strategy": strategy,
        "stable_artifact": True,
        "model_file": "model.keras",
    })
    write_json(metadata_path, metadata)
    run_summary_path = destination / "run_summary.json"
    if run_summary_path.exists():
        summary = read_json(run_summary_path)
        summary["artifact_dir"] = str(destination)
        summary["artifact_role"] = artifact_role
        write_json(run_summary_path, summary)
    return ModelArtifact(
        artifact_id=destination.name,
        artifact_dir=destination,
        model_path=destination / "model.keras",
        metadata_path=metadata_path,
        metrics_path=destination / "metrics.json",
        split_info_path=destination / "split_info.json",
        feature_config_path=destination / "feature_config.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
