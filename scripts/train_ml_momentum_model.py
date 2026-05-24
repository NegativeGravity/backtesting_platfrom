from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from trading_system.core.config import load_config
from trading_system.core.time import timestamp_for_run_id
from trading_system.data.csv_loader import load_ohlcv_csv
from trading_system.data.market_data import filter_date_range
from trading_system.data.validator import validate_ohlcv
from trading_system.ml.features import FEATURE_COLUMNS, FeatureConfig, build_feature_frame


def time_split(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        raise ValueError("Feature frame is empty.")
    train_end = int(len(frame) * train_ratio)
    val_end = int(len(frame) * (train_ratio + validation_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= len(frame):
        raise ValueError("Invalid train/validation/test split sizes.")
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:val_end].copy(),
        frame.iloc[val_end:].copy(),
    )


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (proba >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "positive_rate": float(pred.mean()),
        "base_rate": float(y_true.mean()),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = None
    return metrics


def select_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_threshold = 0.60
    best_score = -1.0
    for threshold in np.arange(0.50, 0.81, 0.01):
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        # penalize extremely high turnover / always-on models
        trade_rate = pred.mean()
        if trade_rate > 0.35:
            score *= 0.75
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ml_momentum model artifact.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--output-dir", default=None, help="Default: config.ml.model_artifact_dir/model_<timestamp>")
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_ohlcv_csv(config.data.path, config.data.timestamp_column)
    data = filter_date_range(data, config.backtest.start_date, config.backtest.end_date)
    validate_ohlcv(data)

    feature_config = FeatureConfig(
        horizon_bars=config.ml.horizon_bars,
        min_return_threshold=config.ml.min_return_threshold,
        fee_rate=config.execution.fee_rate,
        slippage_bps=config.execution.slippage_bps,
    )
    frame = build_feature_frame(data, feature_config)
    train, val, test = time_split(frame, config.ml.train_ratio, config.ml.validation_ratio)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    x_val = scaler.transform(val[FEATURE_COLUMNS])
    x_test = scaler.transform(test[FEATURE_COLUMNS])
    y_train = train["target"].to_numpy(dtype=int)
    y_val = val["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    val_proba = model.predict_proba(x_val)[:, 1]
    threshold = select_threshold(y_val, val_proba)
    test_proba = model.predict_proba(x_test)[:, 1]

    artifact_dir = Path(args.output_dir) if args.output_dir else config.ml.model_artifact_dir / f"ml_momentum_{timestamp_for_run_id()}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "exit_probability": config.ml.exit_probability,
    }
    joblib.dump(model_bundle, artifact_dir / "model.joblib")

    write_json(artifact_dir / "metadata.json", {
        "strategy": "ml_momentum",
        "symbol": config.data.symbol,
        "timeframe": config.data.timeframe,
        "model_type": "LogisticRegression",
        "artifact_format": "directory_with_model_joblib",
        "selected_threshold": threshold,
        "exit_probability": config.ml.exit_probability,
        "horizon_bars": config.ml.horizon_bars,
        "no_data_leakage_controls": [
            "causal_features_only",
            "time_based_split",
            "scaler_fit_on_train_only",
            "threshold_selected_on_validation_only",
            "test_not_used_for_training_or_threshold_selection",
        ],
    })
    write_json(artifact_dir / "metrics.json", {
        "validation": classification_metrics(y_val, val_proba, threshold),
        "test": classification_metrics(y_test, test_proba, threshold),
    })
    write_json(artifact_dir / "split_info.json", {
        "rows_total": len(frame),
        "rows_train": len(train),
        "rows_validation": len(val),
        "rows_test": len(test),
        "train_start": str(train["timestamp"].iloc[0]),
        "train_end": str(train["timestamp"].iloc[-1]),
        "validation_start": str(val["timestamp"].iloc[0]),
        "validation_end": str(val["timestamp"].iloc[-1]),
        "test_start": str(test["timestamp"].iloc[0]),
        "test_end": str(test["timestamp"].iloc[-1]),
    })
    write_json(artifact_dir / "feature_config.json", {
        "feature_columns": FEATURE_COLUMNS,
        "horizon_bars": config.ml.horizon_bars,
        "min_return_threshold": config.ml.min_return_threshold,
        "fee_rate": config.execution.fee_rate,
        "slippage_bps": config.execution.slippage_bps,
    })

    print(f"ARTIFACT_DIR={artifact_dir.as_posix()}")


if __name__ == "__main__":
    main()
