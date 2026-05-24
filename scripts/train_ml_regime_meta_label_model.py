from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from trading_system.core.config import load_config
from trading_system.core.time import timestamp_for_run_id
from trading_system.data.csv_loader import load_ohlcv_csv
from trading_system.data.market_data import filter_date_range
from trading_system.data.validator import validate_ohlcv

FEATURE_COLUMNS = [
    "ret_1", "logret_1", "ret_3", "logret_3", "ret_6", "logret_6",
    "ret_12", "logret_12", "ret_24", "logret_24", "ret_48", "logret_48",
    "ema_fast_distance", "ema_slow_distance", "trend_slope", "realized_vol",
    "vol_of_vol", "atr_pct", "channel_position", "breakout_pressure",
    "breakdown_pressure", "volume_z", "downside_vol", "upside_vol",
    "return_skew", "return_kurt",
]


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    tr = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def build_features(data: pd.DataFrame, volatility_window: int = 32, trend_window: int = 64) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    returns = close.pct_change()
    log_returns = np.log(close).diff()

    features = pd.DataFrame(index=df.index)
    features["timestamp"] = df["timestamp"]
    features["close"] = close
    for horizon in (1, 3, 6, 12, 24, 48):
        features[f"ret_{horizon}"] = close.pct_change(horizon)
        features[f"logret_{horizon}"] = np.log(close).diff(horizon)

    ema_fast = close.ewm(span=16, adjust=False).mean()
    ema_slow = close.ewm(span=trend_window, adjust=False).mean()
    realized_vol = log_returns.rolling(volatility_window).std()
    range_atr = atr(high, low, close, 14)
    channel_high = high.rolling(trend_window).max().shift(1)
    channel_low = low.rolling(trend_window).min().shift(1)
    channel_width = (channel_high - channel_low).replace(0.0, np.nan)
    volume_mean = volume.rolling(volatility_window).mean()
    volume_std = volume.rolling(volatility_window).std().replace(0.0, np.nan)

    features["ema_fast_distance"] = close / ema_fast - 1.0
    features["ema_slow_distance"] = close / ema_slow - 1.0
    features["trend_slope"] = ema_slow.pct_change(8)
    features["realized_vol"] = realized_vol
    features["vol_of_vol"] = realized_vol.rolling(volatility_window).std()
    features["atr_pct"] = range_atr / close
    features["channel_position"] = (close - channel_low) / channel_width
    features["breakout_pressure"] = (close - channel_high) / range_atr.replace(0.0, np.nan)
    features["breakdown_pressure"] = (channel_low - close) / range_atr.replace(0.0, np.nan)
    features["volume_z"] = (volume - volume_mean) / volume_std
    features["downside_vol"] = log_returns.where(log_returns < 0.0, 0.0).rolling(volatility_window).std()
    features["upside_vol"] = log_returns.where(log_returns > 0.0, 0.0).rolling(volatility_window).std()
    features["return_skew"] = returns.rolling(volatility_window).skew()
    features["return_kurt"] = returns.rolling(volatility_window).kurt()
    return features.replace([np.inf, -np.inf], np.nan)


def add_targets(frame: pd.DataFrame, horizon_bars: int, min_return_threshold: float, fee_rate: float, slippage_bps: float) -> pd.DataFrame:
    future_return = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    round_trip_cost = 2.0 * (fee_rate + slippage_bps / 10_000.0)
    long_edge = future_return - round_trip_cost
    short_edge = -future_return - round_trip_cost
    target = np.where(long_edge > min_return_threshold, 1, np.where(short_edge > min_return_threshold, -1, 0))
    out = frame.copy()
    out["future_return"] = future_return
    out["long_edge"] = long_edge
    out["short_edge"] = short_edge
    out["target"] = target.astype(int)
    return out.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)


def time_split(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(len(frame) * train_ratio)
    val_end = int(len(frame) * (train_ratio + validation_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= len(frame):
        raise ValueError("Invalid split sizes.")
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:val_end].copy(), frame.iloc[val_end:].copy()


def select_thresholds(model: Any, scaler: StandardScaler, validation: pd.DataFrame) -> tuple[float, float, float]:
    x_val = scaler.transform(validation[FEATURE_COLUMNS])
    proba = model.predict_proba(x_val)
    classes = list(model.classes_)
    idx_down = classes.index(-1) if -1 in classes else 0
    idx_up = classes.index(1) if 1 in classes else len(classes) - 1
    y = validation["target"].to_numpy()
    best_long, best_short = 0.58, 0.58
    best_score = -1.0
    for long_t in np.arange(0.46, 0.76, 0.02):
        for short_t in np.arange(0.46, 0.76, 0.02):
            pred = np.zeros(len(validation), dtype=int)
            pred[(proba[:, idx_up] >= long_t) & (proba[:, idx_up] > proba[:, idx_down])] = 1
            pred[(proba[:, idx_down] >= short_t) & (proba[:, idx_down] > proba[:, idx_up])] = -1
            trade_rate = np.mean(pred != 0)
            score = f1_score(y, pred, labels=[-1, 1], average="macro", zero_division=0)
            if trade_rate < 0.01:
                score *= 0.5
            if trade_rate > 0.45:
                score *= 0.75
            if score > best_score:
                best_score = float(score)
                best_long = float(long_t)
                best_short = float(short_t)
    return best_long, best_short, 0.50


def eval_model(model: Any, scaler: StandardScaler, frame: pd.DataFrame, long_t: float, short_t: float) -> dict[str, Any]:
    x = scaler.transform(frame[FEATURE_COLUMNS])
    y = frame["target"].to_numpy()
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    idx_down = classes.index(-1) if -1 in classes else 0
    idx_up = classes.index(1) if 1 in classes else len(classes) - 1
    pred = np.zeros(len(frame), dtype=int)
    pred[(proba[:, idx_up] >= long_t) & (proba[:, idx_up] > proba[:, idx_down])] = 1
    pred[(proba[:, idx_down] >= short_t) & (proba[:, idx_down] > proba[:, idx_up])] = -1
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1_directional": float(f1_score(y, pred, labels=[-1, 1], average="macro", zero_division=0)),
        "trade_rate": float(np.mean(pred != 0)),
        "class_distribution_true": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "class_distribution_pred": {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "confusion_matrix_labels_-1_0_1": confusion_matrix(y, pred, labels=[-1, 0, 1]).tolist(),
        "classification_report": classification_report(y, pred, labels=[-1, 0, 1], zero_division=0, output_dict=True),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ml_regime_meta_label artifact.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--output", default=None, help="Default: outputs/models/ml_regime_meta_label_<timestamp>.joblib")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_ohlcv_csv(config.data.path, config.data.timestamp_column)
    data = filter_date_range(data, config.backtest.start_date, config.backtest.end_date)
    validate_ohlcv(data)

    frame = build_features(data)
    frame = add_targets(
        frame,
        horizon_bars=config.ml.horizon_bars,
        min_return_threshold=config.ml.min_return_threshold,
        fee_rate=config.execution.fee_rate,
        slippage_bps=config.execution.slippage_bps,
    )
    train, val, test = time_split(frame, config.ml.train_ratio, config.ml.validation_ratio)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["target"].to_numpy(dtype=int)

    if args.model == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=450,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    model.fit(x_train, y_train)

    long_t, short_t, exit_t = select_thresholds(model, scaler, val)
    artifact_path = Path(args.output) if args.output else config.ml.model_artifact_dir / f"ml_regime_meta_label_{timestamp_for_run_id()}.joblib"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "long_threshold": long_t,
        "short_threshold": short_t,
        "exit_threshold": exit_t,
        "metadata": {
            "strategy": "ml_regime_meta_label",
            "symbol": config.data.symbol,
            "timeframe": config.data.timeframe,
            "model_type": model.__class__.__name__,
            "horizon_bars": config.ml.horizon_bars,
            "target_labels": {"-1": "short_edge", "0": "flat/no_trade", "1": "long_edge"},
        },
    }
    joblib.dump(artifact, artifact_path)

    metrics = {
        "thresholds": {"long_threshold": long_t, "short_threshold": short_t, "exit_threshold": exit_t},
        "validation": eval_model(model, scaler, val, long_t, short_t),
        "test": eval_model(model, scaler, test, long_t, short_t),
        "split_info": {
            "rows_total": len(frame),
            "rows_train": len(train),
            "rows_validation": len(val),
            "rows_test": len(test),
        },
    }
    metrics_path = artifact_path.with_suffix(".metrics.json")
    write_json(metrics_path, metrics)
    print(f"ARTIFACT_PATH={artifact_path.as_posix()}")
    print(f"METRICS_PATH={metrics_path.as_posix()}")


if __name__ == "__main__":
    main()
