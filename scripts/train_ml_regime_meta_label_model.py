from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from backend.data.market_store import MarketDataStore


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "ret_48",
    "ema_fast_distance",
    "ema_slow_distance",
    "trend_slope",
    "realized_vol",
    "vol_of_vol",
    "atr_pct",
    "channel_position",
    "breakout_pressure",
    "breakdown_pressure",
    "volume_z",
    "downside_vol",
    "upside_vol",
    "return_skew",
    "return_kurt",
]


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False).mean()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    returns = close.pct_change()
    log_returns = np.log(close).diff()

    features = pd.DataFrame(index=df.index)
    features["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for horizon in (1, 3, 6, 12, 24, 48):
        features[f"ret_{horizon}"] = close.pct_change(horizon)

    ema_fast = close.ewm(span=16, adjust=False).mean()
    ema_slow = close.ewm(span=64, adjust=False).mean()
    realized_vol = log_returns.rolling(32).std()
    range_atr = atr(high, low, close, 14)
    channel_high = high.rolling(64).max().shift(1)
    channel_low = low.rolling(64).min().shift(1)
    channel_width = (channel_high - channel_low).replace(0.0, np.nan)
    volume_mean = volume.rolling(48).mean()
    volume_std = volume.rolling(48).std().replace(0.0, np.nan)

    features["ema_fast_distance"] = close / ema_fast - 1.0
    features["ema_slow_distance"] = close / ema_slow - 1.0
    features["trend_slope"] = ema_slow.pct_change(8)
    features["realized_vol"] = realized_vol
    features["vol_of_vol"] = realized_vol.rolling(32).std()
    features["atr_pct"] = range_atr / close
    features["channel_position"] = (close - channel_low) / channel_width
    features["breakout_pressure"] = (close - channel_high) / range_atr.replace(0.0, np.nan)
    features["breakdown_pressure"] = (channel_low - close) / range_atr.replace(0.0, np.nan)
    features["volume_z"] = (volume - volume_mean) / volume_std
    features["downside_vol"] = log_returns.where(log_returns < 0.0, 0.0).rolling(32).std()
    features["upside_vol"] = log_returns.where(log_returns > 0.0, 0.0).rolling(32).std()
    features["return_skew"] = returns.rolling(32).skew()
    features["return_kurt"] = returns.rolling(32).kurt()

    return features


def build_dataset(frame: pd.DataFrame, horizon: int, min_return: float, fee_rate: float, slippage_bps: float) -> pd.DataFrame:
    features = build_features(frame)
    close = frame["close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    round_trip_cost = (fee_rate * 2.0) + (slippage_bps / 10_000.0 * 2.0)

    net_forward = future_return - round_trip_cost
    target = np.where(net_forward > min_return, 1, np.where(net_forward < -min_return, -1, 0))
    features["target"] = target
    return features.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)


def describe(model, scaler, frame: pd.DataFrame) -> dict:
    x = scaler.transform(frame[FEATURE_COLUMNS])
    y = frame["target"].astype(int).to_numpy()
    pred = model.predict(x)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        pred,
        labels=[-1, 0, 1],
        zero_division=0,
    )
    return {
        "rows": int(len(frame)),
        "class_distribution": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "per_class": {
            "-1": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
            "0": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
            "1": {"precision": float(precision[2]), "recall": float(recall[2]), "f1": float(f1[2]), "support": int(support[2])},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-config", default="configs/market_data.yaml")
    parser.add_argument("--split-policy", default="recommended")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--min-return", type=float, default=0.0012)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--output-dir", default="outputs/models")
    args = parser.parse_args()

    store = MarketDataStore.from_yaml(args.market_config)
    train = build_dataset(store.load_split_from_yaml("train", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    validation = build_dataset(store.load_split_from_yaml("validation", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    test = build_dataset(store.load_split_from_yaml("test", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["target"].astype(int).to_numpy()

    if args.model == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.035,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )

    model.fit(x_train, y_train)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_path = Path(args.output_dir) / f"ml_regime_meta_label_1h_{timestamp}.joblib"
    metrics_path = artifact_path.with_suffix(".metrics.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "long_threshold": 0.58,
        "short_threshold": 0.58,
        "exit_threshold": 0.50,
        "metadata": {
            "strategy": "ml_regime_meta_label",
            "symbol": store._config.symbol,
            "interval": store._config.interval,
            "horizon": args.horizon,
            "min_return": args.min_return,
            "fee_rate": args.fee_rate,
            "slippage_bps": args.slippage_bps,
            "created_at": timestamp,
        },
    }
    joblib.dump(artifact, artifact_path)

    metrics = {
        "train": describe(model, scaler, train),
        "validation": describe(model, scaler, validation),
        "test": describe(model, scaler, test),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"ARTIFACT_PATH={artifact_path}")
    print(f"METRICS_PATH={metrics_path}")


if __name__ == "__main__":
    main()
