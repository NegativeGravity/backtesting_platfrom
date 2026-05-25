from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from backend.data.market_store import MarketDataStore


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "ma_distance_12",
    "ma_distance_24",
    "volatility_12",
    "volatility_24",
    "volume_z_24",
]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    features = pd.DataFrame(index=df.index)
    features["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    features["return_1"] = close.pct_change(1)
    features["return_3"] = close.pct_change(3)
    features["return_6"] = close.pct_change(6)
    features["return_12"] = close.pct_change(12)
    features["return_24"] = close.pct_change(24)
    features["ma_distance_12"] = close / close.rolling(12).mean() - 1.0
    features["ma_distance_24"] = close / close.rolling(24).mean() - 1.0
    features["volatility_12"] = close.pct_change().rolling(12).std()
    features["volatility_24"] = close.pct_change().rolling(24).std()
    features["volume_z_24"] = (volume - volume.rolling(24).mean()) / volume.rolling(24).std().replace(0, np.nan)
    return features


def build_dataset(frame: pd.DataFrame, horizon: int, min_return: float, fee_rate: float, slippage_bps: float) -> pd.DataFrame:
    features = build_features(frame)
    close = frame["close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    round_trip_cost = (fee_rate * 2.0) + (slippage_bps / 10_000.0 * 2.0)

    features["target"] = (future_return - round_trip_cost > min_return).astype(int)
    features = features.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return features


def metrics(model, scaler, frame: pd.DataFrame) -> dict:
    x = scaler.transform(frame[FEATURE_COLUMNS])
    y = frame["target"].astype(int).to_numpy()
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    output = {
        "rows": int(len(frame)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }

    if len(set(y)) > 1:
        output["roc_auc"] = float(roc_auc_score(y, probabilities))
    else:
        output["roc_auc"] = None

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-config", default="configs/market_data.yaml")
    parser.add_argument("--split-policy", default="recommended")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--min-return", type=float, default=0.001)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output-dir", default="outputs/models")
    args = parser.parse_args()

    store = MarketDataStore.from_yaml(args.market_config)
    train_raw = store.load_split_from_yaml("train", args.market_config, args.split_policy)
    validation_raw = store.load_split_from_yaml("validation", args.market_config, args.split_policy)
    test_raw = store.load_split_from_yaml("test", args.market_config, args.split_policy)

    train = build_dataset(train_raw, args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    validation = build_dataset(validation_raw, args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    test = build_dataset(test_raw, args.horizon, args.min_return, args.fee_rate, args.slippage_bps)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS])
    y_train = train["target"].astype(int).to_numpy()

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(x_train, y_train)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.output_dir) / f"ml_momentum_1h_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "model.joblib")
    joblib.dump(scaler, artifact_dir / "scaler.joblib")

    metadata = {
        "strategy": "ml_momentum",
        "symbol": store._config.symbol,
        "interval": store._config.interval,
        "feature_columns": FEATURE_COLUMNS,
        "horizon": args.horizon,
        "min_return": args.min_return,
        "fee_rate": args.fee_rate,
        "slippage_bps": args.slippage_bps,
        "created_at": timestamp,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifact_dir / "feature_config.json").write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "train": metrics(model, scaler, train),
                "validation": metrics(model, scaler, validation),
                "test": metrics(model, scaler, test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"ARTIFACT_DIR={artifact_dir}")


if __name__ == "__main__":
    main()
