from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from backend.data.market_store import MarketDataStore


FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_3",
    "log_return_12",
    "body_atr",
    "range_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "ema_16_distance",
    "ema_64_distance",
    "ema_spread",
    "realized_vol_24",
    "realized_vol_64",
    "volume_z",
    "channel_position",
    "breakout_distance_atr",
    "breakdown_distance_atr",
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
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)
    log_close = np.log(close)
    log_return = log_close.diff()
    range_atr = atr(high, low, close, 14)

    ema_16 = close.ewm(span=16, adjust=False).mean()
    ema_64 = close.ewm(span=64, adjust=False).mean()
    volume_mean = volume.rolling(48).mean()
    volume_std = volume.rolling(48).std().replace(0.0, np.nan)
    channel_high = high.rolling(64).max().shift(1)
    channel_low = low.rolling(64).min().shift(1)
    channel_width = (channel_high - channel_low).replace(0.0, np.nan)

    features = pd.DataFrame(index=df.index)
    features["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    features["log_return_1"] = log_return
    features["log_return_3"] = log_close.diff(3)
    features["log_return_12"] = log_close.diff(12)
    features["body_atr"] = (close - open_) / range_atr.replace(0.0, np.nan)
    features["range_atr"] = (high - low) / range_atr.replace(0.0, np.nan)
    features["upper_wick_atr"] = (high - np.maximum(open_, close)) / range_atr.replace(0.0, np.nan)
    features["lower_wick_atr"] = (np.minimum(open_, close) - low) / range_atr.replace(0.0, np.nan)
    features["ema_16_distance"] = close / ema_16 - 1.0
    features["ema_64_distance"] = close / ema_64 - 1.0
    features["ema_spread"] = ema_16 / ema_64 - 1.0
    features["realized_vol_24"] = log_return.rolling(24).std()
    features["realized_vol_64"] = log_return.rolling(64).std()
    features["volume_z"] = (volume - volume_mean) / volume_std
    features["channel_position"] = (close - channel_low) / channel_width
    features["breakout_distance_atr"] = (close - channel_high) / range_atr.replace(0.0, np.nan)
    features["breakdown_distance_atr"] = (channel_low - close) / range_atr.replace(0.0, np.nan)
    return features


def build_xy(frame: pd.DataFrame, horizon: int, min_return: float, fee_rate: float, slippage_bps: float) -> tuple[pd.DataFrame, np.ndarray]:
    features = build_features(frame)
    close = frame["close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    round_trip_cost = (fee_rate * 2.0) + (slippage_bps / 10_000.0 * 2.0)
    net_forward = future_return - round_trip_cost
    target = np.where(net_forward > min_return, 2, np.where(net_forward < -min_return, 0, 1))
    features["target"] = target
    dataset = features.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return dataset[FEATURE_COLUMNS].astype(np.float32), dataset["target"].astype(np.int64).to_numpy()


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, sequence_length: int) -> None:
        self.x = x
        self.y = y
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.x) - self.sequence_length + 1)

    def __getitem__(self, index: int):
        end = index + self.sequence_length
        return torch.from_numpy(self.x[index:end]), torch.tensor(self.y[end - 1], dtype=torch.long)


class TemporalFusionLite(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states, _ = self.gru(x)
        weights = torch.softmax(self.attention(states), dim=1)
        context = torch.sum(states * weights, dim=1)
        return self.head(context)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_pred: list[int] = []
    all_y: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_pred.extend(pred)
            all_y.extend(y.numpy().tolist())

    return {
        "rows": len(all_y),
        "accuracy": float(accuracy_score(all_y, all_pred)) if all_y else 0.0,
        "macro_f1": float(f1_score(all_y, all_pred, average="macro", zero_division=0)) if all_y else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-config", default="configs/market_data.yaml")
    parser.add_argument("--split-policy", default="recommended")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--min-return", type=float, default=0.0012)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output-dir", default="outputs/models")
    args = parser.parse_args()

    store = MarketDataStore.from_yaml(args.market_config)
    train_features, train_y = build_xy(store.load_split_from_yaml("train", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    val_features, val_y = build_xy(store.load_split_from_yaml("validation", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)
    test_features, test_y = build_xy(store.load_split_from_yaml("test", args.market_config, args.split_policy), args.horizon, args.min_return, args.fee_rate, args.slippage_bps)

    feature_mean = train_features.mean(axis=0).to_numpy(dtype=np.float32)
    feature_std = train_features.std(axis=0).replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def norm(features: pd.DataFrame) -> np.ndarray:
        return ((features.to_numpy(dtype=np.float32) - feature_mean) / feature_std).astype(np.float32)

    train_ds = SequenceDataset(norm(train_features), train_y, args.sequence_length)
    val_ds = SequenceDataset(norm(val_features), val_y, args.sequence_length)
    test_ds = SequenceDataset(norm(test_features), test_y, args.sequence_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalFusionLite(len(FEATURE_COLUMNS), args.hidden_size, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(f"epoch={epoch} loss={total_loss / max(1, len(train_loader)):.6f} val_macro_f1={val_metrics['macro_f1']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    artifact_dir = Path(args.output_dir) / f"dl_temporal_fusion_momentum_1h_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    scripted = torch.jit.script(model.cpu())
    scripted.save(str(artifact_dir / "model.pt"))

    metadata = {
        "strategy": "dl_temporal_fusion_momentum",
        "symbol": store._config.symbol,
        "interval": store._config.interval,
        "sequence_length": args.sequence_length,
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "long_threshold": 0.60,
        "short_threshold": 0.60,
        "exit_threshold": 0.48,
        "max_entropy": 0.72,
        "horizon": args.horizon,
        "min_return": args.min_return,
        "created_at": timestamp,
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifact_dir / "metrics.json").write_text(
        json.dumps(
            {
                "validation": evaluate(model.to(device), val_loader, device),
                "test": evaluate(model.to(device), test_loader, device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"ARTIFACT_DIR={artifact_dir}")


if __name__ == "__main__":
    main()
