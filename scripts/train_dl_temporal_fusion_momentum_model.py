from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from trading_system.core.config import load_config
from trading_system.core.time import timestamp_for_run_id
from trading_system.data.csv_loader import load_ohlcv_csv
from trading_system.data.market_data import filter_date_range
from trading_system.data.validator import validate_ohlcv

FEATURE_COLUMNS = [
    "log_return_1", "log_return_3", "log_return_12", "body_atr", "range_atr",
    "upper_wick_atr", "lower_wick_atr", "ema_16_distance", "ema_64_distance",
    "ema_spread", "realized_vol_24", "realized_vol_64", "volume_z",
    "channel_position", "breakout_distance_atr", "breakdown_distance_atr",
]


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    previous_close = close.shift(1)
    tr = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().reset_index(drop=True)
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
    features["timestamp"] = df["timestamp"]
    features["close"] = close
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
    return features.replace([np.inf, -np.inf], np.nan)


def add_targets(frame: pd.DataFrame, horizon_bars: int, min_return_threshold: float, fee_rate: float, slippage_bps: float) -> pd.DataFrame:
    future_return = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    round_trip_cost = 2.0 * (fee_rate + slippage_bps / 10_000.0)
    long_edge = future_return - round_trip_cost
    short_edge = -future_return - round_trip_cost
    # map to classes: 0=down/short, 1=flat, 2=up/long for CrossEntropyLoss
    y = np.where(long_edge > min_return_threshold, 2, np.where(short_edge > min_return_threshold, 0, 1))
    out = frame.copy()
    out["target"] = y.astype(int)
    out["future_return"] = future_return
    return out.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)


def time_boundaries(n: int, train_ratio: float, validation_ratio: float) -> tuple[int, int]:
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= n:
        raise ValueError("Invalid split sizes.")
    return train_end, val_end


def make_sequences(frame: pd.DataFrame, sequence_length: int, train_end: int, val_end: int) -> tuple[Any, Any, Any]:
    import torch
    values = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y_all = frame["target"].to_numpy(np.int64)
    train_values = values[:train_end]
    mean = train_values.mean(axis=0).astype(np.float32)
    std = train_values.std(axis=0).astype(np.float32)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32)
    values = (values - mean) / std

    xs, ys, indices = [], [], []
    for end_idx in range(sequence_length - 1, len(frame)):
        xs.append(values[end_idx - sequence_length + 1 : end_idx + 1])
        ys.append(y_all[end_idx])
        indices.append(end_idx)
    x = torch.tensor(np.stack(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.long)
    indices_arr = np.array(indices)

    train_mask = indices_arr < train_end
    val_mask = (indices_arr >= train_end) & (indices_arr < val_end)
    test_mask = indices_arr >= val_end
    return (
        (x[train_mask], y[train_mask]),
        (x[val_mask], y[val_mask]),
        (x[test_mask], y[test_mask]),
        mean,
        std,
    )


class SequenceClassifierModule:
    @staticmethod
    def build(input_size: int, hidden_size: int, dropout: float):
        import torch
        import torch.nn as nn

        class GRUAttentionClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(input_size)
                self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout, batch_first=True)
                self.attn = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1))
                self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 3))

            def forward(self, x):
                x = self.norm(x)
                h, _ = self.gru(x)
                weights = torch.softmax(self.attn(h).squeeze(-1), dim=1).unsqueeze(-1)
                context = torch.sum(h * weights, dim=1)
                return self.head(context)

        return GRUAttentionClassifier()


def batch_iter(x, y, batch_size: int, shuffle: bool = True):
    import torch
    indices = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        yield x[idx], y[idx]


def evaluate(model, x, y) -> dict[str, Any]:
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(x)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        pred = proba.argmax(axis=1)
        true = y.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "class_distribution_true": {str(k): int(v) for k, v in zip(*np.unique(true, return_counts=True))},
        "class_distribution_pred": {str(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "confusion_matrix_labels_0_1_2": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(true, pred, labels=[0, 1, 2], zero_division=0, output_dict=True),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dl_temporal_fusion_momentum TorchScript artifact.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

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
    train_end, val_end = time_boundaries(len(frame), config.ml.train_ratio, config.ml.validation_ratio)
    (x_train, y_train), (x_val, y_val), (x_test, y_test), mean, std = make_sequences(frame, args.sequence_length, train_end, val_end)

    model = SequenceClassifierModule.build(len(FEATURE_COLUMNS), args.hidden_size, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = torch.bincount(y_train, minlength=3).float()
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_state = None
    best_val = -1.0
    patience = 7
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in batch_iter(x_train, y_train, args.batch_size, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = evaluate(model, x_val, y_val)
        score = val_metrics["macro_f1"]
        print(f"epoch={epoch:03d} loss={np.mean(losses):.5f} val_macro_f1={score:.4f}")
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else config.ml.model_artifact_dir / f"dl_temporal_fusion_momentum_{timestamp_for_run_id()}"
    output_dir.mkdir(parents=True, exist_ok=False)

    example = torch.zeros(1, args.sequence_length, len(FEATURE_COLUMNS), dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save(str(output_dir / "model.pt"))

    metadata = {
        "strategy": "dl_temporal_fusion_momentum",
        "symbol": config.data.symbol,
        "timeframe": config.data.timeframe,
        "model_type": "GRUAttentionClassifierTorchScript",
        "sequence_length": args.sequence_length,
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "long_threshold": 0.60,
        "short_threshold": 0.60,
        "exit_threshold": 0.48,
        "max_entropy": 0.72,
        "class_mapping": {"0": "down/short", "1": "flat", "2": "up/long"},
        "horizon_bars": config.ml.horizon_bars,
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "metrics.json", {
        "validation": evaluate(model, x_val, y_val),
        "test": evaluate(model, x_test, y_test),
        "split_info": {
            "rows_total": len(frame),
            "sequences_train": len(x_train),
            "sequences_validation": len(x_val),
            "sequences_test": len(x_test),
        },
    })

    print(f"ARTIFACT_DIR={output_dir.as_posix()}")


if __name__ == "__main__":
    main()
