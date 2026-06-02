from __future__ import annotations

import gc
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from backend.core.config import AppConfig
from backend.core.time import timestamp_for_run_id
from backend.ml.artifacts import ModelArtifact

SEED = 42


@dataclass(frozen=True)
class HelformerTrainingConfig:
    runtime_profile: str = "local_light_train_2020_2024_test_2025_24h_window_real_close_directional_loss"
    development_start: str = "2020-01-01"
    test_start: str = "2025-01-01"
    test_end: str = "2026-01-01"
    window_size: int = 24
    forecast_horizon: int = 1
    purge_hours: int = 48
    n_walk_forward_folds: int = 2
    fold_validation_days: int = 120
    min_train_days_per_fold: int = 365
    max_train_samples_per_fold: int = 6000
    max_validation_samples_per_fold: int = 1800
    max_sequence_imputed_ratio: float = 0.05
    max_sequence_imputed_run: int = 1
    n_trials: int = 5
    tuning_epochs: int = 8
    final_epochs_cap: int = 14
    min_final_epochs: int = 5
    extra_final_epochs: int = 1
    patience: int = 3
    min_delta: float = 3e-5
    prediction_batch_size: int = 512
    fee_rate: float = 0.0004
    allow_short: bool = False
    max_runtime_minutes: int = 210
    reserved_final_minutes: int = 35
    calibrator_warmup_folds: bool = True
    max_calibrator_epochs: int = 4
    loss_log_weight: float = 0.35
    loss_price_weight: float = 1.0
    loss_direction_weight: float = 2.5
    loss_direction_temperature: float = 0.0015
    loss_direction_min_abs_move: float = 0.0008
    loss_price_huber_delta: float = 0.003
    loss_log_huber_delta: float = 1.0
    fallback_params: dict[str, Any] = field(default_factory=lambda: {
        "model_dim": 64,
        "num_heads": 4,
        "num_blocks": 1,
        "lstm_units": 64,
        "dense_units": 64,
        "dropout_rate": 0.12,
        "learning_rate": 7e-4,
        "weight_decay": 2e-5,
        "batch_size": 128,
    })


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_hours: int


@dataclass
class RuntimeBudget:
    max_minutes: float

    def __post_init__(self) -> None:
        self.started_at = time.time()
        self.max_seconds = float(self.max_minutes) * 60.0

    def elapsed_minutes(self) -> float:
        return (time.time() - self.started_at) / 60.0

    def remaining_minutes(self) -> float:
        return max(0.0, self.max_seconds / 60.0 - self.elapsed_minutes())

    def has_reserve(self, reserve_minutes: float) -> bool:
        return self.remaining_minutes() > float(reserve_minutes)


@dataclass
class RegimeAwareCalibrator:
    low_vol_threshold: float
    high_vol_threshold: float
    base_alpha: float = 0.06
    fast_alpha: float = 0.34
    shock_z: float = 2.25
    max_abs_move: float = 0.08

    def __post_init__(self) -> None:
        self.bias = {"low": 0.0, "normal": 0.0, "high": 0.0}
        self.error_scale = {"low": 0.001, "normal": 0.0015, "high": 0.0025}
        self.recent_abs_error = 0.0
        self.count = {"low": 0, "normal": 0, "high": 0}

    def classify(self, volatility: float) -> str:
        if volatility <= self.low_vol_threshold:
            return "low"
        if volatility >= self.high_vol_threshold:
            return "high"
        return "normal"

    def predict(self, raw_log_move: float, volatility: float) -> tuple[float, str, float]:
        regime = self.classify(float(volatility))
        scale = max(self.error_scale[regime], 1e-6)
        stress = max(0.0, self.recent_abs_error / (self.shock_z * scale) - 1.0)
        confidence = 1.0 / (1.0 + stress)
        calibrated = confidence * (float(raw_log_move) + self.bias[regime])
        cap = min(self.max_abs_move, max(3.5 * float(volatility), 0.0025))
        return float(np.clip(calibrated, -cap, cap)), regime, float(confidence)

    def update(self, actual_log_move: float, predicted_log_move: float, regime: str) -> None:
        residual = float(actual_log_move) - float(predicted_log_move)
        abs_error = abs(residual)
        current_scale = max(self.error_scale[regime], 1e-6)
        alpha = self.fast_alpha if abs_error > self.shock_z * current_scale else self.base_alpha
        self.bias[regime] = (1.0 - alpha) * self.bias[regime] + alpha * residual
        self.error_scale[regime] = (1.0 - alpha) * self.error_scale[regime] + alpha * abs_error
        self.recent_abs_error = 0.75 * self.recent_abs_error + 0.25 * abs_error
        self.count[regime] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "low_vol_threshold": self.low_vol_threshold,
            "high_vol_threshold": self.high_vol_threshold,
            "base_alpha": self.base_alpha,
            "fast_alpha": self.fast_alpha,
            "shock_z": self.shock_z,
            "max_abs_move": self.max_abs_move,
            "bias": dict(self.bias),
            "error_scale": dict(self.error_scale),
            "recent_abs_error": self.recent_abs_error,
            "count": dict(self.count),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "RegimeAwareCalibrator":
        calibrator = cls(
            low_vol_threshold=float(state.get("low_vol_threshold", 0.001)),
            high_vol_threshold=float(state.get("high_vol_threshold", 0.003)),
            base_alpha=float(state.get("base_alpha", 0.06)),
            fast_alpha=float(state.get("fast_alpha", 0.34)),
            shock_z=float(state.get("shock_z", 2.25)),
            max_abs_move=float(state.get("max_abs_move", 0.08)),
        )
        calibrator.bias.update({key: float(value) for key, value in dict(state.get("bias", {})).items() if key in calibrator.bias})
        calibrator.error_scale.update({key: float(value) for key, value in dict(state.get("error_scale", {})).items() if key in calibrator.error_scale})
        calibrator.recent_abs_error = float(state.get("recent_abs_error", 0.0))
        calibrator.count.update({key: int(value) for key, value in dict(state.get("count", {})).items() if key in calibrator.count})
        return calibrator


def seed_runtime(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        try:
            if tf.config.list_physical_devices("GPU"):
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
    except Exception:
        pass


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean()


def make_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def longest_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def normalize_helformer_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Helformer training data is empty.")
    frame = data.copy()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp", drop=True)
    elif "ts" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.set_index("timestamp", drop=True)
    else:
        frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index(kind="mergesort")
    frame = frame[~frame.index.duplicated(keep="last")]
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    if "quote_asset_volume" not in frame.columns:
        if "quote_volume" in frame.columns:
            frame["quote_asset_volume"] = pd.to_numeric(frame["quote_volume"], errors="coerce")
        else:
            frame["quote_asset_volume"] = frame["volume"] * frame["close"]
    if "number_of_trades" not in frame.columns:
        if "trade_count" in frame.columns:
            frame["number_of_trades"] = pd.to_numeric(frame["trade_count"], errors="coerce")
        else:
            frame["number_of_trades"] = 0.0
    if "taker_buy_base_asset_volume" not in frame.columns:
        if "taker_buy_volume" in frame.columns:
            frame["taker_buy_base_asset_volume"] = pd.to_numeric(frame["taker_buy_volume"], errors="coerce")
        else:
            frame["taker_buy_base_asset_volume"] = 0.0
    if "taker_buy_quote_asset_volume" not in frame.columns:
        if "taker_buy_quote_volume" in frame.columns:
            frame["taker_buy_quote_asset_volume"] = pd.to_numeric(frame["taker_buy_quote_volume"], errors="coerce")
        else:
            frame["taker_buy_quote_asset_volume"] = 0.5 * frame["quote_asset_volume"]
    if "is_imputed" not in frame.columns:
        frame["is_imputed"] = 0.0
    numeric = [
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "is_imputed",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype("float64")
    return frame


def add_causal_features(data: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    out = normalize_helformer_frame(data)
    out["is_imputed"] = out["is_imputed"].fillna(0.0).astype("float32")
    close = out["close"].astype(float)
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float).clip(lower=0.0)
    quote_volume = out["quote_asset_volume"].astype(float).clip(lower=0.0)
    trades = out["number_of_trades"].astype(float).clip(lower=0.0)
    log_close = np.log(close)
    normalized_now = log_close.diff().fillna(0.0)
    out["log_close"] = log_close
    out["normalized_now"] = normalized_now
    out["range_log"] = np.log((high + eps) / (low + eps))
    out["body_log"] = np.log((close + eps) / (open_ + eps))
    out["upper_wick_log"] = np.log((high + eps) / (np.maximum(open_, close) + eps))
    out["lower_wick_log"] = np.log((np.minimum(open_, close) + eps) / (low + eps))
    out["close_location"] = ((close - low) / (high - low + eps)).clip(0.0, 1.0)
    out["log_volume"] = np.log1p(volume)
    out["log_quote_volume"] = np.log1p(quote_volume)
    out["log_trades"] = np.log1p(trades)
    out["taker_buy_ratio"] = (out["taker_buy_quote_asset_volume"] / quote_volume.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    out["taker_buy_imbalance"] = 2.0 * out["taker_buy_ratio"] - 1.0
    for window in [3, 6, 12, 24, 72, 168]:
        rolling_volume_mean = out["log_volume"].rolling(window, min_periods=max(3, window // 4)).mean()
        rolling_volume_std = out["log_volume"].rolling(window, min_periods=max(3, window // 4)).std().replace(0.0, np.nan)
        out[f"momentum_{window}h"] = log_close - log_close.shift(window)
        out[f"realized_vol_{window}h"] = normalized_now.rolling(window, min_periods=max(3, window // 4)).std()
        out[f"range_mean_{window}h"] = out["range_log"].rolling(window, min_periods=max(3, window // 4)).mean()
        out[f"volume_z_{window}h"] = ((out["log_volume"] - rolling_volume_mean) / rolling_volume_std).fillna(0.0)
    out["ema_12_gap"] = log_close - np.log(ema(close, 12))
    out["ema_24_gap"] = log_close - np.log(ema(close, 24))
    out["ema_72_gap"] = log_close - np.log(ema(close, 72))
    out["ema_168_gap"] = log_close - np.log(ema(close, 168))
    macd_fast = ema(close, 12)
    macd_slow = ema(close, 26)
    macd = np.log(macd_fast) - np.log(macd_slow)
    out["macd"] = macd
    out["macd_signal_gap"] = macd - ema(macd, 9)
    out["rsi_14"] = make_rsi(close, 14) / 100.0
    out["rsi_48"] = make_rsi(close, 48) / 100.0
    hour = out.index.hour.to_numpy(dtype=float)
    dayofweek = out.index.dayofweek.to_numpy(dtype=float)
    dayofyear = out.index.dayofyear.to_numpy(dtype=float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    out["dayofweek_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
    out["dayofweek_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
    out["dayofyear_sin"] = np.sin(2.0 * np.pi * dayofyear / 365.25)
    out["dayofyear_cos"] = np.cos(2.0 * np.pi * dayofyear / 365.25)
    out["current_close"] = close
    out["target_close"] = close.shift(-1)
    out["target_time"] = out.index.to_series().shift(-1)
    out["target_is_imputed"] = out["is_imputed"].shift(-1).fillna(1.0).astype("float32")
    out["supervised_target_ok"] = ((out["is_imputed"] < 0.5) & (out["target_is_imputed"] < 0.5)).astype("float32")
    out["normalized_next_close"] = np.log(out["target_close"] / out["current_close"])
    return out.replace([np.inf, -np.inf], np.nan)


def infer_feature_columns(features: pd.DataFrame) -> list[str]:
    metadata_columns = {
        "close_time",
        "ignore",
        "current_close",
        "target_close",
        "target_time",
        "target_is_imputed",
        "supervised_target_ok",
        "normalized_next_close",
    }
    columns = [column for column in features.columns if column not in metadata_columns]
    return ["normalized_now"] + [column for column in columns if column != "normalized_now"]


def prepare_supervised_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = add_causal_features(data)
    feature_columns = infer_feature_columns(features)
    required_columns = feature_columns + ["current_close", "target_close", "target_time", "normalized_next_close", "supervised_target_ok"]
    features = features.dropna(subset=required_columns).copy()
    features[feature_columns] = features[feature_columns].astype("float32")
    features["normalized_next_close"] = features["normalized_next_close"].astype("float32")
    features["supervised_target_ok"] = features["supervised_target_ok"].astype("float32")
    return features, feature_columns


def utc_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def target_mask(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    target_time = pd.to_datetime(frame["target_time"], utc=True)
    return (target_time >= start) & (target_time < end) & (frame["supervised_target_ok"].to_numpy(dtype=float) > 0.5)


def fit_scalers(frame: pd.DataFrame, columns: list[str], fit_mask: pd.Series) -> tuple[StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(frame.loc[fit_mask, columns].values)
    y_scaler.fit(frame.loc[fit_mask, ["normalized_next_close"]].values)
    return x_scaler, y_scaler


def make_loss_target(y_scaled: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.asarray(y_scaled).reshape(-1),
        meta["current_close"].to_numpy(dtype=np.float32),
        meta["actual_close"].to_numpy(dtype=np.float32),
        meta["normalized_next_close"].to_numpy(dtype=np.float32),
    ]).astype("float32")


def build_sequences(
    frame: pd.DataFrame,
    columns: list[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    start: pd.Timestamp,
    end: pd.Timestamp,
    training_config: HelformerTrainingConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    window_size = int(training_config.window_size)
    x_all = x_scaler.transform(frame[columns].values).astype(np.float32)
    y_all = y_scaler.transform(frame[["normalized_next_close"]].values).ravel().astype(np.float32)
    mask = target_mask(frame, start, end).to_numpy()
    positions = np.flatnonzero(mask)
    positions = positions[positions >= window_size - 1]
    imputed_values = frame["is_imputed"].fillna(1.0).to_numpy(dtype=np.float32)
    max_imputed_count = int(math.floor(window_size * float(training_config.max_sequence_imputed_ratio)))
    max_imputed_run = int(training_config.max_sequence_imputed_run)
    x_items = []
    y_items = []
    rows = []
    skipped = 0
    index_values = frame.index.to_numpy()
    target_times = pd.to_datetime(frame["target_time"], utc=True).to_numpy()
    current_close = frame["current_close"].to_numpy(dtype=float)
    target_close = frame["target_close"].to_numpy(dtype=float)
    normalized_next_close = frame["normalized_next_close"].to_numpy(dtype=float)
    regime_vol = frame["realized_vol_24h"].fillna(frame["realized_vol_24h"].median()).to_numpy(dtype=float)
    for position in positions:
        left = position - window_size + 1
        right = position + 1
        seq_imputed = imputed_values[left:right] > 0.5
        seq_imputed_count = int(seq_imputed.sum())
        seq_imputed_run = longest_true_run(seq_imputed)
        if seq_imputed_count > max_imputed_count or seq_imputed_run > max_imputed_run:
            skipped += 1
            continue
        x_items.append(x_all[left:right])
        y_items.append(y_all[position])
        rows.append({
            "decision_time": pd.Timestamp(index_values[position]),
            "target_time": pd.Timestamp(target_times[position]),
            "current_close": float(current_close[position]),
            "actual_close": float(target_close[position]),
            "normalized_next_close": float(normalized_next_close[position]),
            "regime_vol": float(regime_vol[position]),
            "sequence_imputed_count": seq_imputed_count,
            "sequence_imputed_ratio": float(seq_imputed_count / window_size),
            "sequence_imputed_run": int(seq_imputed_run),
        })
    if not x_items:
        raise RuntimeError(f"No valid sequences were built for range {start} to {end}.")
    x = np.asarray(x_items, dtype=np.float32)
    y = np.asarray(y_items, dtype=np.float32)
    meta = pd.DataFrame(rows)
    meta.attrs["skipped_for_imputation"] = int(skipped)
    return x, y, meta


def tail_limit(x: np.ndarray, y: np.ndarray, meta: pd.DataFrame, max_samples: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    attrs = dict(getattr(meta, "attrs", {}))
    if max_samples <= 0 or len(x) <= max_samples:
        result_meta = meta.reset_index(drop=True)
        result_meta.attrs.update(attrs)
        return x, y, result_meta
    result_meta = meta.iloc[-max_samples:].reset_index(drop=True)
    result_meta.attrs.update(attrs)
    return x[-max_samples:], y[-max_samples:], result_meta


def make_purged_walk_forward_folds(training_config: HelformerTrainingConfig) -> list[WalkForwardFold]:
    development_start = utc_timestamp(training_config.development_start)
    test_start = utc_timestamp(training_config.test_start)
    validation_span = pd.Timedelta(days=int(training_config.fold_validation_days))
    purge = pd.Timedelta(hours=int(training_config.purge_hours))
    min_train = pd.Timedelta(days=int(training_config.min_train_days_per_fold))
    first_validation_start = test_start - validation_span * int(training_config.n_walk_forward_folds)
    folds = []
    for fold_id in range(int(training_config.n_walk_forward_folds)):
        validation_start = first_validation_start + validation_span * fold_id
        validation_end = min(validation_start + validation_span, test_start)
        train_start = development_start
        train_end = validation_start - purge
        if train_end - train_start < min_train:
            raise ValueError("Fold has less training history than min_train_days_per_fold.")
        folds.append(WalkForwardFold(
            name=f"fold_{fold_id + 1}",
            train_start=train_start,
            train_end=train_end,
            validation_start=validation_start,
            validation_end=validation_end,
            purge_hours=int(training_config.purge_hours),
        ))
    return folds


def build_fold_dataset(frame: pd.DataFrame, columns: list[str], fold: WalkForwardFold, training_config: HelformerTrainingConfig) -> dict[str, Any]:
    train_fit_mask = target_mask(frame, fold.train_start, fold.train_end)
    x_scaler, y_scaler = fit_scalers(frame, columns, train_fit_mask)
    x_train, y_train, meta_train = build_sequences(frame, columns, x_scaler, y_scaler, fold.train_start, fold.train_end, training_config)
    x_val, y_val, meta_val = build_sequences(frame, columns, x_scaler, y_scaler, fold.validation_start, fold.validation_end, training_config)
    x_train, y_train, meta_train = tail_limit(x_train, y_train, meta_train, int(training_config.max_train_samples_per_fold))
    x_val, y_val, meta_val = tail_limit(x_val, y_val, meta_val, int(training_config.max_validation_samples_per_fold))
    return {
        "fold": fold,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "X_train": x_train,
        "y_train": y_train,
        "y_train_fit": make_loss_target(y_train, meta_train),
        "meta_train": meta_train,
        "X_val": x_val,
        "y_val": y_val,
        "y_val_fit": make_loss_target(y_val, meta_val),
        "meta_val": meta_val,
    }


def build_helformer_custom_objects():
    import tensorflow as tf
    from tensorflow.keras import layers

    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    @tf.keras.utils.register_keras_serializable()
    class HoltWintersDecomposition(layers.Layer):
        def build(self, input_shape):
            self.alpha_raw = self.add_weight(name="alpha_raw", shape=(), initializer=tf.keras.initializers.Constant(0.0), trainable=True)
            self.beta_raw = self.add_weight(name="beta_raw", shape=(), initializer=tf.keras.initializers.Constant(-1.0), trainable=True)
            self.gamma_raw = self.add_weight(name="gamma_raw", shape=(), initializer=tf.keras.initializers.Constant(-1.0), trainable=True)
            super().build(input_shape)

        def call(self, inputs):
            x = inputs[..., :1]
            alpha = tf.nn.sigmoid(self.alpha_raw)
            beta = tf.nn.sigmoid(self.beta_raw)
            gamma = tf.nn.sigmoid(self.gamma_raw)
            sequence = tf.transpose(x, [1, 0, 2])
            initial_level = sequence[0]
            initial_trend = tf.zeros_like(initial_level)
            initial_seasonal = tf.zeros_like(initial_level)

            def step(state, observation):
                previous_level, previous_trend, previous_seasonal = state
                adjusted_observation = observation - previous_seasonal
                level = alpha * adjusted_observation + (1.0 - alpha) * (previous_level + previous_trend)
                trend = beta * (level - previous_level) + (1.0 - beta) * previous_trend
                seasonal = gamma * (observation - level) + (1.0 - gamma) * previous_seasonal
                return level, trend, seasonal

            levels, trends, seasonals = tf.scan(step, sequence, initializer=(initial_level, initial_trend, initial_seasonal))
            levels = tf.transpose(levels, [1, 0, 2])
            trends = tf.transpose(trends, [1, 0, 2])
            seasonals = tf.transpose(seasonals, [1, 0, 2])
            residuals = x - levels - seasonals
            return tf.concat([inputs, levels, trends, seasonals, residuals], axis=-1)

    @tf.keras.utils.register_keras_serializable()
    class TrainablePositionEmbedding(layers.Layer):
        def build(self, input_shape):
            self.position = self.add_weight(
                name="position",
                shape=(1, int(input_shape[1]), int(input_shape[2])),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            return inputs + self.position

    @tf.keras.utils.register_keras_serializable()
    class LastToken(layers.Layer):
        def call(self, inputs):
            return inputs[:, -1, :]

        def compute_output_shape(self, input_shape):
            return input_shape[0], input_shape[-1]

    @tf.keras.utils.register_keras_serializable()
    class RealCloseDirectionalLoss(tf.keras.losses.Loss):
        def __init__(
            self,
            y_mean,
            y_std,
            log_weight=0.35,
            price_weight=1.0,
            direction_weight=2.5,
            direction_temperature=0.0015,
            direction_min_abs_move=0.0008,
            price_huber_delta=0.003,
            log_huber_delta=1.0,
            name="real_close_directional_loss",
        ):
            super().__init__(name=name)
            self.y_mean = float(y_mean)
            self.y_std = float(y_std)
            self.log_weight = float(log_weight)
            self.price_weight = float(price_weight)
            self.direction_weight = float(direction_weight)
            self.direction_temperature = float(direction_temperature)
            self.direction_min_abs_move = float(direction_min_abs_move)
            self.price_huber_delta = float(price_huber_delta)
            self.log_huber_delta = float(log_huber_delta)

        @staticmethod
        def _huber(error, delta):
            abs_error = tf.abs(error)
            quadratic = 0.5 * tf.square(error)
            linear = delta * (abs_error - 0.5 * delta)
            return tf.where(abs_error <= delta, quadratic, linear)

        def call(self, y_true, y_pred_scaled):
            y_mean = tf.constant(self.y_mean, dtype=tf.float32)
            y_std = tf.constant(self.y_std, dtype=tf.float32)
            y_true_scaled = y_true[:, 0:1]
            current_close = y_true[:, 1:2]
            actual_close = y_true[:, 2:3]
            actual_log_move = y_true[:, 3:4]
            pred_log_move = y_pred_scaled * y_std + y_mean
            pred_log_move = tf.clip_by_value(pred_log_move, -0.08, 0.08)
            pred_close = current_close * tf.exp(pred_log_move)
            relative_price_error = (pred_close - actual_close) / tf.maximum(current_close, 1e-12)
            price_loss = self._huber(relative_price_error, self.price_huber_delta)
            log_error_scaled = y_pred_scaled - y_true_scaled
            log_loss = self._huber(log_error_scaled, self.log_huber_delta)
            direction_label = tf.cast(actual_log_move > 0.0, tf.float32)
            direction_prob = tf.sigmoid(pred_log_move / self.direction_temperature)
            direction_prob = tf.clip_by_value(direction_prob, 1e-6, 1.0 - 1e-6)
            bce = -(direction_label * tf.math.log(direction_prob) + (1.0 - direction_label) * tf.math.log(1.0 - direction_prob))
            tradable_mask = tf.cast(tf.abs(actual_log_move) >= self.direction_min_abs_move, tf.float32)
            move_weight = tf.stop_gradient(tf.clip_by_value(tf.abs(actual_log_move) / self.direction_min_abs_move, 1.0, 4.0))
            direction_loss = bce * tradable_mask * move_weight
            total_loss = self.price_weight * price_loss + self.log_weight * log_loss + self.direction_weight * direction_loss
            return tf.reduce_mean(total_loss)

        def get_config(self):
            config = super().get_config()
            config.update({
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "log_weight": self.log_weight,
                "price_weight": self.price_weight,
                "direction_weight": self.direction_weight,
                "direction_temperature": self.direction_temperature,
                "direction_min_abs_move": self.direction_min_abs_move,
                "price_huber_delta": self.price_huber_delta,
                "log_huber_delta": self.log_huber_delta,
            })
            return config

    @tf.keras.utils.register_keras_serializable()
    class RealCloseMaeBps(tf.keras.metrics.Metric):
        def __init__(self, y_mean, y_std, name="real_close_mae_bps", **kwargs):
            super().__init__(name=name, **kwargs)
            self.y_mean = float(y_mean)
            self.y_std = float(y_std)
            self.total = self.add_weight(name="total", initializer="zeros")
            self.count = self.add_weight(name="count", initializer="zeros")

        def update_state(self, y_true, y_pred_scaled, sample_weight=None):
            y_mean = tf.constant(self.y_mean, dtype=tf.float32)
            y_std = tf.constant(self.y_std, dtype=tf.float32)
            current_close = y_true[:, 1:2]
            actual_close = y_true[:, 2:3]
            pred_log_move = y_pred_scaled * y_std + y_mean
            pred_log_move = tf.clip_by_value(pred_log_move, -0.08, 0.08)
            pred_close = current_close * tf.exp(pred_log_move)
            error_bps = tf.abs(pred_close - actual_close) / tf.maximum(current_close, 1e-12) * 10000.0
            self.total.assign_add(tf.reduce_sum(error_bps))
            self.count.assign_add(tf.cast(tf.size(error_bps), tf.float32))

        def result(self):
            return self.total / tf.maximum(self.count, 1.0)

        def reset_state(self):
            self.total.assign(0.0)
            self.count.assign(0.0)

        def get_config(self):
            config = super().get_config()
            config.update({"y_mean": self.y_mean, "y_std": self.y_std})
            return config

    @tf.keras.utils.register_keras_serializable()
    class DirectionalAccuracyClean(tf.keras.metrics.Metric):
        def __init__(self, y_mean, y_std, min_abs_move=0.0008, name="directional_accuracy_clean", **kwargs):
            super().__init__(name=name, **kwargs)
            self.y_mean = float(y_mean)
            self.y_std = float(y_std)
            self.min_abs_move = float(min_abs_move)
            self.correct = self.add_weight(name="correct", initializer="zeros")
            self.count = self.add_weight(name="count", initializer="zeros")

        def update_state(self, y_true, y_pred_scaled, sample_weight=None):
            y_mean = tf.constant(self.y_mean, dtype=tf.float32)
            y_std = tf.constant(self.y_std, dtype=tf.float32)
            actual_log_move = y_true[:, 3:4]
            pred_log_move = y_pred_scaled * y_std + y_mean
            mask = tf.abs(actual_log_move) >= self.min_abs_move
            correct = tf.cast(tf.equal(actual_log_move > 0.0, pred_log_move > 0.0), tf.float32)
            mask_float = tf.cast(mask, tf.float32)
            self.correct.assign_add(tf.reduce_sum(correct * mask_float))
            self.count.assign_add(tf.reduce_sum(mask_float))

        def result(self):
            return self.correct / tf.maximum(self.count, 1.0)

        def reset_state(self):
            self.correct.assign(0.0)
            self.count.assign(0.0)

        def get_config(self):
            config = super().get_config()
            config.update({"y_mean": self.y_mean, "y_std": self.y_std, "min_abs_move": self.min_abs_move})
            return config

    return {
        "mish": mish,
        "HoltWintersDecomposition": HoltWintersDecomposition,
        "TrainablePositionEmbedding": TrainablePositionEmbedding,
        "LastToken": LastToken,
        "RealCloseDirectionalLoss": RealCloseDirectionalLoss,
        "RealCloseMaeBps": RealCloseMaeBps,
        "DirectionalAccuracyClean": DirectionalAccuracyClean,
    }


def helformer_encoder_block(x, model_dim: int, num_heads: int, lstm_units: int, dropout_rate: float):
    from tensorflow.keras import layers

    attention_input = layers.LayerNormalization(epsilon=1e-6)(x)
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=model_dim // num_heads, dropout=dropout_rate)(attention_input, attention_input)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attention_output]))
    lstm_input = x
    lstm_output = layers.LSTM(lstm_units, return_sequences=True)(lstm_input)
    lstm_output = layers.Dense(model_dim)(lstm_output)
    lstm_output = layers.Dropout(dropout_rate)(lstm_output)
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([lstm_input, lstm_output]))


def build_helformer_model(input_shape: tuple[int, int], params: dict[str, Any], y_mean: float, y_std: float, training_config: HelformerTrainingConfig):
    import tensorflow as tf
    from tensorflow.keras import Model, layers

    objects = build_helformer_custom_objects()
    inputs = layers.Input(shape=input_shape)
    x = objects["HoltWintersDecomposition"]()(inputs)
    x = layers.Dense(int(params["model_dim"]))(x)
    x = objects["TrainablePositionEmbedding"]()(x)
    x = layers.Dropout(float(params["dropout_rate"]))(x)
    for _ in range(int(params["num_blocks"])):
        x = helformer_encoder_block(x, int(params["model_dim"]), int(params["num_heads"]), int(params["lstm_units"]), float(params["dropout_rate"]))
    last_token = objects["LastToken"]()(x)
    pooled = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([last_token, pooled])
    x = layers.Dense(int(params["dense_units"]), activation=objects["mish"])(x)
    x = layers.Dropout(float(params["dropout_rate"]))(x)
    outputs = layers.Dense(1, dtype="float32")(x)
    model = Model(inputs, outputs, name="LeakageFree_PurgedWF_Helformer")
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        clipnorm=1.0,
    )
    loss_fn = objects["RealCloseDirectionalLoss"](
        y_mean=y_mean,
        y_std=y_std,
        log_weight=training_config.loss_log_weight,
        price_weight=training_config.loss_price_weight,
        direction_weight=training_config.loss_direction_weight,
        direction_temperature=training_config.loss_direction_temperature,
        direction_min_abs_move=training_config.loss_direction_min_abs_move,
        price_huber_delta=training_config.loss_price_huber_delta,
        log_huber_delta=training_config.loss_log_huber_delta,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            objects["RealCloseMaeBps"](y_mean, y_std),
            objects["DirectionalAccuracyClean"](y_mean, y_std, min_abs_move=training_config.loss_direction_min_abs_move),
        ],
    )
    return model


def suggest_params(trial) -> dict[str, Any]:
    model_dim = trial.suggest_categorical("model_dim", [32, 48, 64])
    valid_heads = [head for head in [2, 4] if model_dim % head == 0]
    return {
        "model_dim": model_dim,
        "num_heads": trial.suggest_categorical("num_heads", valid_heads),
        "num_blocks": trial.suggest_int("num_blocks", 1, 2),
        "lstm_units": trial.suggest_categorical("lstm_units", [32, 48, 64]),
        "dense_units": trial.suggest_categorical("dense_units", [32, 48, 64]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.05, 0.20),
        "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 192]),
    }


def model_params_only(params: dict[str, Any]) -> dict[str, Any]:
    result = dict(params)
    result.pop("batch_size", None)
    return result


def fit_fold_model(params: dict[str, Any], dataset: dict[str, Any], training_config: HelformerTrainingConfig, runtime: RuntimeBudget, epochs: int, trial=None, verbose: int = 0, reserve_minutes: float | None = None, step_offset: int = 0):
    import optuna
    import tensorflow as tf

    class OptunaPruningCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            value = logs.get("val_loss") if logs else None
            if value is None:
                return
            trial.report(float(value), step=step_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    class StopWhenTimeLow(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if reserve_minutes is not None and not runtime.has_reserve(reserve_minutes):
                self.model.stop_training = True

    tf.keras.backend.clear_session()
    model = build_helformer_model(
        (int(training_config.window_size), int(dataset["X_train"].shape[-1])),
        model_params_only(params),
        y_mean=float(dataset["y_scaler"].mean_[0]),
        y_std=float(dataset["y_scaler"].scale_[0]),
        training_config=training_config,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=training_config.patience, restore_best_weights=True, min_delta=training_config.min_delta),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, training_config.patience // 2), min_lr=1e-5),
    ]
    if trial is not None:
        callbacks.append(OptunaPruningCallback())
    if reserve_minutes is not None:
        callbacks.append(StopWhenTimeLow())
    history = model.fit(
        dataset["X_train"],
        dataset["y_train_fit"],
        validation_data=(dataset["X_val"], dataset["y_val_fit"]),
        epochs=int(epochs),
        batch_size=int(params["batch_size"]),
        verbose=verbose,
        callbacks=callbacks,
    )
    best_loss = float(np.min(history.history["val_loss"]))
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    return model, history, best_loss, best_epoch


def tune_helformer(fold_datasets: list[dict[str, Any]], training_config: HelformerTrainingConfig, runtime: RuntimeBudget, artifact_dir: Path) -> tuple[dict[str, Any], list[float], list[int], float, pd.DataFrame]:
    try:
        import optuna
    except ImportError:
        return dict(training_config.fallback_params), [], [], float(training_config.min_final_epochs), pd.DataFrame()

    def objective(trial):
        if not runtime.has_reserve(training_config.reserved_final_minutes):
            raise optuna.TrialPruned()
        params = suggest_params(trial)
        fold_losses = []
        fold_epochs = []
        for fold_index, dataset in enumerate(fold_datasets):
            if not runtime.has_reserve(training_config.reserved_final_minutes):
                raise optuna.TrialPruned()
            model, history, loss, epoch = fit_fold_model(
                params,
                dataset,
                training_config,
                runtime,
                int(training_config.tuning_epochs),
                trial=trial,
                verbose=0,
                reserve_minutes=training_config.reserved_final_minutes,
                step_offset=fold_index * int(training_config.tuning_epochs),
            )
            fold_losses.append(loss)
            fold_epochs.append(epoch)
            del model, history
            gc.collect()
        trial.set_user_attr("fold_losses", fold_losses)
        trial.set_user_attr("fold_best_epochs", fold_epochs)
        trial.set_user_attr("mean_best_epoch", float(np.mean(fold_epochs)))
        return float(np.mean(fold_losses))

    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    search_timeout_seconds = max(60, int((runtime.remaining_minutes() - training_config.reserved_final_minutes) * 60))
    study.optimize(objective, n_trials=int(training_config.n_trials), timeout=search_timeout_seconds, gc_after_trial=True, show_progress_bar=False)
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        params = dict(study.best_trial.params)
        fold_losses = list(study.best_trial.user_attrs.get("fold_losses", []))
        fold_epochs = list(study.best_trial.user_attrs.get("fold_best_epochs", []))
        mean_best_epoch = float(study.best_trial.user_attrs.get("mean_best_epoch", training_config.min_final_epochs))
    else:
        params = dict(training_config.fallback_params)
        fold_losses = []
        fold_epochs = []
        mean_best_epoch = float(training_config.min_final_epochs)
    trials_frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    if not trials_frame.empty:
        trials_frame.to_csv(artifact_dir / "optuna_purged_walk_forward_trials.csv", index=False)
    return params, fold_losses, fold_epochs, mean_best_epoch, trials_frame


def inverse_target(values: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def candle_by_candle_predict(model, x: np.ndarray, meta: pd.DataFrame, y_scaler: StandardScaler, calibrator: RegimeAwareCalibrator | None = None, batch_size: int = 1024) -> pd.DataFrame:
    raw_scaled = model.predict(x, batch_size=batch_size, verbose=0).ravel()
    raw_log_move = inverse_target(raw_scaled, y_scaler)
    records = []
    for i, row in meta.reset_index(drop=True).iterrows():
        volatility = max(float(row["regime_vol"]), 1e-8)
        if calibrator is None:
            predicted_log_move = float(raw_log_move[i])
            regime = "none"
            confidence = 1.0
        else:
            predicted_log_move, regime, confidence = calibrator.predict(float(raw_log_move[i]), volatility)
        predicted_close = float(row["current_close"]) * math.exp(predicted_log_move)
        raw_predicted_close = float(row["current_close"]) * math.exp(float(raw_log_move[i]))
        actual_log_move = float(row["normalized_next_close"])
        records.append({
            "decision_time": row["decision_time"],
            "target_time": row["target_time"],
            "current_close": float(row["current_close"]),
            "actual_close": float(row["actual_close"]),
            "raw_predicted_log_move": float(raw_log_move[i]),
            "predicted_log_move": predicted_log_move,
            "raw_predicted_close": raw_predicted_close,
            "predicted_close": predicted_close,
            "normalized_next_close": actual_log_move,
            "regime_vol": volatility,
            "regime": regime,
            "calibration_confidence": float(confidence),
        })
        if calibrator is not None:
            calibrator.update(actual_log_move, predicted_log_move, regime)
    return pd.DataFrame(records)


def forecast_metrics(frame: pd.DataFrame, price_column: str, log_column: str) -> dict[str, float]:
    actual_price = frame["actual_close"].to_numpy(dtype=float)
    predicted_price = frame[price_column].to_numpy(dtype=float)
    actual_log = frame["normalized_next_close"].to_numpy(dtype=float)
    predicted_log = frame[log_column].to_numpy(dtype=float)
    return {
        "price_rmse": float(math.sqrt(mean_squared_error(actual_price, predicted_price))),
        "price_mae": float(mean_absolute_error(actual_price, predicted_price)),
        "price_mape_percent": mean_absolute_percentage_error(actual_price, predicted_price),
        "price_explained_variance": float(explained_variance_score(actual_price, predicted_price)),
        "log_rmse": float(math.sqrt(mean_squared_error(actual_log, predicted_log))),
        "log_mae": float(mean_absolute_error(actual_log, predicted_log)),
        "log_bias": float(np.mean(predicted_log - actual_log)),
        "log_r2": float(r2_score(actual_log, predicted_log)),
        "directional_accuracy": directional_accuracy(actual_log, predicted_log),
        "median_abs_price_error": float(np.median(np.abs(actual_price - predicted_price))),
        "p95_abs_price_error": float(np.quantile(np.abs(actual_price - predicted_price), 0.95)),
    }


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-12))) * 100.0)


def directional_accuracy(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true_log) == np.sign(y_pred_log)))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(drawdown))


def annualized_sharpe(log_returns: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    log_returns = np.asarray(log_returns, dtype=float)
    std = np.std(log_returns)
    if std <= 1e-12:
        return 0.0
    return float(np.mean(log_returns) / std * np.sqrt(periods_per_year))


def annualized_sortino(log_returns: np.ndarray, periods_per_year: int = 24 * 365) -> float:
    log_returns = np.asarray(log_returns, dtype=float)
    downside = log_returns[log_returns < 0.0]
    std = np.std(downside) if len(downside) else 0.0
    if std <= 1e-12:
        return 0.0
    return float(np.mean(log_returns) / std * np.sqrt(periods_per_year))


def run_forecast_strategy(frame: pd.DataFrame, fee_rate: float = 0.0004, allow_short: bool = False) -> pd.DataFrame:
    data = frame.copy().reset_index(drop=True)
    edge = data["predicted_log_move"].to_numpy(dtype=float)
    realized = data["normalized_next_close"].to_numpy(dtype=float)
    volatility = np.maximum(data["regime_vol"].to_numpy(dtype=float), 1e-6)
    threshold = np.maximum(2.0 * fee_rate, 0.12 * volatility)
    raw_position = np.tanh(edge / (1.5 * volatility))
    if not allow_short:
        raw_position = np.clip(raw_position, 0.0, 1.0)
    raw_position[np.abs(edge) < threshold] = 0.0
    turnover = np.abs(np.diff(np.r_[0.0, raw_position]))
    strategy_log_return = raw_position * realized - fee_rate * turnover
    buy_hold_log_return = realized
    data["position"] = raw_position
    data["turnover"] = turnover
    data["strategy_log_return"] = strategy_log_return
    data["buy_hold_log_return"] = buy_hold_log_return
    data["strategy_equity"] = np.exp(np.cumsum(strategy_log_return))
    data["buy_hold_equity"] = np.exp(np.cumsum(buy_hold_log_return))
    return data


def strategy_metrics(data: pd.DataFrame) -> pd.DataFrame:
    strategy_returns = data["strategy_log_return"].values
    buy_hold_returns = data["buy_hold_log_return"].values
    return pd.DataFrame({
        "strategy": {
            "total_return_percent": 100.0 * (data["strategy_equity"].iloc[-1] - 1.0),
            "annualized_sharpe": annualized_sharpe(strategy_returns),
            "annualized_sortino": annualized_sortino(strategy_returns),
            "max_drawdown_percent": 100.0 * max_drawdown(data["strategy_equity"].values),
            "mean_position": float(np.mean(data["position"])),
            "mean_turnover": float(np.mean(data["turnover"])),
            "trading_hours": int(np.sum(data["position"] > 0.0)),
        },
        "buy_hold": {
            "total_return_percent": 100.0 * (data["buy_hold_equity"].iloc[-1] - 1.0),
            "annualized_sharpe": annualized_sharpe(buy_hold_returns),
            "annualized_sortino": annualized_sortino(buy_hold_returns),
            "max_drawdown_percent": 100.0 * max_drawdown(data["buy_hold_equity"].values),
            "mean_position": 1.0,
            "mean_turnover": 0.0,
            "trading_hours": len(data),
        },
    }).T


def warm_calibrator_with_oof_folds(calibrator: RegimeAwareCalibrator, params: dict[str, Any], fold_datasets: list[dict[str, Any]], training_config: HelformerTrainingConfig, runtime: RuntimeBudget) -> pd.DataFrame:
    warmup_frames = []
    for dataset in fold_datasets:
        if not runtime.has_reserve(20):
            break
        model, history, loss, epoch = fit_fold_model(params, dataset, training_config, runtime, int(training_config.max_calibrator_epochs), verbose=0, reserve_minutes=20)
        frame = candle_by_candle_predict(model, dataset["X_val"], dataset["meta_val"], dataset["y_scaler"], calibrator, training_config.prediction_batch_size)
        frame["fold"] = dataset["fold"].name
        warmup_frames.append(frame)
        del model, history
        gc.collect()
    if warmup_frames:
        return pd.concat(warmup_frames, ignore_index=True)
    return pd.DataFrame()


def train_helformer_next_close_model(data: pd.DataFrame, config: AppConfig, training_config: HelformerTrainingConfig | None = None) -> ModelArtifact:
    training_config = training_config or HelformerTrainingConfig(fee_rate=float(config.execution.fee_rate))
    seed_runtime(SEED)
    runtime = RuntimeBudget(training_config.max_runtime_minutes)
    artifact_dir = config.ml.model_artifact_dir / f"helformer_next_close_{timestamp_for_run_id()}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    features, feature_columns = prepare_supervised_features(data)
    development_start = utc_timestamp(training_config.development_start)
    test_start = utc_timestamp(training_config.test_start)
    test_end = utc_timestamp(training_config.test_end)
    folds = make_purged_walk_forward_folds(training_config)
    fold_datasets = [build_fold_dataset(features, feature_columns, fold, training_config) for fold in folds]
    development_mask = target_mask(features, development_start, test_start)
    final_x_scaler, final_y_scaler = fit_scalers(features, feature_columns, development_mask)
    x_dev, y_dev, meta_dev = build_sequences(features, feature_columns, final_x_scaler, final_y_scaler, development_start, test_start, training_config)
    x_test, y_test, meta_test = build_sequences(features, feature_columns, final_x_scaler, final_y_scaler, test_start, test_end, training_config)
    y_dev_fit = make_loss_target(y_dev, meta_dev)
    best_params, best_fold_losses, best_fold_epochs, mean_best_epoch, trials_frame = tune_helformer(fold_datasets, training_config, runtime, artifact_dir)
    recommended_final_epochs = int(np.clip(math.ceil(mean_best_epoch + training_config.extra_final_epochs), training_config.min_final_epochs, training_config.final_epochs_cap))
    import tensorflow as tf

    tf.keras.backend.clear_session()
    final_model = build_helformer_model(
        (int(training_config.window_size), len(feature_columns)),
        model_params_only(best_params),
        y_mean=float(final_y_scaler.mean_[0]),
        y_std=float(final_y_scaler.scale_[0]),
        training_config=training_config,
    )

    class StopWhenTimeLow(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if not runtime.has_reserve(10):
                self.model.stop_training = True

    final_history = final_model.fit(
        x_dev,
        y_dev_fit,
        epochs=recommended_final_epochs,
        batch_size=int(best_params.get("batch_size", 128)),
        verbose=1,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            StopWhenTimeLow(),
            tf.keras.callbacks.ModelCheckpoint(str(artifact_dir / "model_best.keras"), monitor="loss", save_best_only=True),
        ],
    )
    history_frame = pd.DataFrame(final_history.history)
    history_frame.to_csv(artifact_dir / "final_training_history.csv", index=False)
    final_model.save(artifact_dir / "model.keras")
    volatility_reference = features.loc[development_mask, "realized_vol_24h"].dropna()
    low_vol_threshold = float(volatility_reference.quantile(0.35))
    high_vol_threshold = float(volatility_reference.quantile(0.75))
    calibrator = RegimeAwareCalibrator(low_vol_threshold=low_vol_threshold, high_vol_threshold=high_vol_threshold)
    warmup_predictions = warm_calibrator_with_oof_folds(calibrator, best_params, fold_datasets, training_config, runtime) if training_config.calibrator_warmup_folds else pd.DataFrame()
    if not warmup_predictions.empty:
        warmup_predictions.to_csv(artifact_dir / "calibrator_oof_warmup_predictions.csv", index=False)
    initial_calibrator_state = calibrator.state_dict()
    test_predictions = candle_by_candle_predict(final_model, x_test, meta_test, final_y_scaler, calibrator, training_config.prediction_batch_size)
    test_predictions.to_csv(artifact_dir / "test_predictions_2025_candle_by_candle.csv", index=False)
    metrics = pd.DataFrame({
        "raw_model": forecast_metrics(test_predictions, "raw_predicted_close", "raw_predicted_log_move"),
        "regime_calibrated": forecast_metrics(test_predictions, "predicted_close", "predicted_log_move"),
    }).T
    metrics.to_csv(artifact_dir / "forecast_metrics_2025.csv")
    strategy_frame = run_forecast_strategy(test_predictions, fee_rate=training_config.fee_rate, allow_short=training_config.allow_short)
    strategy_stats = strategy_metrics(strategy_frame)
    strategy_frame.to_csv(artifact_dir / "strategy_2025.csv", index=False)
    strategy_stats.to_csv(artifact_dir / "strategy_metrics_2025.csv")
    split_info = {
        "policy": "purged_walk_forward_train_2020_2024_test_2025",
        "train_start": str(development_start),
        "train_end": str(test_start),
        "test_start": str(test_start),
        "test_end": str(test_end),
        "development_sequences": int(len(x_dev)),
        "test_sequences": int(len(x_test)),
        "folds": [asdict(fold) for fold in folds],
        "best_fold_losses": best_fold_losses,
        "best_fold_epochs": best_fold_epochs,
        "recommended_final_epochs": recommended_final_epochs,
    }
    metadata = {
        "symbol": config.data.symbol,
        "timeframe": config.data.timeframe,
        "model_type": "LeakageFree_PurgedWF_Helformer",
        "strategy": "helformer_momentum",
        "model_file": "model.keras",
        "sequence_length": int(training_config.window_size),
        "feature_columns": feature_columns,
        "target": "normalized_next_close",
        "target_definition": "log(close[t+1] / close[t])",
        "prediction_definition": "predicted_close[t+1] = close[t] * exp(predicted_log_move)",
        "allow_short": bool(training_config.allow_short),
        "fee_rate": float(training_config.fee_rate),
        "max_abs_log_move": 0.08,
        "calibrator_initial_state": initial_calibrator_state,
        "calibrator_post_test_state": calibrator.state_dict(),
        "best_params": best_params,
        "feature_mean": final_x_scaler.mean_.astype(float).tolist(),
        "feature_std": final_x_scaler.scale_.astype(float).tolist(),
        "target_mean": float(final_y_scaler.mean_[0]),
        "target_std": float(final_y_scaler.scale_[0]),
        "no_data_leakage_controls": [
            "causal_features_only",
            "target_is_next_close_log_move",
            "scaler_fit_on_train_only",
            "purged_walk_forward_validation",
            "train_2020_2024_validation_before_2025",
            "2025_reserved_for_platform_backtest_and_report",
        ],
    }
    training_payload = asdict(training_config)
    training_payload.update({"symbol": config.data.symbol, "interval": config.data.timeframe, "artifact_dir": str(artifact_dir)})
    write_json(artifact_dir / "metadata.json", metadata)
    write_json(artifact_dir / "metrics.json", {"forecast_2025": metrics.to_dict(), "strategy_2025": strategy_stats.to_dict()})
    write_json(artifact_dir / "split_info.json", split_info)
    write_json(artifact_dir / "feature_config.json", training_payload)
    write_json(artifact_dir / "feature_columns.json", {"feature_columns": feature_columns})
    write_json(artifact_dir / "config.json", training_payload)
    write_json(artifact_dir / "best_params.json", best_params)
    with (artifact_dir / "scalers.pkl").open("wb") as file:
        pickle.dump({"x_scaler": final_x_scaler, "y_scaler": final_y_scaler}, file)
    summary = {
        "elapsed_minutes": runtime.elapsed_minutes(),
        "recommended_final_epochs": recommended_final_epochs,
        "best_fold_losses": best_fold_losses,
        "best_fold_epochs": best_fold_epochs,
        "forecast_metrics": metrics.to_dict(),
        "strategy_metrics": strategy_stats.to_dict(),
        "calibrator_initial_state": initial_calibrator_state,
        "calibrator_post_test_state": calibrator.state_dict(),
        "artifacts": sorted([str(path) for path in artifact_dir.glob("*")]),
    }
    write_json(artifact_dir / "run_summary.json", summary)
    return ModelArtifact(
        artifact_id=artifact_dir.name,
        artifact_dir=artifact_dir,
        model_path=artifact_dir / "model.keras",
        metadata_path=artifact_dir / "metadata.json",
        metrics_path=artifact_dir / "metrics.json",
        split_info_path=artifact_dir / "split_info.json",
        feature_config_path=artifact_dir / "feature_config.json",
    )


def build_helformer_feature_matrix_from_arrays(
    ts: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    extra: dict[str, np.ndarray],
    feature_columns: list[str],
) -> np.ndarray:
    eps = 1e-12
    close = np.asarray(close, dtype=np.float64)
    open_ = np.asarray(open_, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    quote_volume = extra_array(extra, ["quote_asset_volume", "quote_volume"], volume * close)
    trades = extra_array(extra, ["number_of_trades", "trade_count"], np.zeros_like(close))
    taker_buy_base = extra_array(extra, ["taker_buy_base_asset_volume", "taker_buy_volume"], np.zeros_like(close))
    taker_buy_quote = extra_array(extra, ["taker_buy_quote_asset_volume", "taker_buy_quote_volume"], 0.5 * quote_volume)
    is_imputed = extra_array(extra, ["is_imputed"], np.zeros_like(close))
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(ts, dtype=np.int64), utc=True))
    close_series = pd.Series(close)
    log_close = np.log(np.maximum(close, eps))
    normalized_now = pd.Series(log_close).diff().fillna(0.0).to_numpy(dtype=np.float64)
    log_volume = np.log1p(np.maximum(volume, 0.0))
    log_quote_volume = np.log1p(np.maximum(quote_volume, 0.0))
    log_trades = np.log1p(np.maximum(trades, 0.0))
    taker_buy_ratio = pd.Series(taker_buy_quote / safe_denominator_array(quote_volume)).clip(0.0, 1.0).fillna(0.5).to_numpy(dtype=np.float64)
    feature_map: dict[str, np.ndarray] = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_asset_volume": quote_volume,
        "quote_volume": quote_volume,
        "number_of_trades": trades,
        "trade_count": trades,
        "taker_buy_base_asset_volume": taker_buy_base,
        "taker_buy_volume": taker_buy_base,
        "taker_buy_quote_asset_volume": taker_buy_quote,
        "taker_buy_quote_volume": taker_buy_quote,
        "is_imputed": is_imputed,
        "log_close": log_close,
        "normalized_now": normalized_now,
        "range_log": np.log((high + eps) / (low + eps)),
        "body_log": np.log((close + eps) / (open_ + eps)),
        "upper_wick_log": np.log((high + eps) / (np.maximum(open_, close) + eps)),
        "lower_wick_log": np.log((np.minimum(open_, close) + eps) / (low + eps)),
        "close_location": np.clip((close - low) / (high - low + eps), 0.0, 1.0),
        "log_volume": log_volume,
        "log_quote_volume": log_quote_volume,
        "log_trades": log_trades,
        "taker_buy_ratio": taker_buy_ratio,
        "taker_buy_imbalance": 2.0 * taker_buy_ratio - 1.0,
        "hour_sin": np.sin(2.0 * np.pi * timestamp_index.hour.to_numpy(dtype=float) / 24.0),
        "hour_cos": np.cos(2.0 * np.pi * timestamp_index.hour.to_numpy(dtype=float) / 24.0),
        "dayofweek_sin": np.sin(2.0 * np.pi * timestamp_index.dayofweek.to_numpy(dtype=float) / 7.0),
        "dayofweek_cos": np.cos(2.0 * np.pi * timestamp_index.dayofweek.to_numpy(dtype=float) / 7.0),
        "dayofyear_sin": np.sin(2.0 * np.pi * timestamp_index.dayofyear.to_numpy(dtype=float) / 365.25),
        "dayofyear_cos": np.cos(2.0 * np.pi * timestamp_index.dayofyear.to_numpy(dtype=float) / 365.25),
    }
    for window in [3, 6, 12, 24, 72, 168]:
        log_volume_series = pd.Series(log_volume)
        volume_mean = log_volume_series.rolling(window, min_periods=max(3, window // 4)).mean()
        volume_std = log_volume_series.rolling(window, min_periods=max(3, window // 4)).std().replace(0.0, np.nan)
        feature_map[f"momentum_{window}h"] = pd.Series(log_close).diff(window).to_numpy(dtype=np.float64)
        feature_map[f"realized_vol_{window}h"] = pd.Series(normalized_now).rolling(window, min_periods=max(3, window // 4)).std().to_numpy(dtype=np.float64)
        feature_map[f"range_mean_{window}h"] = pd.Series(feature_map["range_log"]).rolling(window, min_periods=max(3, window // 4)).mean().to_numpy(dtype=np.float64)
        feature_map[f"volume_z_{window}h"] = ((log_volume_series - volume_mean) / volume_std).fillna(0.0).to_numpy(dtype=np.float64)
    feature_map["ema_12_gap"] = log_close - np.log(ema(close_series, 12).to_numpy(dtype=np.float64))
    feature_map["ema_24_gap"] = log_close - np.log(ema(close_series, 24).to_numpy(dtype=np.float64))
    feature_map["ema_72_gap"] = log_close - np.log(ema(close_series, 72).to_numpy(dtype=np.float64))
    feature_map["ema_168_gap"] = log_close - np.log(ema(close_series, 168).to_numpy(dtype=np.float64))
    macd_fast = ema(close_series, 12)
    macd_slow = ema(close_series, 26)
    macd = np.log(macd_fast.to_numpy(dtype=np.float64)) - np.log(macd_slow.to_numpy(dtype=np.float64))
    feature_map["macd"] = macd
    feature_map["macd_signal_gap"] = macd - ema(pd.Series(macd), 9).to_numpy(dtype=np.float64)
    feature_map["rsi_14"] = make_rsi(close_series, 14).to_numpy(dtype=np.float64) / 100.0
    feature_map["rsi_48"] = make_rsi(close_series, 48).to_numpy(dtype=np.float64) / 100.0
    matrix = np.column_stack([feature_map.get(column, np.zeros_like(close, dtype=np.float64)) for column in feature_columns])
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def extra_array(extra: dict[str, np.ndarray], names: list[str], fallback: np.ndarray) -> np.ndarray:
    for name in names:
        if name in extra:
            return np.asarray(extra[name], dtype=np.float64)
    return np.asarray(fallback, dtype=np.float64)


def safe_denominator_array(values: np.ndarray) -> np.ndarray:
    safe = np.asarray(values, dtype=np.float64).copy()
    safe[~np.isfinite(safe) | (np.abs(safe) <= 1e-15)] = np.nan
    return safe


class HelformerNextCloseForecaster:
    def __init__(self, artifact_path: str | Path) -> None:
        self.artifact_path = Path(artifact_path)
        self.artifact_dir = self.artifact_path.parent if self.artifact_path.is_file() else self.artifact_path
        self.metadata = read_json(self.artifact_dir / "metadata.json")
        self.feature_columns = self._load_feature_columns()
        self.sequence_length = int(self.metadata.get("sequence_length", 24))
        self.max_abs_log_move = float(self.metadata.get("max_abs_log_move", 0.08))
        self.model_path = self._resolve_model_path()
        self.model = self._load_model()
        self.x_mean, self.x_std, self.y_mean, self.y_std = self._load_scaler_state()
        calibrator_state = self.metadata.get("calibrator_initial_state") or self.metadata.get("calibrator_state")
        self.calibrator = RegimeAwareCalibrator.from_state(calibrator_state) if isinstance(calibrator_state, dict) else None

    def predict_latest(self, frame: pd.DataFrame, use_calibrator: bool = True) -> dict[str, Any] | None:
        normalized = normalize_helformer_frame(frame)
        if len(normalized) < self.sequence_length:
            return None
        features = add_causal_features(normalized)
        matrix = np.nan_to_num(features[self.feature_columns].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if len(matrix) < self.sequence_length:
            return None
        index = len(matrix) - 1
        current_close = float(features["current_close"].iloc[index])
        regime_vol = float(features["realized_vol_24h"].fillna(features["realized_vol_24h"].median()).iloc[index])
        active_calibrator = self.calibrator if use_calibrator else None
        return self._predict_from_matrix(matrix, index, current_close, regime_vol, pd.Timestamp(features.index[index]), active_calibrator)

    def predict_market(self, market: Any, index: int, calibrator: RegimeAwareCalibrator | None | bool = True) -> dict[str, Any] | None:
        if index + 1 < self.sequence_length:
            return None
        matrix = self._market_feature_matrix(market)
        current_close = float(market.arrays.close[index])
        regime_vol = self._regime_vol_at(matrix, index)
        if calibrator is True:
            active_calibrator = self.calibrator
        elif calibrator is False:
            active_calibrator = None
        else:
            active_calibrator = calibrator
        return self._predict_from_matrix(matrix, index, current_close, regime_vol, market.timestamp(index), active_calibrator)

    def _predict_from_matrix(self, matrix: np.ndarray, index: int, current_close: float, regime_vol: float, timestamp: pd.Timestamp, calibrator: RegimeAwareCalibrator | None | bool) -> dict[str, Any]:
        left = index + 1 - self.sequence_length
        sequence = matrix[left:index + 1].astype(np.float32, copy=True)
        sequence = (sequence - self.x_mean) / self.x_std
        raw_scaled = float(np.asarray(self.model.predict(sequence[np.newaxis, ...], verbose=0, batch_size=1)).reshape(-1)[0])
        raw_log_move = float(np.clip(raw_scaled * self.y_std + self.y_mean, -self.max_abs_log_move, self.max_abs_log_move))
        active_calibrator = calibrator if isinstance(calibrator, RegimeAwareCalibrator) else None
        if active_calibrator is None:
            predicted_log_move = raw_log_move
            regime = "none"
            confidence = 1.0
        else:
            predicted_log_move, regime, confidence = active_calibrator.predict(raw_log_move, max(regime_vol, 1e-8))
        return {
            "decision_time": timestamp,
            "current_close": current_close,
            "raw_scaled_prediction": raw_scaled,
            "raw_predicted_log_move": raw_log_move,
            "predicted_log_move": float(predicted_log_move),
            "raw_predicted_close": float(current_close * math.exp(raw_log_move)),
            "predicted_close": float(current_close * math.exp(float(predicted_log_move))),
            "regime_vol": float(max(regime_vol, 1e-8)),
            "regime": regime,
            "calibration_confidence": float(confidence),
        }

    def _market_feature_matrix(self, market: Any) -> np.ndarray:
        key = "helformer:" + ",".join(self.feature_columns)
        if key in market._cache:
            return market._cache[key]
        extra = getattr(market.arrays, "extra", {}) or {}
        matrix = build_helformer_feature_matrix_from_arrays(
            ts=market.arrays.ts,
            open_=market.arrays.open,
            high=market.arrays.high,
            low=market.arrays.low,
            close=market.arrays.close,
            volume=market.arrays.volume,
            extra=extra,
            feature_columns=self.feature_columns,
        )
        market._cache[key] = matrix
        return matrix

    def _regime_vol_at(self, matrix: np.ndarray, index: int) -> float:
        try:
            column_index = self.feature_columns.index("realized_vol_24h")
            value = float(matrix[index, column_index])
            if np.isfinite(value) and value > 0.0:
                return value
        except ValueError:
            pass
        return 1e-8

    def _load_feature_columns(self) -> list[str]:
        columns = self.metadata.get("feature_columns")
        if isinstance(columns, list) and columns:
            return [str(column) for column in columns]
        payload = read_json(self.artifact_dir / "feature_columns.json")
        if isinstance(payload, dict):
            return [str(column) for column in payload.get("feature_columns", [])]
        if isinstance(payload, list):
            return [str(column) for column in payload]
        raise ValueError(f"Helformer feature columns not found in {self.artifact_dir}.")

    def _resolve_model_path(self) -> Path:
        if self.artifact_path.is_file() and self.artifact_path.suffix.lower() == ".keras":
            return self.artifact_path
        candidates = [
            self.artifact_dir / str(self.metadata.get("model_file", "")),
            self.artifact_dir / "model.keras",
            self.artifact_dir / "helformer_final.keras",
        ]
        for candidate in candidates:
            if candidate.name and candidate.exists():
                return candidate
        raise FileNotFoundError(f"Helformer model file not found in {self.artifact_dir}.")

    def _load_model(self):
        import tensorflow as tf

        custom_objects = build_helformer_custom_objects()
        try:
            return tf.keras.models.load_model(str(self.model_path), custom_objects=custom_objects, compile=False)
        except ValueError as exc:
            message = str(exc)
            if "Lambda" not in message and "safe_mode" not in message and "unsafe_deserialization" not in message:
                raise
            try:
                return tf.keras.models.load_model(str(self.model_path), custom_objects=custom_objects, compile=False, safe_mode=False)
            except TypeError:
                if hasattr(tf.keras, "config") and hasattr(tf.keras.config, "enable_unsafe_deserialization"):
                    tf.keras.config.enable_unsafe_deserialization()
                return tf.keras.models.load_model(str(self.model_path), custom_objects=custom_objects, compile=False)

    def _load_scaler_state(self) -> tuple[np.ndarray, np.ndarray, float, float]:
        scaler_path = self.artifact_dir / "scalers.pkl"
        if scaler_path.exists():
            with scaler_path.open("rb") as file:
                scalers = pickle.load(file)
            x_scaler = scalers["x_scaler"]
            y_scaler = scalers["y_scaler"]
            x_mean = np.asarray(x_scaler.mean_, dtype=np.float32)
            x_std = np.asarray(x_scaler.scale_, dtype=np.float32)
            y_mean = float(y_scaler.mean_[0])
            y_std = float(y_scaler.scale_[0])
        else:
            x_mean = np.asarray(self.metadata["feature_mean"], dtype=np.float32)
            x_std = np.asarray(self.metadata["feature_std"], dtype=np.float32)
            y_mean = float(self.metadata["target_mean"])
            y_std = float(self.metadata["target_std"])
        x_std = np.where(np.abs(x_std) <= 1e-12, 1.0, x_std).astype(np.float32)
        if len(x_mean) != len(self.feature_columns):
            raise ValueError("Helformer scaler size does not match feature_columns.")
        return x_mean, x_std, y_mean, y_std


def read_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=json_default)


def json_default(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return str(value.item())
    return str(value)
