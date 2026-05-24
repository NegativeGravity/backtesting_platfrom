from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from trading_system.strategy.base import BaseStrategy, PortfolioView
from trading_system.strategy.signals import Signal, SignalType


@dataclass(frozen=True)
class RegimeMetaLabelConfig:
    lookback: int = 96
    volatility_window: int = 32
    trend_window: int = 64
    min_confidence: float = 0.58
    exit_confidence: float = 0.50
    max_regime_entropy: float = 0.68


class MLRegimeMetaLabelStrategy(BaseStrategy):
    """Meta-labeling strategy for single-asset directional trading.

    Expected artifact format, stored with joblib:
        {
            "model": classifier with predict_proba,
            "scaler": optional transformer,
            "feature_columns": list[str],
            "long_threshold": optional float,
            "short_threshold": optional float,
            "exit_threshold": optional float,
        }

    The strategy builds causal OHLCV features from data available up to the current
    closed candle. It does not peek into future bars. The backtest engine should
    execute the generated order on the next candle open.
    """

    def __init__(
        self,
        symbol: str,
        model_artifact_path: str | Path,
        lookback: int = 96,
        volatility_window: int = 32,
        trend_window: int = 64,
        min_confidence: float = 0.58,
        exit_confidence: float = 0.50,
        max_regime_entropy: float = 0.68,
    ) -> None:
        self._symbol = symbol
        self._config = RegimeMetaLabelConfig(
            lookback=lookback,
            volatility_window=volatility_window,
            trend_window=trend_window,
            min_confidence=min_confidence,
            exit_confidence=exit_confidence,
            max_regime_entropy=max_regime_entropy,
        )
        artifact = joblib.load(Path(model_artifact_path))
        if not isinstance(artifact, dict):
            raise ValueError("ML regime artifact must be a joblib dictionary.")
        if "model" not in artifact or "feature_columns" not in artifact:
            raise ValueError("ML regime artifact must contain 'model' and 'feature_columns'.")

        self._model = artifact["model"]
        self._scaler = artifact.get("scaler")
        self._feature_columns = list(artifact["feature_columns"])
        self._long_threshold = float(artifact.get("long_threshold", min_confidence))
        self._short_threshold = float(artifact.get("short_threshold", min_confidence))
        self._exit_threshold = float(artifact.get("exit_threshold", exit_confidence))

    def generate_signal(self, market_window: pd.DataFrame, portfolio: PortfolioView) -> Signal:
        timestamp = pd.Timestamp(market_window["timestamp"].iloc[-1])
        features = self._build_latest_features(market_window)
        metadata: dict[str, Any] = {"strategy": "ml_regime_meta_label"}

        if features is None:
            return self._hold(timestamp, "ml_regime_not_enough_history", metadata)

        missing = [column for column in self._feature_columns if column not in features.columns]
        if missing:
            metadata["missing_columns"] = missing
            return self._hold(timestamp, "ml_regime_missing_features", metadata)

        row = features[self._feature_columns]
        if self._scaler is not None:
            row = self._scaler.transform(row)

        probabilities = self._predict_probabilities(row)
        down_probability = float(probabilities.get("down", 0.0))
        flat_probability = float(probabilities.get("flat", 0.0))
        up_probability = float(probabilities.get("up", 0.0))
        confidence = max(up_probability, down_probability)
        entropy = self._normalized_entropy([down_probability, flat_probability, up_probability])

        metadata.update(
            {
                "p_down": down_probability,
                "p_flat": flat_probability,
                "p_up": up_probability,
                "confidence": confidence,
                "regime_entropy": entropy,
            }
        )

        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))
        position_side = str(getattr(portfolio, "position_side", ""))

        if position_quantity == 0:
            if entropy > self._config.max_regime_entropy:
                return self._hold(timestamp, "ml_regime_uncertain_entropy_gate", metadata)
            if up_probability >= self._long_threshold and up_probability > down_probability:
                return Signal(timestamp, self._symbol, SignalType.LONG, up_probability, "ml_regime_long_meta_label", metadata)
            if down_probability >= self._short_threshold and down_probability > up_probability:
                return Signal(timestamp, self._symbol, SignalType.SHORT, down_probability, "ml_regime_short_meta_label", metadata)
            return self._hold(timestamp, "ml_regime_no_edge", metadata)

        if position_side == "LONG" and up_probability < self._exit_threshold:
            return Signal(timestamp, self._symbol, SignalType.EXIT, 1.0 - up_probability, "ml_regime_exit_long_edge_decay", metadata)
        if position_side == "SHORT" and down_probability < self._exit_threshold:
            return Signal(timestamp, self._symbol, SignalType.EXIT, 1.0 - down_probability, "ml_regime_exit_short_edge_decay", metadata)

        return self._hold(timestamp, "ml_regime_hold_position", metadata)

    def _build_latest_features(self, market_window: pd.DataFrame) -> pd.DataFrame | None:
        required = max(self._config.lookback, self._config.trend_window, self._config.volatility_window) + 5
        if len(market_window) < required:
            return None

        df = market_window.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        returns = close.pct_change()
        log_returns = np.log(close).diff()

        trend_window = self._config.trend_window
        vol_window = self._config.volatility_window

        features = pd.DataFrame(index=df.index)
        for horizon in (1, 3, 6, 12, 24, 48):
            features[f"ret_{horizon}"] = close.pct_change(horizon)
            features[f"logret_{horizon}"] = np.log(close).diff(horizon)

        ema_fast = close.ewm(span=16, adjust=False).mean()
        ema_slow = close.ewm(span=trend_window, adjust=False).mean()
        realized_vol = log_returns.rolling(vol_window).std()
        vol_of_vol = realized_vol.rolling(vol_window).std()
        range_atr = self._atr(high, low, close, 14)
        channel_high = high.rolling(trend_window).max().shift(1)
        channel_low = low.rolling(trend_window).min().shift(1)
        channel_width = (channel_high - channel_low).replace(0.0, np.nan)
        volume_mean = volume.rolling(vol_window).mean()
        volume_std = volume.rolling(vol_window).std().replace(0.0, np.nan)

        features["ema_fast_distance"] = close / ema_fast - 1.0
        features["ema_slow_distance"] = close / ema_slow - 1.0
        features["trend_slope"] = ema_slow.pct_change(8)
        features["realized_vol"] = realized_vol
        features["vol_of_vol"] = vol_of_vol
        features["atr_pct"] = range_atr / close
        features["channel_position"] = (close - channel_low) / channel_width
        features["breakout_pressure"] = (close - channel_high) / range_atr.replace(0.0, np.nan)
        features["breakdown_pressure"] = (channel_low - close) / range_atr.replace(0.0, np.nan)
        features["volume_z"] = (volume - volume_mean) / volume_std
        features["downside_vol"] = log_returns.where(log_returns < 0.0, 0.0).rolling(vol_window).std()
        features["upside_vol"] = log_returns.where(log_returns > 0.0, 0.0).rolling(vol_window).std()
        features["return_skew"] = returns.rolling(vol_window).skew()
        features["return_kurt"] = returns.rolling(vol_window).kurt()

        latest = features.tail(1).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        if latest.empty:
            return None
        return latest.fillna(0.0)

    def _predict_probabilities(self, row: Any) -> dict[str, float]:
        raw = self._model.predict_proba(row)[0]
        classes = list(getattr(self._model, "classes_", [-1, 0, 1]))
        output = {"down": 0.0, "flat": 0.0, "up": 0.0}
        for klass, probability in zip(classes, raw, strict=False):
            label = str(klass).lower()
            if label in {"-1", "down", "short", "bear"}:
                output["down"] = float(probability)
            elif label in {"0", "flat", "neutral"}:
                output["flat"] = float(probability)
            elif label in {"1", "up", "long", "bull"}:
                output["up"] = float(probability)
        if output == {"down": 0.0, "flat": 0.0, "up": 0.0} and len(raw) == 2:
            output["down"] = float(raw[0])
            output["up"] = float(raw[1])
        return output

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        previous_close = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
            axis=1,
        ).max(axis=1)
        return true_range.ewm(alpha=1.0 / window, adjust=False).mean()

    @staticmethod
    def _normalized_entropy(probabilities: list[float]) -> float:
        values = np.array(probabilities, dtype=float)
        values = values[values > 0.0]
        if values.size <= 1:
            return 0.0
        entropy = -float(np.sum(values * np.log(values)))
        return entropy / float(np.log(len(probabilities)))

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
