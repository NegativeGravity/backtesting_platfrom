from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.strategy.base import BaseStrategy, PortfolioView
from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


@dataclass(frozen=True)
class RegimeMetaLabelConfig:
    lookback: int = 96
    volatility_window: int = 32
    trend_window: int = 64
    min_confidence: float = 0.58
    exit_confidence: float = 0.50
    max_regime_entropy: float = 0.68


class MLRegimeMetaLabelStrategy(BaseStrategy):
    """Meta-labeling strategy with precomputed array-backed feature rows."""

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
        self.max_lookback = max(lookback, trend_window, volatility_window) + 64

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: PortfolioView) -> Signal:
        timestamp = market.timestamp(index)
        metadata: dict[str, Any] = {"strategy": "ml_regime_meta_label"}
        features = market.ml_regime_features_at(
            index,
            trend_window=self._config.trend_window,
            volatility_window=self._config.volatility_window,
        )
        if features is None:
            return self._hold(timestamp, "ml_regime_not_enough_history", metadata)

        missing = [column for column in self._feature_columns if column not in features.columns]
        if missing:
            metadata["missing_columns"] = missing
            return self._hold(timestamp, "ml_regime_missing_features", metadata)

        row: Any = features[self._feature_columns]
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

    def generate_signal(self, market_window: pd.DataFrame, portfolio: PortfolioView) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

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
    def _normalized_entropy(probabilities: list[float]) -> float:
        values = np.array(probabilities, dtype=float)
        values = values[values > 0.0]
        if values.size <= 1:
            return 0.0
        entropy = -float(np.sum(values * np.log(values)))
        return entropy / float(np.log(len(probabilities)))

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)