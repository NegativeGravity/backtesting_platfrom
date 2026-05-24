from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from trading_system.ml.artifacts import load_model_bundle
from trading_system.ml.features import FEATURE_COLUMNS, build_latest_inference_features
from trading_system.strategy.signals import Signal, SignalType


class MLMomentumStrategy:
    """
    ML momentum strategy with long and short support.

    Current model is binary:
    - high probability means upward momentum -> LONG
    - low probability means weak/upside-negative forecast -> SHORT

    Later this should become a three-class model:
    UP / DOWN / NEUTRAL.
    """

    def __init__(
        self,
        symbol: str,
        model_artifact_path: Path | str,
        probability_threshold: float,
        exit_probability: float,
        short_probability_threshold: float | None = None,
    ) -> None:
        self._symbol = symbol
        self._artifact_path = Path(model_artifact_path)
        self._long_threshold = float(probability_threshold)
        self._exit_probability = float(exit_probability)
        self._short_threshold = (
            float(short_probability_threshold)
            if short_probability_threshold is not None
            else 1.0 - float(probability_threshold)
        )

        bundle = load_model_bundle(self._artifact_path)

        self._model = bundle["model"]
        self._scaler = bundle["scaler"]

        feature_columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        self._feature_columns = list(feature_columns)

    def generate_signal(
        self,
        market_window: pd.DataFrame,
        portfolio: Any,
    ) -> Signal:
        if market_window.empty:
            return Signal(
                timestamp=pd.Timestamp.utcnow(),
                symbol=self._symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="empty_market_window",
                metadata={},
            )

        current_bar = market_window.iloc[-1]
        timestamp = pd.Timestamp(current_bar["timestamp"])
        current_close = float(current_bar["close"])

        metadata: dict[str, Any] = {
            "close": current_close,
            "probability": None,
            "long_threshold": self._long_threshold,
            "short_threshold": self._short_threshold,
            "exit_probability": self._exit_probability,
            "model_artifact_path": str(self._artifact_path),
        }

        feature_row = build_latest_inference_features(market_window)

        if feature_row is None or feature_row.empty:
            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="ml_not_enough_feature_history",
                metadata=metadata,
            )

        missing_columns = [
            column for column in self._feature_columns if column not in feature_row.columns
        ]

        if missing_columns:
            metadata["missing_columns"] = missing_columns

            return Signal(
                timestamp=timestamp,
                symbol=self._symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="ml_missing_feature_columns",
                metadata=metadata,
            )

        features = feature_row[self._feature_columns]
        scaled_features = self._scaler.transform(features)
        probability = float(self._model.predict_proba(scaled_features)[0, 1])

        metadata["probability"] = probability

        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))
        position_side = str(getattr(portfolio, "position_side", ""))

        if position_quantity == 0:
            if probability >= self._long_threshold:
                return Signal(
                    timestamp=timestamp,
                    symbol=self._symbol,
                    signal_type=SignalType.LONG,
                    confidence=probability,
                    reason="ml_probability_above_long_threshold",
                    metadata=metadata,
                )

            if probability <= self._short_threshold:
                return Signal(
                    timestamp=timestamp,
                    symbol=self._symbol,
                    signal_type=SignalType.SHORT,
                    confidence=1.0 - probability,
                    reason="ml_probability_below_short_threshold",
                    metadata=metadata,
                )

        if position_quantity != 0:
            if position_side == "LONG" and probability < self._exit_probability:
                return Signal(
                    timestamp=timestamp,
                    symbol=self._symbol,
                    signal_type=SignalType.EXIT,
                    confidence=1.0 - probability,
                    reason="ml_long_exit_probability_below_threshold",
                    metadata=metadata,
                )

            if position_side == "SHORT" and probability > self._exit_probability:
                return Signal(
                    timestamp=timestamp,
                    symbol=self._symbol,
                    signal_type=SignalType.EXIT,
                    confidence=probability,
                    reason="ml_short_exit_probability_above_threshold",
                    metadata=metadata,
                )

        return Signal(
            timestamp=timestamp,
            symbol=self._symbol,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            reason="ml_no_action",
            metadata=metadata,
        )