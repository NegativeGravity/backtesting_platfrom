from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backend.ml.artifacts import load_model_bundle
from backend.ml.features import FEATURE_COLUMNS, build_latest_inference_features
from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class MLMomentumStrategy:
    """ML momentum strategy with cached, array-backed inference features."""

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
        self._feature_columns = list(bundle.get("feature_columns", FEATURE_COLUMNS))
        self.max_lookback = 64

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        current_close = market.close_at(index)
        metadata: dict[str, Any] = {
            "close": current_close,
            "probability": None,
            "long_threshold": self._long_threshold,
            "short_threshold": self._short_threshold,
            "exit_probability": self._exit_probability,
            "model_artifact_path": str(self._artifact_path),
        }

        feature_row = market.ml_momentum_features_at(index)
        if feature_row is None or feature_row.empty:
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "ml_not_enough_feature_history", metadata)

        missing_columns = [column for column in self._feature_columns if column not in feature_row.columns]
        if missing_columns:
            metadata["missing_columns"] = missing_columns
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "ml_missing_feature_columns", metadata)

        features = feature_row[self._feature_columns]
        scaled_features = self._scaler.transform(features)
        probability = float(self._model.predict_proba(scaled_features)[0, 1])
        metadata["probability"] = probability

        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))
        position_side = str(getattr(portfolio, "position_side", ""))

        if position_quantity == 0:
            if probability >= self._long_threshold:
                return Signal(timestamp, self._symbol, SignalType.LONG, probability, "ml_probability_above_long_threshold", metadata)
            if probability <= self._short_threshold:
                return Signal(timestamp, self._symbol, SignalType.SHORT, 1.0 - probability, "ml_probability_below_short_threshold", metadata)

        if position_quantity != 0:
            if position_side == "LONG" and probability < self._exit_probability:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 1.0 - probability, "ml_long_exit_probability_below_threshold", metadata)
            if position_side == "SHORT" and probability > self._exit_probability:
                return Signal(timestamp, self._symbol, SignalType.EXIT, probability, "ml_short_exit_probability_above_threshold", metadata)

        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "ml_no_action", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        if market_window.empty:
            return Signal(pd.Timestamp.utcnow(), self._symbol, SignalType.HOLD, 0.0, "empty_market_window", {})
        # Legacy compatibility; internal engines use generate_signal_at.
        feature_row = build_latest_inference_features(market_window)
        market = MarketDataView.from_frame(market_window)
        if feature_row is not None and not feature_row.empty:
            market._cache["legacy_ml_feature_row"] = feature_row  # harmless hint for debugging
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)