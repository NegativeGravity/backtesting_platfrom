from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.helformer import HelformerNextCloseForecaster, RegimeAwareCalibrator
from backend.strategy.base import BaseStrategy, PortfolioView
from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class HelformerMomentumStrategy(BaseStrategy):
    def __init__(
        self,
        symbol: str,
        model_artifact_path: str | Path,
        fee_rate: float = 0.0004,
        allow_short: bool = False,
    ) -> None:
        self._symbol = symbol
        self._artifact_path = Path(model_artifact_path)
        self._forecaster = HelformerNextCloseForecaster(self._artifact_path)
        self._fee_rate = float(self._forecaster.metadata.get("fee_rate", fee_rate))
        self._allow_short = bool(self._forecaster.metadata.get("allow_short", allow_short))
        self._calibrator = self._clone_calibrator()
        self._pending_update: dict[str, Any] | None = None
        self.max_lookback = int(self._forecaster.sequence_length + 220)

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: PortfolioView) -> Signal:
        self._update_calibrator(market, index)
        timestamp = market.timestamp(index)
        prediction = self._forecaster.predict_market(market, index, calibrator=self._calibrator)
        if prediction is None:
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "helformer_not_enough_sequence_history", self._base_metadata(None))
        target_position = self._target_position(prediction)
        metadata = self._base_metadata(prediction)
        metadata["target_position"] = target_position
        metadata["target_notional_fraction"] = abs(target_position)
        self._pending_update = {
            "target_index": index + 1,
            "predicted_log_move": float(prediction["predicted_log_move"]),
            "regime": str(prediction["regime"]),
        }
        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))
        position_side = str(getattr(portfolio, "position_side", "") or "")
        confidence = float(min(1.0, abs(target_position)))
        if position_quantity == 0.0:
            if target_position > 0.0:
                return Signal(timestamp, self._symbol, SignalType.LONG, confidence, "helformer_positive_target_position", metadata)
            if target_position < 0.0 and self._allow_short:
                return Signal(timestamp, self._symbol, SignalType.SHORT, confidence, "helformer_negative_target_position", metadata)
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "helformer_flat_target_position", metadata)
        if position_side == "LONG" and target_position <= 0.0:
            return Signal(timestamp, self._symbol, SignalType.EXIT, confidence, "helformer_long_target_cleared", metadata)
        if position_side == "SHORT" and target_position >= 0.0:
            return Signal(timestamp, self._symbol, SignalType.EXIT, confidence, "helformer_short_target_cleared", metadata)
        return Signal(timestamp, self._symbol, SignalType.HOLD, confidence, "helformer_keep_position", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: PortfolioView) -> Signal:
        if market_window.empty:
            return Signal(pd.Timestamp.utcnow(), self._symbol, SignalType.HOLD, 0.0, "empty_market_window", {})
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _target_position(self, prediction: dict[str, Any]) -> float:
        edge = float(prediction["predicted_log_move"])
        volatility = max(float(prediction["regime_vol"]), 1e-6)
        threshold = max(2.0 * self._fee_rate, 0.12 * volatility)
        position = float(np.tanh(edge / (1.5 * volatility)))
        if not self._allow_short:
            position = float(np.clip(position, 0.0, 1.0))
        if abs(edge) < threshold:
            return 0.0
        return position

    def _update_calibrator(self, market: MarketDataView, index: int) -> None:
        if self._calibrator is None or self._pending_update is None:
            return
        if int(self._pending_update["target_index"]) != index:
            return
        if index <= 0:
            return
        previous_close = float(market.arrays.close[index - 1])
        current_close = float(market.arrays.close[index])
        if previous_close <= 0.0 or current_close <= 0.0:
            return
        actual_log_move = math.log(current_close / previous_close)
        self._calibrator.update(
            actual_log_move=actual_log_move,
            predicted_log_move=float(self._pending_update["predicted_log_move"]),
            regime=str(self._pending_update["regime"]),
        )
        self._pending_update = None

    def _clone_calibrator(self) -> RegimeAwareCalibrator | None:
        if self._forecaster.calibrator is None:
            return None
        return RegimeAwareCalibrator.from_state(self._forecaster.calibrator.state_dict())

    def _base_metadata(self, prediction: dict[str, Any] | None) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "strategy": "helformer_momentum",
            "model_artifact_path": str(self._artifact_path),
            "fee_rate": self._fee_rate,
            "allow_short": self._allow_short,
        }
        if prediction:
            metadata.update({
                "current_close": prediction["current_close"],
                "raw_scaled_prediction": prediction["raw_scaled_prediction"],
                "raw_predicted_log_move": prediction["raw_predicted_log_move"],
                "predicted_log_move": prediction["predicted_log_move"],
                "raw_predicted_close": prediction["raw_predicted_close"],
                "predicted_close": prediction["predicted_close"],
                "regime_vol": prediction["regime_vol"],
                "regime": prediction["regime"],
                "calibration_confidence": prediction["calibration_confidence"],
            })
        return metadata
