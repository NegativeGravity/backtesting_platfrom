from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.ml.helformer import HelformerNextCloseForecaster
from backend.strategy.market_view import MarketArrays, MarketDataView, call_strategy_signal
from backend.strategy.signals import Signal, SignalType


_FORECASTER_CACHE: dict[str, HelformerNextCloseForecaster] = {}


class HelformerNextBarProjectionStrategy:
    def __init__(
        self,
        base_strategy: Any,
        symbol: str,
        helformer_artifact_path: str | Path,
        min_abs_log_move: float = 0.0,
    ) -> None:
        self._base_strategy = base_strategy
        self._symbol = symbol
        self._artifact_path = Path(helformer_artifact_path)
        self._forecaster = self._load_forecaster(self._artifact_path)
        self._min_abs_log_move = float(min_abs_log_move)
        self.max_lookback = int(max(
            int(getattr(base_strategy, "max_lookback", 0) or 0),
            int(getattr(self._forecaster, "sequence_length", 24)) + 220,
        ) + 2)

    @staticmethod
    def _load_forecaster(path: Path) -> HelformerNextCloseForecaster:
        key = str(path.resolve())
        forecaster = _FORECASTER_CACHE.get(key)
        if forecaster is None:
            forecaster = HelformerNextCloseForecaster(path)
            _FORECASTER_CACHE[key] = forecaster
        return forecaster

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)

        if self._has_position(portfolio):
            exit_signal = call_strategy_signal(self._base_strategy, market, index, portfolio)
            if exit_signal.signal_type == SignalType.EXIT:
                return exit_signal
            return Signal(
                timestamp,
                self._symbol,
                SignalType.HOLD,
                exit_signal.confidence,
                "helformer_projection_position_management_hold",
                self._metadata(exit_signal, None, None),
            )

        if index + 1 >= len(market):
            probe_signal = Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "helformer_projection_no_next_bar_slot", {})
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, probe_signal.reason, self._metadata(probe_signal, None, None))

        prediction = self._forecaster.predict_market(market, index, calibrator=True)
        if prediction is None:
            probe_signal = Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "helformer_projection_not_enough_history", {})
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, probe_signal.reason, self._metadata(probe_signal, None, None))

        predicted_log_move = float(prediction.get("predicted_log_move", 0.0))
        if abs(predicted_log_move) < self._min_abs_log_move:
            probe_signal = Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, "helformer_projection_move_too_small", {})
            return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, probe_signal.reason, self._metadata(probe_signal, None, prediction))

        projected_market = self._project_next_bar(market, index, prediction)
        projected_signal = call_strategy_signal(self._base_strategy, projected_market, index + 1, portfolio)
        metadata = self._metadata(projected_signal, projected_market, prediction)
        metadata["decision_bar_time"] = timestamp.isoformat()
        metadata["execution_bar_time"] = projected_market.timestamp(index + 1).isoformat()
        metadata["entry_contract"] = "evaluate_t_plus_1_projected_conditions_at_t_execute_next_open"

        if projected_signal.signal_type in {SignalType.LONG, SignalType.SHORT}:
            return Signal(
                timestamp=timestamp,
                symbol=projected_signal.symbol,
                signal_type=projected_signal.signal_type,
                confidence=projected_signal.confidence,
                reason=f"helformer_projected_next_bar_{projected_signal.reason}",
                metadata=metadata,
            )

        return Signal(timestamp, self._symbol, SignalType.HOLD, projected_signal.confidence, "helformer_projected_next_bar_no_entry", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        if market_window.empty:
            return Signal(pd.Timestamp.utcnow(), self._symbol, SignalType.HOLD, 0.0, "empty_market_window", {})
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market, len(market) - 1, portfolio)

    def _project_next_bar(self, market: MarketDataView, index: int, prediction: dict[str, Any]) -> MarketDataView:
        next_index = index + 1
        stop = index + 1
        current_close = float(market.arrays.close[index])
        predicted_close = float(prediction["predicted_close"])
        open_price = current_close
        atr = self._safe_atr(market, index, current_close)
        body = abs(predicted_close - open_price)
        projected_range = max(body, atr * 0.35, current_close * 0.0005)
        high = max(open_price, predicted_close) + projected_range * 0.15
        low = max(1e-12, min(open_price, predicted_close) - projected_range * 0.15)
        volume = self._project_volume(market, index)

        ts = np.concatenate([market.arrays.ts[:stop], np.asarray([market.arrays.ts[next_index]], dtype=np.int64)])
        open_ = np.concatenate([market.arrays.open[:stop], np.asarray([open_price], dtype=np.float64)])
        high_ = np.concatenate([market.arrays.high[:stop], np.asarray([high], dtype=np.float64)])
        low_ = np.concatenate([market.arrays.low[:stop], np.asarray([low], dtype=np.float64)])
        close_ = np.concatenate([market.arrays.close[:stop], np.asarray([predicted_close], dtype=np.float64)])
        volume_ = np.concatenate([market.arrays.volume[:stop], np.asarray([volume], dtype=np.float64)])
        extra = {
            key: np.concatenate([values[:stop], np.asarray([self._project_extra(values, index)], dtype=np.asarray(values).dtype)])
            for key, values in market.arrays.extra.items()
        }
        return MarketDataView(MarketArrays(ts=ts, open=open_, high=high_, low=low_, close=close_, volume=volume_, extra=extra))

    @staticmethod
    def _has_position(portfolio: Any) -> bool:
        if bool(getattr(portfolio, "has_position", False)):
            return True
        return abs(float(getattr(portfolio, "position_quantity", 0.0) or 0.0)) > 0.0

    @staticmethod
    def _safe_atr(market: MarketDataView, index: int, current_close: float) -> float:
        try:
            value = float(market.atr(14)[index])
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:
            pass
        return max(current_close * 0.001, 1e-9)

    @staticmethod
    def _project_volume(market: MarketDataView, index: int) -> float:
        start = max(0, index + 1 - 24)
        values = np.asarray(market.arrays.volume[start:index + 1], dtype=np.float64)
        values = values[np.isfinite(values) & (values >= 0.0)]
        if values.size == 0:
            return 0.0
        return float(np.median(values))

    @staticmethod
    def _project_extra(values: np.ndarray, index: int) -> float:
        array = np.asarray(values)
        if array.size == 0:
            return 0.0
        value = array[min(max(index, 0), array.size - 1)]
        if np.issubdtype(array.dtype, np.number):
            numeric = float(value)
            return numeric if np.isfinite(numeric) else 0.0
        return 0.0

    def _metadata(self, strategy_signal: Signal, projected_market: MarketDataView | None, prediction: dict[str, Any] | None) -> dict[str, Any]:
        metadata = dict(strategy_signal.metadata or {})
        metadata.update({
            "entry_evaluation_model": "helformer_projected_next_bar_conditions",
            "base_strategy_reason": strategy_signal.reason,
            "base_strategy_signal_type": strategy_signal.signal_type.value,
            "helformer_artifact_path": str(self._artifact_path),
        })
        if prediction is not None:
            metadata.update({
                "helformer_current_close": prediction.get("current_close"),
                "helformer_predicted_close_t_plus_1": prediction.get("predicted_close"),
                "helformer_predicted_log_move_t_plus_1": prediction.get("predicted_log_move"),
                "helformer_raw_predicted_log_move_t_plus_1": prediction.get("raw_predicted_log_move"),
                "helformer_regime": prediction.get("regime"),
                "helformer_regime_vol": prediction.get("regime_vol"),
                "helformer_calibration_confidence": prediction.get("calibration_confidence"),
            })
        if projected_market is not None and len(projected_market) > 0:
            idx = len(projected_market) - 1
            metadata.update({
                "projected_bar_time": projected_market.timestamp(idx).isoformat(),
                "projected_open": projected_market.open_at(idx),
                "projected_high": projected_market.high_at(idx),
                "projected_low": projected_market.low_at(idx),
                "projected_close": projected_market.close_at(idx),
                "projected_volume": projected_market.volume_at(idx),
            })
        return metadata
