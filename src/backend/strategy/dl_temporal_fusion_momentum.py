from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.base import BaseStrategy, PortfolioView
from backend.strategy.market_view import MarketDataView
from backend.strategy.signals import Signal, SignalType


class DLTemporalFusionMomentumStrategy(BaseStrategy):
    """TorchScript sequence-model strategy with cached sequence feature matrix."""

    def __init__(
        self,
        symbol: str,
        model_artifact_path: str | Path,
        sequence_length: int = 128,
        long_threshold: float = 0.60,
        short_threshold: float = 0.60,
        exit_threshold: float = 0.48,
        max_entropy: float = 0.72,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("dl_temporal_fusion_momentum requires torch to be installed.") from exc

        self._torch = torch
        self._symbol = symbol
        self._artifact_path = Path(model_artifact_path)
        if self._artifact_path.is_file():
            self._model_path = self._artifact_path
            self._metadata_path = self._artifact_path.with_name("metadata.json")
        else:
            self._model_path = self._artifact_path / "model.pt"
            self._metadata_path = self._artifact_path / "metadata.json"

        if not self._model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {self._model_path}")

        metadata = self._load_metadata()
        self._sequence_length = int(metadata.get("sequence_length", sequence_length))
        self._long_threshold = float(metadata.get("long_threshold", long_threshold))
        self._short_threshold = float(metadata.get("short_threshold", short_threshold))
        self._exit_threshold = float(metadata.get("exit_threshold", exit_threshold))
        self._max_entropy = float(metadata.get("max_entropy", max_entropy))
        self._feature_columns = list(metadata.get("feature_columns", self._default_feature_columns()))
        self._feature_mean = np.array(metadata.get("feature_mean", [0.0] * len(self._feature_columns)), dtype=np.float32)
        self._feature_std = np.array(metadata.get("feature_std", [1.0] * len(self._feature_columns)), dtype=np.float32)
        self._feature_std = np.where(self._feature_std == 0.0, 1.0, self._feature_std)
        self._model = torch.jit.load(str(self._model_path), map_location="cpu")
        self._model.eval()
        self.max_lookback = self._sequence_length + 80

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: PortfolioView) -> Signal:
        timestamp = market.timestamp(index)
        metadata: dict[str, Any] = {"strategy": "dl_temporal_fusion_momentum"}
        sequence = market.dl_sequence_at(
            index,
            feature_columns=self._feature_columns,
            sequence_length=self._sequence_length,
            feature_mean=self._feature_mean,
            feature_std=self._feature_std,
        )
        if sequence is None:
            return self._hold(timestamp, "dl_not_enough_sequence_history", metadata)

        probabilities = self._infer_probabilities(sequence)
        p_down, p_flat, p_up = probabilities
        confidence = float(max(p_down, p_up))
        entropy = self._normalized_entropy(probabilities)
        metadata.update(
            {
                "p_down": float(p_down),
                "p_flat": float(p_flat),
                "p_up": float(p_up),
                "confidence": confidence,
                "entropy": entropy,
            }
        )

        position_quantity = float(getattr(portfolio, "position_quantity", 0.0))
        position_side = str(getattr(portfolio, "position_side", ""))

        if position_quantity == 0:
            if entropy > self._max_entropy:
                return self._hold(timestamp, "dl_entropy_gate", metadata)
            if p_up >= self._long_threshold and p_up > p_down:
                return Signal(timestamp, self._symbol, SignalType.LONG, float(p_up), "dl_temporal_long_edge", metadata)
            if p_down >= self._short_threshold and p_down > p_up:
                return Signal(timestamp, self._symbol, SignalType.SHORT, float(p_down), "dl_temporal_short_edge", metadata)
            return self._hold(timestamp, "dl_no_actionable_edge", metadata)

        if position_side == "LONG" and p_up < self._exit_threshold:
            return Signal(timestamp, self._symbol, SignalType.EXIT, float(1.0 - p_up), "dl_exit_long_edge_decay", metadata)
        if position_side == "SHORT" and p_down < self._exit_threshold:
            return Signal(timestamp, self._symbol, SignalType.EXIT, float(1.0 - p_down), "dl_exit_short_edge_decay", metadata)

        return self._hold(timestamp, "dl_hold_position", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: PortfolioView) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _load_metadata(self) -> dict[str, Any]:
        if not self._metadata_path.exists():
            return {}
        with self._metadata_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _infer_probabilities(self, sequence: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            tensor = torch.from_numpy(sequence).unsqueeze(0).float()
            output = self._model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            values = output.detach().cpu().numpy()[0].astype(np.float64)
            if values.shape[0] == 1:
                p_up = float(1.0 / (1.0 + np.exp(-values[0])))
                return np.array([1.0 - p_up, 0.0, p_up], dtype=np.float64)
            if not np.isclose(values.sum(), 1.0, atol=1e-4) or np.any(values < 0.0):
                values = self._softmax(values)
            if values.shape[0] == 2:
                return np.array([values[0], 0.0, values[1]], dtype=np.float64)
            return values[:3]

    @staticmethod
    def _default_feature_columns() -> list[str]:
        return [
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

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values)

    @staticmethod
    def _normalized_entropy(probabilities: np.ndarray) -> float:
        values = probabilities[probabilities > 0.0]
        if values.size <= 1:
            return 0.0
        entropy = -float(np.sum(values * np.log(values)))
        return entropy / float(np.log(len(probabilities)))

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
