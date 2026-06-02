from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.strategy.adaptive_trend_expansion_pro import AdaptiveTrendExpansionProStrategy
from backend.strategy.capitulation_reversal_pro import CapitulationReversalProStrategy
from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import (
    bars_since_entry,
    position_side,
    regime_snapshot,
    safe_denominator,
)
from backend.strategy.signals import Signal, SignalType
from backend.strategy.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy


class MetaLabeledAlphaAllocatorProStrategy:
    def __init__(
        self,
        symbol: str,
        min_probability: float = 0.58,
        min_edge_atr: float = 0.15,
        high_conviction_probability: float = 0.65,
        fee_rate: float = 0.0004,
        slippage_bps: float = 2.0,
        cooldown_after_loss_bars: int = 4,
        max_holding_bars: int = 24,
    ) -> None:
        self._symbol = symbol
        self._min_probability = float(min_probability)
        self._min_edge_atr = float(min_edge_atr)
        self._high_conviction_probability = float(high_conviction_probability)
        self._fee_rate = float(fee_rate)
        self._slippage_bps = float(slippage_bps)
        self._cooldown_after_loss_bars = int(cooldown_after_loss_bars)
        self._max_holding_bars = int(max_holding_bars)
        self._trend = AdaptiveTrendExpansionProStrategy(symbol=symbol)
        self._reversal = CapitulationReversalProStrategy(symbol=symbol)
        self._squeeze = VolatilitySqueezeBreakoutStrategy(symbol=symbol)
        self._traders = [
            ("adaptive_trend_expansion_pro", self._trend, 1.8, 1.1),
            ("capitulation_reversal_pro", self._reversal, 1.2, 0.8),
            ("volatility_squeeze_breakout", self._squeeze, 1.7, 1.0),
        ]
        self._min_bars = max(getattr(strategy, "max_lookback", 256) for _, strategy, _, _ in self._traders)
        self.max_lookback = self._min_bars + self._max_holding_bars + 8

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        side = position_side(portfolio)
        raw_signals = [(name, avg_win, avg_loss, strategy.generate_signal_at(market, index, portfolio)) for name, strategy, avg_win, avg_loss in self._traders]
        regime = regime_snapshot(market, index)
        base_metadata = {
            "strategy": "meta_labeled_alpha_allocator_pro",
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
            "local_traders": [
                {
                    "trader": name,
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                }
                for name, _, _, signal in raw_signals
            ],
        }

        if side in {"LONG", "SHORT"}:
            exit_signal = self._select_exit(raw_signals, side)
            if exit_signal is not None:
                name, signal = exit_signal
                metadata = {**base_metadata, **signal.metadata, "selected_trader": name}
                return Signal(timestamp, self._symbol, SignalType.EXIT, max(0.65, signal.confidence), f"meta_allocator_exit:{signal.reason}", metadata)
            if bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.62, "meta_allocator_time_stop", base_metadata)
            return self._hold(timestamp, "meta_allocator_position_hold", base_metadata)

        candidates = []
        for name, avg_win, avg_loss, signal in raw_signals:
            if signal.signal_type not in {SignalType.LONG, SignalType.SHORT}:
                continue
            probability = self._probability(signal, name, regime)
            edge_atr = self._edge_atr(probability, avg_win, avg_loss, signal)
            uncertainty = max(0.0, 1.0 - abs(probability - 0.5) * 2.0)
            strategy_ok = self._strategy_regime_ok(name, signal.signal_type, regime)
            threshold = self._high_conviction_probability if self._risk_state_is_defensive(portfolio) else self._min_probability
            if probability <= threshold or edge_atr <= self._min_edge_atr or not strategy_ok:
                continue
            score = edge_atr * probability * max(0.1, signal.confidence)
            candidates.append((score, name, probability, edge_atr, uncertainty, signal))

        if not candidates:
            return self._hold(timestamp, "meta_gate_flat_no_positive_edge", base_metadata)

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_name, probability, edge_atr, uncertainty, selected_signal = candidates[0]
        metadata = {
            **base_metadata,
            **selected_signal.metadata,
            "selected_trader": selected_name,
            "meta_probability": float(probability),
            "expected_edge_atr": float(edge_atr),
            "uncertainty": float(uncertainty),
            "meta_gate": "accepted",
        }
        confidence = min(1.0, max(selected_signal.confidence, probability))
        return Signal(timestamp, self._symbol, selected_signal.signal_type, confidence, f"meta_allocator_accept:{selected_signal.reason}", metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _select_exit(self, raw_signals: list[tuple[str, float, float, Signal]], side: str) -> tuple[str, Signal] | None:
        exit_signals = [(name, signal) for name, _, _, signal in raw_signals if signal.signal_type == SignalType.EXIT]
        if not exit_signals:
            return None
        exit_signals.sort(key=lambda item: item[1].confidence, reverse=True)
        return exit_signals[0]

    def _probability(self, signal: Signal, trader_name: str, regime: dict[str, Any]) -> float:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        probability = 0.52 + 0.18 * max(0.0, min(1.0, float(signal.confidence)))
        volume_z = float(metadata.get("volume_z", 0.0) or 0.0)
        adx = float(metadata.get("adx", regime.get("adx", 20.0)) or 20.0)
        choppiness = float(metadata.get("choppiness", regime.get("choppiness", 50.0)) or 50.0)
        vwap_dev = abs(float(metadata.get("vwap_dev_atr", 0.0) or 0.0))
        if volume_z > 0.75:
            probability += min(0.06, 0.015 * volume_z)
        if trader_name == "adaptive_trend_expansion_pro":
            probability += 0.04 if signal.signal_type == SignalType.LONG and regime["probabilities"].get("steady_uptrend", 0.0) + regime["probabilities"].get("rapid_uptrend", 0.0) > 0.45 else 0.0
            probability += 0.04 if signal.signal_type == SignalType.SHORT and regime["probabilities"].get("steady_downtrend", 0.0) + regime["probabilities"].get("rapid_downtrend", 0.0) > 0.45 else 0.0
            probability += min(0.04, max(0.0, (adx - 20.0) / 100.0))
            probability -= min(0.05, max(0.0, (choppiness - 52.0) / 100.0))
        elif trader_name == "capitulation_reversal_pro":
            probability += min(0.07, 0.02 * vwap_dev)
            probability += 0.04 if regime["probabilities"].get("shock", 0.0) > 0.25 else 0.0
            probability -= 0.08 if signal.signal_type == SignalType.LONG and regime["regime"] == "rapid_downtrend" else 0.0
            probability -= 0.08 if signal.signal_type == SignalType.SHORT and regime["regime"] == "rapid_uptrend" else 0.0
        else:
            probability += 0.04 if regime["regime"] in {"steady_uptrend", "steady_downtrend", "rapid_uptrend", "rapid_downtrend"} else 0.0
            probability += min(0.05, max(0.0, (volume_z - 1.0) / 20.0))
        return float(min(0.86, max(0.0, probability)))

    def _edge_atr(self, probability: float, avg_win: float, avg_loss: float, signal: Signal) -> float:
        cost_atr = self._cost_atr(signal)
        return float(probability * avg_win - (1.0 - probability) * avg_loss - cost_atr)

    def _cost_atr(self, signal: Signal) -> float:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        close = float(metadata.get("close", 0.0) or 0.0)
        atr = float(metadata.get("atr", 0.0) or 0.0)
        if close <= 0.0 or atr <= 0.0:
            return 0.05
        cost_price = close * (self._fee_rate * 2.0 + self._slippage_bps / 10000.0 * 2.0)
        return float(cost_price / safe_denominator(atr, 1.0))

    def _strategy_regime_ok(self, trader_name: str, signal_type: SignalType, regime: dict[str, Any]) -> bool:
        if trader_name == "adaptive_trend_expansion_pro":
            if signal_type == SignalType.LONG:
                return regime["probabilities"].get("steady_uptrend", 0.0) + regime["probabilities"].get("rapid_uptrend", 0.0) >= 0.35
            if signal_type == SignalType.SHORT:
                return regime["probabilities"].get("steady_downtrend", 0.0) + regime["probabilities"].get("rapid_downtrend", 0.0) >= 0.35
        if trader_name == "capitulation_reversal_pro":
            if signal_type == SignalType.LONG:
                return regime["regime"] != "rapid_downtrend"
            if signal_type == SignalType.SHORT:
                return regime["regime"] != "rapid_uptrend"
        return regime["regime"] != "chop" or regime["probabilities"].get("shock", 0.0) < 0.35

    @staticmethod
    def _risk_state_is_defensive(portfolio: Any) -> bool:
        losses = int(getattr(portfolio, "consecutive_losses", 0) or 0)
        daily_drawdown = float(getattr(portfolio, "daily_drawdown", 0.0) or 0.0)
        return losses >= 3 or daily_drawdown <= -0.02

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
