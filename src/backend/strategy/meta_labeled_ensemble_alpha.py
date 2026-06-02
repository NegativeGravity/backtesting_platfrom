from __future__ import annotations

from typing import Any

import pandas as pd

from backend.strategy.liquidation_shock_mean_reversion import LiquidationShockMeanReversionStrategy
from backend.strategy.market_view import MarketDataView
from backend.strategy.pro_indicators import bars_since_entry, position_side, regime_snapshot, safe_denominator
from backend.strategy.regime_adaptive_btc_trend_breakout import RegimeAdaptiveBtcTrendBreakoutStrategy
from backend.strategy.signals import Signal, SignalType
from backend.strategy.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy


class MetaLabeledEnsembleAlphaStrategy:
    def __init__(
        self,
        symbol: str,
        min_probability: float = 0.58,
        min_edge_atr: float = 0.12,
        high_conviction_probability: float = 0.65,
        fee_rate: float = 0.0004,
        slippage_bps: float = 2.0,
        max_holding_bars: int = 24,
    ) -> None:
        self._symbol = symbol
        self._min_probability = float(min_probability)
        self._min_edge_atr = float(min_edge_atr)
        self._high_conviction_probability = float(high_conviction_probability)
        self._fee_rate = float(fee_rate)
        self._slippage_bps = float(slippage_bps)
        self._max_holding_bars = int(max_holding_bars)
        self._trend = RegimeAdaptiveBtcTrendBreakoutStrategy(symbol=symbol)
        self._reversion = LiquidationShockMeanReversionStrategy(symbol=symbol)
        self._squeeze = VolatilitySqueezeBreakoutStrategy(symbol=symbol)
        self._traders = [
            ("regime_adaptive_btc_trend_breakout", self._trend, 1.5, 1.0),
            ("liquidation_shock_mean_reversion", self._reversion, 1.2, 0.9),
            ("volatility_squeeze_breakout", self._squeeze, 1.5, 1.0),
        ]
        self._min_bars = max(int(getattr(strategy, "max_lookback", 256)) for _, strategy, _, _ in self._traders)
        self.max_lookback = self._min_bars + self._max_holding_bars + 16

    def generate_signal_at(self, market: MarketDataView, index: int, portfolio: Any) -> Signal:
        timestamp = market.timestamp(index)
        if index + 1 < self._min_bars:
            return self._hold(timestamp, "not_enough_history", {"required_bars": self._min_bars})

        regime = regime_snapshot(market, index)
        raw_signals = [(name, avg_win, avg_loss, strategy.generate_signal_at(market, index, portfolio)) for name, strategy, avg_win, avg_loss in self._traders]
        metadata = {
            "strategy": "meta_labeled_ensemble_alpha",
            "regime": regime["regime"],
            "regime_probabilities": regime["probabilities"],
            "local_signals": [
                {
                    "strategy": name,
                    "signal_type": signal.signal_type.value,
                    "confidence": signal.confidence,
                    "reason": signal.reason,
                }
                for name, _, _, signal in raw_signals
            ],
        }

        side = position_side(portfolio)
        if side in {"LONG", "SHORT"}:
            exit_signal = self._best_exit(raw_signals)
            if exit_signal is not None:
                name, signal = exit_signal
                return Signal(
                    timestamp,
                    self._symbol,
                    SignalType.EXIT,
                    max(0.64, signal.confidence),
                    f"meta_labeled_ensemble_exit:{signal.reason}",
                    {**metadata, **signal.metadata, "selected_strategy": name},
                )
            if bars_since_entry(portfolio) >= self._max_holding_bars:
                return Signal(timestamp, self._symbol, SignalType.EXIT, 0.62, "meta_labeled_ensemble_time_stop", metadata)
            return self._hold(timestamp, "meta_labeled_ensemble_position_hold", metadata)

        candidates = []
        threshold = self._high_conviction_probability if self._defensive_state(portfolio) else self._min_probability
        for name, avg_win, avg_loss, signal in raw_signals:
            if signal.signal_type not in {SignalType.LONG, SignalType.SHORT}:
                continue
            probability = self._probability(name, signal, regime)
            edge_atr = self._edge_atr(probability, avg_win, avg_loss, signal)
            if probability <= threshold or edge_atr <= self._min_edge_atr or not self._regime_allows(name, signal.signal_type, regime):
                continue
            score = probability * edge_atr * max(0.1, signal.confidence)
            candidates.append((score, name, probability, edge_atr, signal))

        if not candidates:
            return self._hold(timestamp, "meta_labeled_ensemble_flat", metadata)

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_name, probability, edge_atr, signal = candidates[0]
        selected_metadata = {
            **metadata,
            **signal.metadata,
            "selected_strategy": selected_name,
            "meta_probability": float(probability),
            "expected_edge_atr": float(edge_atr),
            "meta_gate": "accepted",
        }
        confidence = min(1.0, max(signal.confidence, probability))
        return Signal(timestamp, self._symbol, signal.signal_type, confidence, f"meta_labeled_ensemble_accept:{signal.reason}", selected_metadata)

    def generate_signal(self, market_window: pd.DataFrame, portfolio: Any) -> Signal:
        market = MarketDataView.from_frame(market_window)
        return self.generate_signal_at(market=market, index=len(market) - 1, portfolio=portfolio)

    def _best_exit(self, raw_signals: list[tuple[str, float, float, Signal]]) -> tuple[str, Signal] | None:
        exits = [(name, signal) for name, _, _, signal in raw_signals if signal.signal_type == SignalType.EXIT]
        if not exits:
            return None
        exits.sort(key=lambda item: item[1].confidence, reverse=True)
        return exits[0]

    def _probability(self, name: str, signal: Signal, regime: dict[str, Any]) -> float:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        probability = 0.51 + 0.20 * max(0.0, min(1.0, float(signal.confidence)))
        volume = float(metadata.get("volume_z", 0.0) or 0.0)
        adx = float(metadata.get("adx", regime.get("adx", 20.0)) or 20.0)
        shock = abs(float(metadata.get("shock_sigma", 0.0) or metadata.get("robust_ret_z", 0.0) or 0.0))

        if volume > 0.5:
            probability += min(0.05, 0.012 * volume)
        if name == "regime_adaptive_btc_trend_breakout":
            trend_score = self._trend_probability(signal.signal_type, regime)
            probability += 0.06 * trend_score + min(0.04, max(0.0, (adx - 22.0) / 100.0))
        elif name == "liquidation_shock_mean_reversion":
            probability += min(0.08, 0.018 * shock)
            probability += 0.04 if regime["probabilities"].get("shock", 0.0) > 0.2 else 0.0
            probability -= 0.08 if signal.signal_type == SignalType.LONG and regime["regime"] == "rapid_downtrend" else 0.0
            probability -= 0.08 if signal.signal_type == SignalType.SHORT and regime["regime"] == "rapid_uptrend" else 0.0
        else:
            probability += 0.04 if regime["regime"] in {"steady_uptrend", "steady_downtrend", "rapid_uptrend", "rapid_downtrend"} else 0.0
        return float(min(0.88, max(0.0, probability)))

    def _edge_atr(self, probability: float, avg_win: float, avg_loss: float, signal: Signal) -> float:
        return float(probability * avg_win - (1.0 - probability) * avg_loss - self._cost_atr(signal))

    def _cost_atr(self, signal: Signal) -> float:
        metadata = signal.metadata if isinstance(signal.metadata, dict) else {}
        close = float(metadata.get("close", 0.0) or 0.0)
        atr = float(metadata.get("atr", 0.0) or 0.0)
        if close <= 0.0 or atr <= 0.0:
            return 0.05
        cost = close * (self._fee_rate * 2.0 + self._slippage_bps * 2.0 / 10000.0)
        return float(cost / safe_denominator(atr, 1.0))

    def _regime_allows(self, name: str, signal_type: SignalType, regime: dict[str, Any]) -> bool:
        if name == "regime_adaptive_btc_trend_breakout":
            return self._trend_probability(signal_type, regime) >= 0.30
        if name == "liquidation_shock_mean_reversion":
            if signal_type == SignalType.LONG:
                return regime["regime"] != "rapid_downtrend"
            if signal_type == SignalType.SHORT:
                return regime["regime"] != "rapid_uptrend"
        return regime["regime"] != "chop" or regime["probabilities"].get("shock", 0.0) < 0.35

    @staticmethod
    def _trend_probability(signal_type: SignalType, regime: dict[str, Any]) -> float:
        probabilities = regime["probabilities"]
        if signal_type == SignalType.LONG:
            return float(probabilities.get("steady_uptrend", 0.0) + probabilities.get("rapid_uptrend", 0.0))
        if signal_type == SignalType.SHORT:
            return float(probabilities.get("steady_downtrend", 0.0) + probabilities.get("rapid_downtrend", 0.0))
        return 0.0

    @staticmethod
    def _defensive_state(portfolio: Any) -> bool:
        losses = int(getattr(portfolio, "consecutive_losses", 0) or 0)
        daily_drawdown = float(getattr(portfolio, "daily_drawdown", 0.0) or 0.0)
        return losses >= 3 or daily_drawdown <= -0.02

    def _hold(self, timestamp: pd.Timestamp, reason: str, metadata: dict[str, Any]) -> Signal:
        return Signal(timestamp, self._symbol, SignalType.HOLD, 0.0, reason, metadata)
