from __future__ import annotations

from pathlib import Path
from typing import Any

from trading_system.core.config import AppConfig
from trading_system.core.paths import resolve_project_path
from trading_system.strategy.adaptive_trend_breakout import AdaptiveTrendBreakoutStrategy
from trading_system.strategy.dl_temporal_fusion_momentum import DLTemporalFusionMomentumStrategy
from trading_system.strategy.liquidity_sweep_reversal import LiquiditySweepReversalStrategy
from trading_system.strategy.mean_reversion import MeanReversionStrategy
from trading_system.strategy.ml_momentum import MLMomentumStrategy
from trading_system.strategy.ml_regime_meta_label import MLRegimeMetaLabelStrategy


SUPPORTED_STRATEGIES = {
    "mean_reversion",
    "ml_momentum",
    "adaptive_trend_breakout",
    "liquidity_sweep_reversal",
    "ml_regime_meta_label",
    "dl_temporal_fusion_momentum",
}

STRATEGIES_REQUIRING_ARTIFACT = {
    "ml_momentum",
    "ml_regime_meta_label",
    "dl_temporal_fusion_momentum",
}


def create_strategy(
    config: AppConfig,
    strategy_name: str | None = None,
    model_artifact_path: Path | str | None = None,
) -> Any:
    selected_strategy = strategy_name or _get_config_value(
        config.strategy,
        "name",
        default="mean_reversion",
    )

    if selected_strategy == "mean_reversion":
        return MeanReversionStrategy(
            symbol=config.data.symbol,
            lookback=int(
                _get_strategy_parameter(
                    config,
                    "lookback",
                    _get_strategy_parameter(config, "lookback_window", 20),
                )
            ),
            entry_z_score=float(_get_strategy_parameter(config, "entry_z_score", 2.0)),
            exit_z_score=float(_get_strategy_parameter(config, "exit_z_score", 0.5)),
        )

    if selected_strategy == "adaptive_trend_breakout":
        return AdaptiveTrendBreakoutStrategy(
            symbol=config.data.symbol,
            fast_ema=int(_get_strategy_parameter(config, "fast_ema", 20)),
            slow_ema=int(_get_strategy_parameter(config, "slow_ema", 50)),
            channel_window=int(_get_strategy_parameter(config, "channel_window", 40)),
            atr_window=int(_get_strategy_parameter(config, "atr_window", 14)),
            atr_baseline_window=int(_get_strategy_parameter(config, "atr_baseline_window", 80)),
            min_atr_expansion=float(_get_strategy_parameter(config, "min_atr_expansion", 1.05)),
            exit_ema=int(_get_strategy_parameter(config, "exit_ema", 20)),
        )

    if selected_strategy == "liquidity_sweep_reversal":
        return LiquiditySweepReversalStrategy(
            symbol=config.data.symbol,
            sweep_window=int(_get_strategy_parameter(config, "sweep_window", 30)),
            atr_window=int(_get_strategy_parameter(config, "atr_window", 14)),
            min_wick_atr=float(_get_strategy_parameter(config, "min_wick_atr", 0.35)),
            min_reclaim_pct=float(_get_strategy_parameter(config, "min_reclaim_pct", 0.15)),
            volume_window=int(_get_strategy_parameter(config, "volume_window", 40)),
            min_volume_z=float(_get_strategy_parameter(config, "min_volume_z", -0.25)),
            trend_ema=int(_get_strategy_parameter(config, "trend_ema", 100)),
            max_trend_distance_atr=float(_get_strategy_parameter(config, "max_trend_distance_atr", 3.0)),
            exit_zscore=float(_get_strategy_parameter(config, "exit_zscore", 0.10)),
            max_holding_bars=int(_get_strategy_parameter(config, "max_holding_bars", 24)),
        )

    if selected_strategy == "ml_momentum":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        probability_threshold = float(_get_config_value(config.ml, "probability_threshold", 0.60))
        return MLMomentumStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            probability_threshold=probability_threshold,
            exit_probability=float(_get_config_value(config.ml, "exit_probability", 0.50)),
            short_probability_threshold=float(
                _get_config_value(config.ml, "short_probability_threshold", 1.0 - probability_threshold)
            ),
        )

    if selected_strategy == "ml_regime_meta_label":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        return MLRegimeMetaLabelStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            lookback=int(_get_strategy_parameter(config, "ml_regime_lookback", 96)),
            volatility_window=int(_get_strategy_parameter(config, "ml_regime_volatility_window", 32)),
            trend_window=int(_get_strategy_parameter(config, "ml_regime_trend_window", 64)),
            min_confidence=float(_get_strategy_parameter(config, "ml_regime_min_confidence", 0.58)),
            exit_confidence=float(_get_strategy_parameter(config, "ml_regime_exit_confidence", 0.50)),
            max_regime_entropy=float(_get_strategy_parameter(config, "ml_regime_max_entropy", 0.68)),
        )

    if selected_strategy == "dl_temporal_fusion_momentum":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        return DLTemporalFusionMomentumStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            sequence_length=int(_get_strategy_parameter(config, "dl_sequence_length", 128)),
            long_threshold=float(_get_strategy_parameter(config, "dl_long_threshold", 0.60)),
            short_threshold=float(_get_strategy_parameter(config, "dl_short_threshold", 0.60)),
            exit_threshold=float(_get_strategy_parameter(config, "dl_exit_threshold", 0.48)),
            max_entropy=float(_get_strategy_parameter(config, "dl_max_entropy", 0.72)),
        )

    raise ValueError(
        f"Unknown strategy: {selected_strategy}. "
        f"Supported strategies are: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
    )


def _resolve_required_artifact(
    config: AppConfig,
    model_artifact_path: Path | str | None,
    strategy_name: str,
) -> Path:
    artifact_path = model_artifact_path or getattr(config.ml, "model_artifact_path", None)
    if artifact_path is None or not str(artifact_path).strip():
        raise ValueError(f"model_artifact_path is required for {strategy_name}.")
    return resolve_project_path(artifact_path)


def _get_strategy_parameter(config: AppConfig, key: str, default: Any) -> Any:
    direct_value = getattr(config.strategy, key, None)
    if direct_value is not None:
        return direct_value

    parameters = getattr(config.strategy, "parameters", None)
    if isinstance(parameters, dict):
        return parameters.get(key, default)

    return default


def _get_config_value(config_section: Any, key: str, default: Any) -> Any:
    value = getattr(config_section, key, None)
    if value is not None:
        return value

    if isinstance(config_section, dict):
        return config_section.get(key, default)

    return default
