from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.config import AppConfig
from backend.core.paths import resolve_model_artifact_path
from backend.strategy.adaptive_trend_breakout import AdaptiveTrendBreakoutStrategy
from backend.strategy.adaptive_trend_expansion_pro import AdaptiveTrendExpansionProStrategy
from backend.strategy.capitulation_reversal_pro import CapitulationReversalProStrategy
from backend.strategy.dl_temporal_fusion_momentum import DLTemporalFusionMomentumStrategy
from backend.strategy.helformer_momentum import HelformerMomentumStrategy
from backend.strategy.helformer_next_bar_projection import HelformerNextBarProjectionStrategy
from backend.strategy.liquidity_sweep_reversal import LiquiditySweepReversalStrategy
from backend.strategy.mean_reversion import MeanReversionStrategy
from backend.strategy.meta_labeled_alpha_allocator_pro import MetaLabeledAlphaAllocatorProStrategy
from backend.strategy.ml_momentum import MLMomentumStrategy
from backend.strategy.ml_regime_meta_label import MLRegimeMetaLabelStrategy
from backend.strategy.volatility_squeeze_breakout import VolatilitySqueezeBreakoutStrategy


SUPPORTED_STRATEGIES = {
    "mean_reversion",
    "ml_momentum",
    "adaptive_trend_breakout",
    "liquidity_sweep_reversal",
    "ml_regime_meta_label",
    "dl_temporal_fusion_momentum",
    "helformer_momentum",
    "adaptive_trend_expansion_pro",
    "capitulation_reversal_pro",
    "volatility_squeeze_breakout",
    "meta_labeled_alpha_allocator_pro",
}

STRATEGIES_REQUIRING_ARTIFACT = {
    "ml_momentum",
    "ml_regime_meta_label",
    "dl_temporal_fusion_momentum",
    "helformer_momentum",
}

STRATEGIES_USING_HELFORMER_FORECASTER = SUPPORTED_STRATEGIES - {"helformer_momentum"}


def create_strategy(
    config: AppConfig,
    strategy_name: str | None = None,
    model_artifact_path: Path | str | None = None,
    helformer_artifact_path: Path | str | None = None,
) -> Any:
    selected_strategy = strategy_name or _get_config_value(
        config.strategy,
        "name",
        default="mean_reversion",
    )

    if selected_strategy == "mean_reversion":
        strategy = MeanReversionStrategy(
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
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "adaptive_trend_breakout":
        strategy = AdaptiveTrendBreakoutStrategy(
            symbol=config.data.symbol,
            fast_ema=int(_get_strategy_parameter(config, "fast_ema", 20)),
            slow_ema=int(_get_strategy_parameter(config, "slow_ema", 50)),
            channel_window=int(_get_strategy_parameter(config, "channel_window", 40)),
            atr_window=int(_get_strategy_parameter(config, "atr_window", 14)),
            atr_baseline_window=int(_get_strategy_parameter(config, "atr_baseline_window", 80)),
            min_atr_expansion=float(_get_strategy_parameter(config, "min_atr_expansion", 1.05)),
            exit_ema=int(_get_strategy_parameter(config, "exit_ema", 20)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "liquidity_sweep_reversal":
        strategy = LiquiditySweepReversalStrategy(
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
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "adaptive_trend_expansion_pro":
        strategy = AdaptiveTrendExpansionProStrategy(
            symbol=config.data.symbol,
            kama_fast=int(_get_strategy_parameter(config, "trend_pro_kama_fast", 48)),
            kama_slow=int(_get_strategy_parameter(config, "trend_pro_kama_slow", 144)),
            channel_window=int(_get_strategy_parameter(config, "trend_pro_channel_window", 72)),
            choppiness_window=int(_get_strategy_parameter(config, "trend_pro_choppiness_window", 48)),
            min_adx=float(_get_strategy_parameter(config, "trend_pro_min_adx", 20.0)),
            max_entry_choppiness=float(_get_strategy_parameter(config, "trend_pro_max_entry_choppiness", 52.0)),
            no_trade_choppiness=float(_get_strategy_parameter(config, "trend_pro_no_trade_choppiness", 58.0)),
            min_volume_z=float(_get_strategy_parameter(config, "trend_pro_min_volume_z", 0.75)),
            max_entry_range_atr=float(_get_strategy_parameter(config, "trend_pro_max_entry_range_atr", 2.8)),
            risk_per_trade=float(_get_strategy_parameter(config, "trend_pro_risk_per_trade", 0.005)),
            max_notional_fraction=float(_get_strategy_parameter(config, "trend_pro_max_notional_fraction", 1.5)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "capitulation_reversal_pro":
        strategy = CapitulationReversalProStrategy(
            symbol=config.data.symbol,
            robust_z_threshold=float(_get_strategy_parameter(config, "capitulation_pro_robust_z_threshold", 2.3)),
            min_range_z=float(_get_strategy_parameter(config, "capitulation_pro_min_range_z", 1.6)),
            min_rvol=float(_get_strategy_parameter(config, "capitulation_pro_min_rvol", 1.8)),
            min_vwap_dev_atr=float(_get_strategy_parameter(config, "capitulation_pro_min_vwap_dev_atr", 1.8)),
            risk_per_trade=float(_get_strategy_parameter(config, "capitulation_pro_risk_per_trade", 0.0035)),
            max_notional_fraction=float(_get_strategy_parameter(config, "capitulation_pro_max_notional_fraction", 1.0)),
            require_confirmation=bool(_get_strategy_parameter(config, "capitulation_pro_require_confirmation", True)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "volatility_squeeze_breakout":
        strategy = VolatilitySqueezeBreakoutStrategy(
            symbol=config.data.symbol,
            min_squeeze_bars=int(_get_strategy_parameter(config, "squeeze_min_bars", 8)),
            channel_window=int(_get_strategy_parameter(config, "squeeze_channel_window", 48)),
            min_volume_z=float(_get_strategy_parameter(config, "squeeze_min_volume_z", 1.0)),
            risk_per_trade=float(_get_strategy_parameter(config, "squeeze_risk_per_trade", 0.0045)),
            max_notional_fraction=float(_get_strategy_parameter(config, "squeeze_max_notional_fraction", 1.25)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "meta_labeled_alpha_allocator_pro":
        strategy = MetaLabeledAlphaAllocatorProStrategy(
            symbol=config.data.symbol,
            min_probability=float(_get_strategy_parameter(config, "meta_allocator_min_probability", 0.58)),
            min_edge_atr=float(_get_strategy_parameter(config, "meta_allocator_min_edge_atr", 0.15)),
            high_conviction_probability=float(_get_strategy_parameter(config, "meta_allocator_high_conviction_probability", 0.65)),
            fee_rate=float(getattr(config.execution, "fee_rate", 0.0004)),
            slippage_bps=float(getattr(config.execution, "slippage_bps", 2.0)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "ml_momentum":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        probability_threshold = float(_get_config_value(config.ml, "probability_threshold", 0.60))
        strategy = MLMomentumStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            probability_threshold=probability_threshold,
            exit_probability=float(_get_config_value(config.ml, "exit_probability", 0.50)),
            short_probability_threshold=float(
                _get_config_value(config.ml, "short_probability_threshold", 1.0 - probability_threshold)
            ),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "ml_regime_meta_label":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        strategy = MLRegimeMetaLabelStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            lookback=int(_get_strategy_parameter(config, "ml_regime_lookback", 96)),
            volatility_window=int(_get_strategy_parameter(config, "ml_regime_volatility_window", 32)),
            trend_window=int(_get_strategy_parameter(config, "ml_regime_trend_window", 64)),
            min_confidence=float(_get_strategy_parameter(config, "ml_regime_min_confidence", 0.58)),
            exit_confidence=float(_get_strategy_parameter(config, "ml_regime_exit_confidence", 0.50)),
            max_regime_entropy=float(_get_strategy_parameter(config, "ml_regime_max_entropy", 0.68)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "dl_temporal_fusion_momentum":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        strategy = DLTemporalFusionMomentumStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            sequence_length=int(_get_strategy_parameter(config, "dl_sequence_length", 128)),
            long_threshold=float(_get_strategy_parameter(config, "dl_long_threshold", 0.60)),
            short_threshold=float(_get_strategy_parameter(config, "dl_short_threshold", 0.60)),
            exit_threshold=float(_get_strategy_parameter(config, "dl_exit_threshold", 0.48)),
            max_entropy=float(_get_strategy_parameter(config, "dl_max_entropy", 0.72)),
        )
        return _wrap_with_helformer_projection(config, selected_strategy, strategy, helformer_artifact_path)

    if selected_strategy == "helformer_momentum":
        artifact_path = _resolve_required_artifact(config, model_artifact_path, selected_strategy)
        return HelformerMomentumStrategy(
            symbol=config.data.symbol,
            model_artifact_path=artifact_path,
            fee_rate=float(getattr(config.execution, "fee_rate", 0.0004)),
            allow_short=bool(_get_strategy_parameter(config, "helformer_allow_short", False)),
        )

    raise ValueError(
        f"Unknown strategy: {selected_strategy}. "
        f"Supported strategies are: {', '.join(sorted(SUPPORTED_STRATEGIES))}."
    )


def _wrap_with_helformer_projection(
    config: AppConfig,
    selected_strategy: str,
    strategy: Any,
    helformer_artifact_path: Path | str | None,
) -> Any:
    if selected_strategy not in STRATEGIES_USING_HELFORMER_FORECASTER:
        return strategy
    if helformer_artifact_path is None or not str(helformer_artifact_path).strip():
        raise ValueError(
            f"helformer_artifact_path is required for {selected_strategy}. "
            "All non-Helformer-native strategies must evaluate projected t+1 conditions through the Helformer next-close forecaster."
        )
    artifact_path = resolve_model_artifact_path(helformer_artifact_path)
    return HelformerNextBarProjectionStrategy(
        base_strategy=strategy,
        symbol=config.data.symbol,
        helformer_artifact_path=artifact_path,
        min_abs_log_move=float(_get_strategy_parameter(config, "helformer_projection_min_abs_log_move", 0.0)),
    )


def _resolve_required_artifact(
    config: AppConfig,
    model_artifact_path: Path | str | None,
    strategy_name: str,
) -> Path:
    artifact_path = model_artifact_path or getattr(config.ml, "model_artifact_path", None)
    if artifact_path is None or not str(artifact_path).strip():
        raise ValueError(f"model_artifact_path is required for {strategy_name}.")
    return resolve_model_artifact_path(artifact_path)


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
