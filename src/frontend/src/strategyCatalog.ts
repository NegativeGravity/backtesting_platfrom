import type { StrategyName } from './types';

export interface StrategyCatalogItem {
  value: StrategyName;
  label: string;
  badge: string;
  description: string;
  requiresArtifact: boolean;
  supportsHelformerForecast: boolean;
}

export const STRATEGY_CATALOG: StrategyCatalogItem[] = [
  { value: 'mean_reversion', label: 'Mean Reversion', badge: 'Classic', description: 'Z-score reversal around rolling mean.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'adaptive_trend_breakout', label: 'Adaptive Trend', badge: 'Trend', description: 'EMA regime, Donchian breakout, ATR expansion.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'regime_adaptive_btc_trend_breakout', label: 'Regime-Adaptive BTC Trend', badge: 'BTC', description: 'EMA50/200, Donchian 48, ADX, volatility and volume-confirmed BTC 1H breakout.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'liquidity_sweep_reversal', label: 'Liquidity Sweep', badge: 'Sweep', description: 'False-breakout reversal with wick and volume confirmation.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'liquidation_shock_mean_reversion', label: 'Liquidation Shock Reversion', badge: 'Shock', description: 'Panic/liquidation proxy reversal using return shock, volume spike, RSI and CLV.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'adaptive_trend_expansion_pro', label: 'Trend Expansion Pro', badge: 'Pro', description: 'KAMA, Donchian close, CHOP, Vortex and volatility-targeted trend expansion.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'capitulation_reversal_pro', label: 'Capitulation Reversal Pro', badge: 'Pro', description: 'Robust shock, VWAP stretch, Connors RSI and wick-quality reversal.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'volatility_squeeze_breakout', label: 'Squeeze Breakout', badge: 'SQZ', description: 'TTM squeeze release with Donchian close breakout and volume expansion.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'meta_labeled_ensemble_alpha', label: 'Meta-Labeled Ensemble Alpha', badge: 'Meta', description: 'Meta gate over trend breakout, shock mean reversion, squeeze breakout and cash.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'meta_labeled_alpha_allocator_pro', label: 'Meta Alpha Allocator Pro', badge: 'Meta+', description: 'Regime-aware allocator over trend, capitulation reversal, squeeze and cash.', requiresArtifact: false, supportsHelformerForecast: true },
  { value: 'ml_momentum', label: 'ML Momentum', badge: 'ML', description: 'Causal feature classifier with probability thresholds.', requiresArtifact: true, supportsHelformerForecast: true },
  { value: 'ml_regime_meta_label', label: 'ML Regime Meta', badge: 'Meta', description: 'Regime-aware meta-labeling model.', requiresArtifact: true, supportsHelformerForecast: true },
  { value: 'dl_temporal_fusion_momentum', label: 'DL Temporal Fusion', badge: 'DL', description: 'Sequence model for temporal edge detection.', requiresArtifact: true, supportsHelformerForecast: true },
  { value: 'helformer_momentum', label: 'Helformer Momentum', badge: 'HF', description: 'Regime-calibrated next-close forecast momentum.', requiresArtifact: true, supportsHelformerForecast: false },
];
