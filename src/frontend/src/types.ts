export type StrategyName =
  | "mean_reversion"
  | "ml_momentum"
  | "adaptive_trend_breakout"
  | "liquidity_sweep_reversal"
  | "ml_regime_meta_label"
  | "dl_temporal_fusion_momentum";

export interface BacktestRunListItem {
  run_id: string;
  strategy?: string;
  symbol?: string;
  final_equity?: number;
  total_return?: number;
  benchmark_return?: number;
  max_drawdown?: number;
  created_at?: string;
}

export interface BacktestRunRequest {
  strategy: StrategyName;
  config_path: string;
  model_artifact_path?: string | null;
}

export interface BacktestRunResponse {
  run_id: string;
  run_dir?: string;
  summary: Record<string, unknown>;
}

export interface LiveStrategyWorkerConfig {
  worker_id?: string | null;
  strategy: StrategyName;
  model_artifact_path?: string | null;
}

export interface LiveRobotConfig {
  robot_id: string;
  display_name: string;
  strategy_workers: LiveStrategyWorkerConfig[];
}

export interface RobotBacktestRequest {
  config_path: string;
  robot: LiveRobotConfig;
}

export interface TradeRecord {
  trade_id?: string;
  position_id?: string;
  symbol?: string;
  entry_time?: string;
  exit_time?: string;
  entry_price?: number;
  exit_price?: number;
  quantity?: number;
  gross_pnl?: number;
  net_pnl?: number;
  fees?: number;
  slippage_cost?: number;
  return_pct?: number;
  exit_reason?: string;
  worker_id?: string;
  strategy?: string;
  side?: string;
}

export interface ExecutionLogRecord {
  timestamp?: string;
  robot_id?: string;
  robot_name?: string;
  worker_id?: string;
  strategy?: string;
  symbol?: string;
  action?: string;
  side?: string;
  quantity?: number;
  requested_bar_time?: string;
  filled_bar_time?: string;
  fill_price?: number;
  fee?: number;
  slippage_cost?: number;
  status?: string;
  reason?: string;
}

export interface CandlePoint {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LinePoint {
  time: number;
  value: number;
}

export interface TradeMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown";
  text: string;
}

export interface ChartDataResponse {
  candles: CandlePoint[];
  equity_curve: LinePoint[];
  benchmark_curve: LinePoint[];
  markers: TradeMarker[];
}

export interface ReportResponse {
  run_id?: string;
  summary: Record<string, unknown>;
  trades: TradeRecord[];
  equity_curve?: Record<string, unknown>[];
  benchmark_curve?: Record<string, unknown>[];
  execution_log?: ExecutionLogRecord[];
}

export type PositionSide = "LONG" | "SHORT";

export interface LivePositionPayload {
  position_id: string;
  robot_id: string;
  robot_name: string;
  worker_id: string;
  strategy: string;
  symbol: string;
  side: PositionSide;
  quantity: number;
  entry_time: string;
  exit_time?: string;
  entry_price: number;
  exit_price?: number;
  current_price?: number;
  stop_loss: number;
  take_profit: number;
  notional?: number;
  current_notional?: number;
  gross_unrealized_pnl?: number;
  unrealized_pnl?: number;
  unrealized_return_pct?: number;
  gross_pnl?: number;
  net_pnl?: number;
  return_pct?: number;
  exit_reason?: string;
}
