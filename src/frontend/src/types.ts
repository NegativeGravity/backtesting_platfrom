export type JsonRecord = Record<string, unknown>;

export type StrategyName =
  | 'mean_reversion'
  | 'ml_momentum'
  | 'adaptive_trend_breakout'
  | 'liquidity_sweep_reversal'
  | 'ml_regime_meta_label'
  | 'dl_temporal_fusion_momentum'
  | 'helformer_momentum'
  | 'adaptive_trend_expansion_pro'
  | 'capitulation_reversal_pro'
  | 'volatility_squeeze_breakout'
  | 'meta_labeled_alpha_allocator_pro';

export type BacktestJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface BacktestRunListItem {
  run_id: string;
  strategy?: string;
  symbol?: string;
  final_equity?: number;
  total_return?: number;
  benchmark_return?: number;
  max_drawdown?: number;
  created_at?: string;
  status?: string;
}

export interface BacktestRunRequest {
  strategy: StrategyName;
  config_path: string;
  model_artifact_path?: string | null;
  helformer_artifact_path?: string | null;
}

export interface BacktestRunResponse {
  run_id?: string | null;
  run_dir?: string | null;
  summary?: JsonRecord;
  job_id?: string | null;
  status?: BacktestJobStatus | string | null;
  error?: string | null;
  message?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
  heartbeat_at?: number | null;
  elapsed_seconds?: number | null;
}

export interface BacktestJobResponse {
  job_id: string;
  status: BacktestJobStatus;
  run_id?: string | null;
  run_dir?: string | null;
  summary?: JsonRecord;
  error?: string | null;
  message?: string | null;
  created_at: number;
  updated_at: number;
  heartbeat_at?: number | null;
  elapsed_seconds: number;
}

export interface LiveStrategyWorkerConfig {
  worker_id?: string | null;
  strategy: StrategyName;
  model_artifact_path?: string | null;
  helformer_artifact_path?: string | null;
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

export interface LiveReplayRequest {
  config_path: string;
  robots: LiveRobotConfig[];
  replay_delay_seconds: number;
}

export interface LiveReplayResponse {
  session_id: string;
  websocket_url?: string;
}

export interface LiveEvent {
  event_id?: string;
  event_type: string;
  timestamp?: string;
  source: string;
  payload: JsonRecord;
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
  time: number | string;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface LinePoint {
  time: number | string;
  value: number;
}

export interface TradeMarker {
  time: number | string;
  position: 'aboveBar' | 'belowBar' | 'inBar';
  color?: string;
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  text?: string;
}

export interface ChartDataResponse {
  candles: CandlePoint[];
  equity_curve: LinePoint[];
  benchmark_curve: LinePoint[];
  markers: TradeMarker[];
}

export interface ModelArtifactItem {
  path: string;
  name: string;
  size_bytes: number;
  modified_at: string;
  strategy?: string | null;
  artifact_role?: string | null;
  model_type?: string | null;
}

export interface ReportResponse {
  run_id?: string;
  summary: JsonRecord;
  trades: TradeRecord[];
  closed_positions?: TradeRecord[];
  equity_curve?: JsonRecord[];
  benchmark_curve?: JsonRecord[];
  execution_log?: ExecutionLogRecord[];
  orders?: ExecutionLogRecord[];
  config?: JsonRecord;
  diagnostics?: JsonRecord;
  model?: JsonRecord;
}

export type PositionSide = 'LONG' | 'SHORT';

export interface LivePositionPayload {
  position_id: string;
  robot_id: string;
  robot_name: string;
  worker_id: string;
  strategy: string;
  symbol: string;
  side: PositionSide | string;
  quantity: number;
  entry_time: string;
  exit_time?: string;
  entry_price: number;
  exit_price?: number;
  current_price?: number;
  stop_loss?: number;
  take_profit?: number;
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

export type RunDetailTab =
  | 'overview'
  | 'trades'
  | 'orders'
  | 'equity'
  | 'diagnostics'
  | 'config'
  | 'model';
