import type { BacktestRunResponse, StrategyName } from "../types";

const API_BASE_URL = "http://127.0.0.1:8000";
const WS_BASE_URL = "ws://127.0.0.1:8000";

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

export interface LiveReplayRequest {
  config_path: string;
  robots: LiveRobotConfig[];
  replay_delay_seconds: number;
}

export interface LiveReplayResponse {
  session_id: string;
  websocket_url: string;
}

export interface RobotBacktestRequest {
  config_path: string;
  robot: LiveRobotConfig;
}

export interface LiveEvent {
  event_id?: string;
  event_type: string;
  timestamp?: string;
  source: string;
  payload: Record<string, unknown>;
}

async function parseError(response: Response, fallback: string): Promise<Error> {
  const error = await response.json().catch(() => null);
  return new Error(error?.detail ?? fallback);
}

export async function startLiveReplay(payload: LiveReplayRequest): Promise<LiveReplayResponse> {
  const response = await fetch(`${API_BASE_URL}/live-replay/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw await parseError(response, "Failed to start live replay.");
  }

  return response.json() as Promise<LiveReplayResponse>;
}

export async function runRobotBacktest(payload: RobotBacktestRequest): Promise<BacktestRunResponse> {
  const response = await fetch(`${API_BASE_URL}/robot-backtests/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw await parseError(response, "Failed to run robot backtest.");
  }

  return response.json() as Promise<BacktestRunResponse>;
}

export function createLiveReplaySocket(sessionId: string): WebSocket {
  return new WebSocket(`${WS_BASE_URL}/ws/live-replay/${sessionId}`);
}
