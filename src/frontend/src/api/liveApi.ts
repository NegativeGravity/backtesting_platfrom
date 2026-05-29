import type {
  BacktestRunResponse,
  LiveReplayRequest,
  LiveReplayResponse,
  RobotBacktestRequest,
} from '../types';
import { getBacktestJob, waitForBacktestJob } from './api';
import { buildWsUrl } from './config';
import { request } from './http';

export function startLiveReplay(payload: LiveReplayRequest): Promise<LiveReplayResponse> {
  return request<LiveReplayResponse>('/live-replay/start', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function runLiveRobotBacktest(payload: RobotBacktestRequest): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>('/robot-backtests/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function createLiveReplaySocket(websocketUrl: string): WebSocket {
  return new WebSocket(buildWsUrl(websocketUrl));
}

export { getBacktestJob, waitForBacktestJob };