import type {
  BacktestRunListItem,
  BacktestRunRequest,
  BacktestRunResponse,
  ChartDataResponse,
  ReportResponse,
  RobotBacktestRequest,
} from "../types.ts";

const API_BASE_URL = "http://127.0.0.1:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
    ...options,
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null);
    const message = errorPayload?.detail ?? `Request failed: ${response.status}`;
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export async function listBacktests(): Promise<BacktestRunListItem[]> {
  return request<BacktestRunListItem[]>("/backtests");
}

export async function runBacktest(payload: BacktestRunRequest): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>("/backtests/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function runRobotBacktest(payload: RobotBacktestRequest): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>("/robot-backtests/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getReport(runId: string): Promise<ReportResponse> {
  return request<ReportResponse>(`/backtests/${runId}`);
}

export async function getChartData(runId: string): Promise<ChartDataResponse> {
  return request<ChartDataResponse>(`/backtests/${runId}/chart-data`);
}
