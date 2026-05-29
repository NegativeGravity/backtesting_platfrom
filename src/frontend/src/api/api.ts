import type {
  BacktestRunListItem,
  BacktestRunRequest,
  BacktestRunResponse,
  ChartDataResponse,
  ModelArtifactItem,
  ReportResponse,
  RobotBacktestRequest,
} from '../types';
import { buildHttpUrl } from './config';
import { request } from './http';


export function listModelArtifacts(signal?: AbortSignal): Promise<ModelArtifactItem[]> {
  return request<ModelArtifactItem[]>('/model-artifacts', { signal });
}

export function listBacktests(signal?: AbortSignal): Promise<BacktestRunListItem[]> {
  return request<BacktestRunListItem[]>('/backtests', { signal });
}

export function runBacktest(payload: BacktestRunRequest): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>('/backtests/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function runRobotBacktest(payload: RobotBacktestRequest): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>('/robot-backtests/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export function getReport(runId: string, signal?: AbortSignal): Promise<ReportResponse> {
  return request<ReportResponse>(`/backtests/${encodeURIComponent(runId)}`, { signal });
}

export function getChartData(runId: string, signal?: AbortSignal): Promise<ChartDataResponse> {
  return request<ChartDataResponse>(`/backtests/${encodeURIComponent(runId)}/chart-data`, { signal });
}

export function getBacktestJob(jobId: string, signal?: AbortSignal): Promise<BacktestRunResponse> {
  return request<BacktestRunResponse>(`/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export async function waitForBacktestJob(jobId: string, signal?: AbortSignal): Promise<BacktestRunResponse> {
  const terminalStatuses = new Set(['completed', 'failed', 'cancelled']);
  const startedAt = Date.now();
  let delayMs = 750;

  while (true) {
    if (signal?.aborted) throw new Error('Backtest polling was cancelled.');
    if (Date.now() - startedAt > 15 * 60 * 1000) throw new Error('Backtest job timed out.');

    const job = await getBacktestJob(jobId, signal);
    if (terminalStatuses.has(String(job.status))) {
      if (job.status === 'failed' || job.status === 'cancelled') {
        throw new Error(job.error || `Backtest job ${job.status}.`);
      }
      if (!job.run_id) {
        throw new Error('Backtest job completed without a run_id.');
      }
      return job;
    }

    await new Promise<void>((resolve, reject) => {
      const timeoutId = window.setTimeout(resolve, delayMs);
      signal?.addEventListener('abort', () => {
        window.clearTimeout(timeoutId);
        reject(new Error('Backtest polling was cancelled.'));
      }, { once: true });
    });
    delayMs = Math.min(Math.round(delayMs * 1.35), 5000);
  }
}

export function getBacktestExportUrl(runId: string, format: 'csv' | 'parquet' | 'html'): string {
  return buildHttpUrl(`/backtests/${encodeURIComponent(runId)}/export?format=${format}`);
}