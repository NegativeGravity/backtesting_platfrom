import type {
  BacktestJobResponse,
  BacktestRunListItem,
  BacktestRunRequest,
  BacktestRunResponse,
  ChartDataResponse,
  ModelArtifactItem,
  ReportResponse,
  RobotBacktestRequest,
} from '../types';
import { buildHttpUrl, JOB_STATUS_TIMEOUT_MS } from './config';
import { ApiError, request } from './http';

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

export function getBacktestJob(jobId: string, signal?: AbortSignal): Promise<BacktestJobResponse> {
  return request<BacktestJobResponse>(`/jobs/${encodeURIComponent(jobId)}`, {
    signal,
    timeoutMs: JOB_STATUS_TIMEOUT_MS,
  });
}

export function isTerminalBacktestJob(status: unknown): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled';
}

export async function waitForBacktestJob(jobId: string, signal?: AbortSignal): Promise<BacktestJobResponse> {
  const startedAt = Date.now();
  let delayMs = 1000;
  let transientFailures = 0;

  while (true) {
    if (signal?.aborted) throw new Error('Backtest polling was cancelled.');
    if (Date.now() - startedAt > 12 * 60 * 60 * 1000) throw new Error('Backtest job timed out.');

    try {
      const job = await getBacktestJob(jobId, signal);
      transientFailures = 0;

      if (isTerminalBacktestJob(job.status)) {
        if (job.status === 'failed' || job.status === 'cancelled') {
          throw new Error(job.error || `Backtest job ${job.status}.`);
        }
        if (!job.run_id) {
          throw new Error('Backtest job completed without a run_id.');
        }
        return job;
      }
    } catch (error) {
      if (signal?.aborted) throw error;
      if (error instanceof ApiError && error.status === 404) throw error;
      transientFailures += 1;
      if (transientFailures > 40) throw error;
    }

    await sleep(delayMs, signal);
    delayMs = Math.min(Math.round(delayMs * 1.25), 15000);
  }
}

function sleep(delayMs: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeoutId = window.setTimeout(resolve, delayMs);
    signal?.addEventListener('abort', () => {
      window.clearTimeout(timeoutId);
      reject(new Error('Backtest polling was cancelled.'));
    }, { once: true });
  });
}

export function getBacktestExportUrl(runId: string, format: 'csv' | 'parquet' | 'html'): string {
  return buildHttpUrl(`/backtests/${encodeURIComponent(runId)}/export?format=${format}`);
}
