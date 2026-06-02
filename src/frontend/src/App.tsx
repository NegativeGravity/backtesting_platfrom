import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, Navigate, Outlet, Route, Routes, useNavigate, useOutletContext, useParams } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getBacktestExportUrl, getBacktestJob, getChartData, getReport, isTerminalBacktestJob, listBacktests, listModelArtifacts, runBacktest } from './api/api';
import { ChartPanel } from './components/ChartPanel';
import { ErrorBoundary } from './components/ErrorBoundary';
import { LiveReplayPanel } from './components/LiveReplayPanel';
import { RunComparePanel } from './components/RunComparePanel';
import { SummaryCards } from './components/SummaryCards';
import { TradeBlotter } from './components/TradeBlotter';
import { TradesTable } from './components/TradesTable';
import { WorkerDiagnosticsPanel } from './components/WorkerDiagnosticsPanel';
import { StatusPill } from './components/StatusPill';
import { useUiStore } from './stores/uiStore';
import { STRATEGY_CATALOG } from './strategyCatalog';
import type { BacktestJobResponse, BacktestJobStatus, BacktestRunListItem, ChartDataResponse, ModelArtifactItem, ReportResponse, RunDetailTab, StrategyName } from './types';
import { downloadTextFile, reportToHtml } from './utils/exports';
import { formatDate, formatMoney, formatPercent, getRecord } from './utils/formatters';

const STRATEGIES = STRATEGY_CATALOG;

interface DashboardContext {
  modelArtifactPath: string;
  helformerArtifactPath: string;
  refreshRuns: () => Promise<void>;
}

interface TrackedJob {
  job_id: string;
  strategy: StrategyName;
  label: string;
  submittedAt: number;
  status: BacktestJobStatus;
  run_id?: string | null;
  error?: string | null;
  message?: string | null;
  elapsed_seconds?: number | null;
  updated_at?: number | null;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<DashboardLayout />}>
        <Route index element={<OverviewPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="live" element={<LivePage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function DashboardLayout() {
  const [strategy, setStrategy] = useState<StrategyName>('adaptive_trend_breakout');
  const [modelArtifactPath, setModelArtifactPath] = useState('');
  const [helformerArtifactPath, setHelformerArtifactPath] = useState('');
  const [useHelformerForecast, setUseHelformerForecast] = useState(false);
  const [trackedJobs, setTrackedJobs] = useState<TrackedJob[]>(() => readTrackedJobs());
  const openedRunsRef = useRef<Set<string>>(new Set());
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const currentStrategy = STRATEGIES.find((item) => item.value === strategy) ?? STRATEGIES[0];
  const supportsHelformerForecast = currentStrategy.supportsHelformerForecast;
  const usesHelformerProjection = supportsHelformerForecast && useHelformerForecast;
  const activeJobs = trackedJobs.filter((job) => !isTerminalBacktestJob(job.status));
  const activeJobIds = useMemo(() => activeJobs.map((job) => job.job_id), [activeJobs]);

  const runsQuery = useQuery({ queryKey: ['runs'], queryFn: ({ signal }) => listBacktests(signal), refetchInterval: activeJobs.length > 0 ? 5000 : 15000 });
  const artifactsQuery = useQuery({
    queryKey: ['model-artifacts'],
    queryFn: ({ signal }) => listModelArtifacts(signal),
    enabled: true,
    refetchOnWindowFocus: false,
  });
  const jobsQuery = useQuery({
    queryKey: ['active-backtest-jobs', activeJobIds.join('|')],
    queryFn: ({ signal }) => Promise.all(activeJobIds.map((jobId) => getBacktestJob(jobId, signal))),
    enabled: activeJobIds.length > 0,
    refetchInterval: activeJobIds.length > 0 ? 3000 : false,
    retry: 2,
    refetchOnWindowFocus: true,
  });
  const runs = validRuns(runsQuery.data);

  useEffect(() => {
    persistTrackedJobs(trackedJobs);
  }, [trackedJobs]);

  useEffect(() => {
    if (!jobsQuery.data?.length) return;

    const jobs = jobsQuery.data;
    setTrackedJobs((current) => mergeTrackedJobs(current, jobs));

    for (const job of jobs) {
      const runId = normalizeRunId(job.run_id);
      if (job.status !== 'completed' || !runId || openedRunsRef.current.has(runId)) continue;
      openedRunsRef.current.add(runId);
      void queryClient.invalidateQueries({ queryKey: ['runs'] });
      void queryClient.invalidateQueries({ queryKey: ['report', runId] });
      void queryClient.invalidateQueries({ queryKey: ['chart', runId] });
      navigate(`/runs/${runId}`);
    }
  }, [jobsQuery.data, navigate, queryClient]);

  const runMutation = useMutation({
    mutationFn: async () => {
      if (usesHelformerProjection && helformerArtifactPath.trim().length === 0) {
        throw new Error(`${currentStrategy.label} needs a Helformer next-close forecaster artifact.`);
      }
      if (currentStrategy.requiresArtifact && modelArtifactPath.trim().length === 0) {
        throw new Error(`${currentStrategy.label} needs a strategy model artifact path.`);
      }
      return runBacktest({
        strategy,
        config_path: 'configs/backtest.yaml',
        model_artifact_path: currentStrategy.requiresArtifact ? modelArtifactPath.trim() : null,
        helformer_artifact_path: usesHelformerProjection ? helformerArtifactPath.trim() : null,
        use_helformer_forecast: usesHelformerProjection,
      });
    },
    onSuccess: async (response) => {
      const runId = normalizeRunId(response.run_id);
      const jobId = normalizeRunId(response.job_id);

      if (runId) {
        await queryClient.invalidateQueries({ queryKey: ['runs'] });
        navigate(`/runs/${runId}`);
        return;
      }

      if (!jobId) {
        throw new Error('Backtest was submitted without a job id.');
      }

      const label = currentStrategy.label;
      setTrackedJobs((current) => [
        {
          job_id: jobId,
          strategy,
          label,
          submittedAt: Date.now(),
          status: (response.status as BacktestJobStatus) ?? 'queued',
          run_id: null,
          error: null,
          message: response.message ?? 'Queued',
          elapsed_seconds: response.elapsed_seconds ?? 0,
          updated_at: response.updated_at ?? Date.now() / 1000,
        },
        ...current.filter((job) => job.job_id !== jobId),
      ].slice(0, 12));
      await queryClient.invalidateQueries({ queryKey: ['active-backtest-jobs'] });
    },
  });

  const selectedRun = runs[0];
  const context = useMemo<DashboardContext>(() => ({
    modelArtifactPath,
    helformerArtifactPath,
    refreshRuns: async () => { await queryClient.invalidateQueries({ queryKey: ['runs'] }); },
  }), [modelArtifactPath, helformerArtifactPath, queryClient]);

  return (
    <main className="app-shell">
      <div className="bg-gradient" />
      <header className="topbar">
        <Link to="/" className="brand" aria-label="Quant Research Terminal home">
          <span>Q</span>
          <div><small>Quant Research Terminal</small><strong>Control Center</strong></div>
        </Link>
        <nav className="main-nav">
          <Link to="/">Reports</Link>
          <Link to="/live">Live Replay</Link>
        </nav>
        <div className="top-stats">
          <span>Runs <b>{runs.length}</b></span>
          <span>Status <b>{activeJobs.length > 0 ? `${activeJobs.length} background job${activeJobs.length === 1 ? '' : 's'}` : runsQuery.isFetching ? 'Syncing' : 'Ready'}</b></span>
          {selectedRun && <span>Last <b>{formatPercent(selectedRun.total_return)}</b></span>}
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <section className="panel launch-panel">
            <div className="panel-head">
              <div><span className="kicker">Backtest</span><h2>Launch Strategy</h2><p>Submit long-running jobs without blocking the UI.</p></div>
            </div>
            <div className="strategy-grid">
              {STRATEGIES.map((item) => (
                <button key={item.value} type="button" className={item.value === strategy ? 'active' : ''} onClick={() => setStrategy(item.value)} disabled={runMutation.isPending}>
                  <b>{item.label}</b><em>{item.badge}</em><small>{item.description}</small>
                </button>
              ))}
            </div>
            {supportsHelformerForecast && (
              <label className="helformer-toggle">
                <input
                  type="checkbox"
                  checked={useHelformerForecast}
                  disabled={runMutation.isPending}
                  onChange={(event) => setUseHelformerForecast(event.target.checked)}
                />
                <span>Use Helformer forecast for projected next-bar entries</span>
              </label>
            )}
            {usesHelformerProjection && (
              <ArtifactPicker
                title="Helformer Next-Close Forecaster"
                description="Used at bar t to project bar t+1 before evaluating the selected strategy."
                artifacts={filterArtifacts(artifactsQuery.data ?? [], 'helformer_next_close_forecaster')}
                selectedPath={helformerArtifactPath}
                isLoading={artifactsQuery.isLoading}
                error={artifactsQuery.error}
                onRefresh={() => artifactsQuery.refetch()}
                onSelect={setHelformerArtifactPath}
              />
            )}
            {currentStrategy.requiresArtifact && (
              <ArtifactPicker
                title={strategy === 'helformer_momentum' ? 'Helformer Momentum Strategy' : 'Strategy Model Artifact'}
                description={strategy === 'helformer_momentum' ? 'Dedicated artifact for native Helformer momentum.' : 'Model used by the selected ML/DL strategy itself.'}
                artifacts={filterArtifacts(artifactsQuery.data ?? [], strategy === 'helformer_momentum' ? 'helformer_momentum_strategy' : null)}
                selectedPath={modelArtifactPath}
                isLoading={artifactsQuery.isLoading}
                error={artifactsQuery.error}
                onRefresh={() => artifactsQuery.refetch()}
                onSelect={setModelArtifactPath}
              />
            )}
            <button className="primary" type="button" disabled={runMutation.isPending} onClick={() => runMutation.mutate()}>{runMutation.isPending ? 'Submitting…' : 'Launch Backtest'}</button>
            {runMutation.isPending && <RunProgress />}
            {runMutation.error && <div className="notice error"><strong>Backtest submit failed</strong><p>{runMutation.error.message}</p></div>}
            {jobsQuery.error && activeJobs.length > 0 && <div className="notice error"><strong>Job monitor reconnecting</strong><p>{jobsQuery.error instanceof Error ? jobsQuery.error.message : 'Could not refresh job status.'}</p></div>}
          </section>

          <BackgroundJobsPanel jobs={trackedJobs} onDismiss={(jobId) => setTrackedJobs((current) => current.filter((job) => job.job_id !== jobId))} />

          <section className="panel archive-panel">
            <div className="panel-head">
              <div><span className="kicker">Archive</span><h2>Saved Runs</h2></div>
              <button type="button" onClick={() => queryClient.invalidateQueries({ queryKey: ['runs'] })}>Refresh</button>
            </div>
            <div className="run-list">
              {runs.length === 0 && <div className="empty-state small">No saved runs yet.</div>}
              {runs.map((run) => <RunLink key={run.run_id} run={run} />)}
            </div>
          </section>
        </aside>

        <section className="content">
          <ErrorBoundary>
            <Outlet context={context} />
          </ErrorBoundary>
        </section>
      </section>
    </main>
  );
}

function readTrackedJobs(): TrackedJob[] {
  try {
    const raw = window.localStorage.getItem('quant-background-jobs');
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((job) => typeof job?.job_id === 'string').slice(0, 12) as TrackedJob[];
  } catch {
    return [];
  }
}

function persistTrackedJobs(jobs: TrackedJob[]): void {
  try {
    window.localStorage.setItem('quant-background-jobs', JSON.stringify(jobs.slice(0, 12)));
  } catch {
    return;
  }
}

function mergeTrackedJobs(current: TrackedJob[], updates: BacktestJobResponse[]): TrackedJob[] {
  const byId = new Map(current.map((job) => [job.job_id, job]));
  for (const update of updates) {
    const previous = byId.get(update.job_id);
    byId.set(update.job_id, {
      job_id: update.job_id,
      strategy: previous?.strategy ?? 'adaptive_trend_breakout',
      label: previous?.label ?? update.run_id ?? update.job_id,
      submittedAt: previous?.submittedAt ?? update.created_at * 1000,
      status: update.status,
      run_id: update.run_id,
      error: update.error,
      message: update.message,
      elapsed_seconds: update.elapsed_seconds,
      updated_at: update.updated_at,
    });
  }
  return [...byId.values()].sort((left, right) => right.submittedAt - left.submittedAt).slice(0, 12);
}

function BackgroundJobsPanel({ jobs, onDismiss }: { jobs: TrackedJob[]; onDismiss: (jobId: string) => void }) {
  if (jobs.length === 0) return null;
  return (
    <section className="panel archive-panel">
      <div className="panel-head">
        <div><span className="kicker">Background</span><h2>Backtest Jobs</h2><p>Jobs keep running server-side; completed runs open automatically.</p></div>
      </div>
      <div className="run-list">
        {jobs.map((job) => {
          const terminal = isTerminalBacktestJob(job.status);
          return (
            <article key={job.job_id} className="run-item">
              <b>{job.label}</b>
              <em>{job.status}</em>
              <small>{job.job_id}</small>
              <small>{job.run_id ? `Run ${job.run_id}` : job.error || job.message || 'Waiting for engine update'} · {formatDuration(job.elapsed_seconds)}</small>
              {terminal && <button type="button" onClick={() => onDismiss(job.job_id)}>Dismiss</button>}
            </article>
          );
        })}
      </div>
    </section>
  );
}

function normalizeRunId(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const runId = value.trim();
  return runId && runId !== 'null' && runId !== 'undefined' ? runId : null;
}

function validRuns(runs: BacktestRunListItem[] | undefined): BacktestRunListItem[] {
  return (runs ?? []).filter((run) => Boolean(normalizeRunId(run.run_id)));
}

function RunLink({ run }: { run: BacktestRunListItem }) {
  return (
    <Link to={`/runs/${run.run_id}`} className="run-item">
      <b>{run.run_id}</b>
      <em>{run.strategy ?? 'unknown'}</em>
      <small>Return {formatPercent(run.total_return)} · DD {formatPercent(run.max_drawdown)}</small>
      <small>{formatDate(run.created_at)}</small>
    </Link>
  );
}

function OverviewPage() {
  const runsQuery = useQuery({ queryKey: ['runs'], queryFn: ({ signal }) => listBacktests(signal), refetchInterval: 15000 });
  const runs = validRuns(runsQuery.data);
  return (
    <>
      <section className="panel hero-panel">
        <span className="kicker">Minimal, fast, chart-first</span>
        <h1>TradingView-style research platform for backtests and live replay.</h1>
        <p>Direct run URLs, high-performance charts, virtual tables, live websocket telemetry and clean quant hierarchy without noisy UI.</p>
        <div className="hero-actions">
          <Link to="/live" className="primary-link">Open Live Replay</Link>
          {runs[0] && <Link to={`/runs/${runs[0].run_id}`}>Open latest run</Link>}
        </div>
      </section>
      <RunComparePanel runs={runs} />
    </>
  );
}

function RunDetailPage() {
  const { runId = '' } = useParams();
  const safeRunId = normalizeRunId(runId);
  const activeTab = useUiStore((state) => state.activeRunTab);
  const setActiveTab = useUiStore((state) => state.setActiveRunTab);
  const reportQuery = useQuery({ queryKey: ['report', safeRunId], queryFn: ({ signal }) => getReport(safeRunId!, signal), enabled: Boolean(safeRunId) });
  const chartQuery = useQuery({ queryKey: ['chart', safeRunId], queryFn: ({ signal }) => getChartData(safeRunId!, signal), enabled: Boolean(safeRunId) });

  if (!safeRunId) return <Navigate to="/" replace />;

  const report = normalizeReport(reportQuery.data);
  const chartData = normalizeChart(chartQuery.data);
  const tabs: RunDetailTab[] = ['overview', 'trades', 'orders', 'equity', 'diagnostics', 'config', 'model'];

  if (reportQuery.isLoading || chartQuery.isLoading) return <section className="panel skeleton"><h2>Loading run…</h2><p>Fetching report, chart data and trades.</p></section>;
  if (reportQuery.error || chartQuery.error) {
    const error = reportQuery.error ?? chartQuery.error;
    return <section className="notice error"><strong>Could not load run</strong><p>{error instanceof Error ? error.message : 'Unknown error'}</p></section>;
  }

  return (
    <section className="run-detail">
      <div className="panel run-header">
        <div><span className="kicker">Run detail</span><h1>{safeRunId}</h1><p>Deep report, chart replay, diagnostics and exports.</p></div>
        <div className="run-actions">
          <StatusPill status="loaded" tone="good" />
          <button type="button" onClick={() => window.open(getBacktestExportUrl(safeRunId, 'parquet'), '_blank')}>Parquet</button>
          <button type="button" onClick={() => window.open(getBacktestExportUrl(safeRunId, 'csv'), '_blank')}>Server CSV</button>
          {report && <button type="button" onClick={() => downloadTextFile(`${safeRunId}.html`, reportToHtml(safeRunId, report), 'text/html;charset=utf-8')}>HTML</button>}
        </div>
      </div>

      <div className="tabbar" role="tablist">
        {tabs.map((tab) => <button key={tab} type="button" role="tab" className={activeTab === tab ? 'active' : ''} onClick={() => setActiveTab(tab)}>{tab}</button>)}
      </div>

      {activeTab === 'overview' && <><SummaryCards summary={report?.summary ?? null} /><ChartPanel chartData={chartData} trades={report?.trades ?? []} /></>}
      {activeTab === 'trades' && <TradesTable trades={report?.trades ?? []} />}
      {activeTab === 'orders' && <TradeBlotter records={report?.execution_log ?? report?.orders ?? []} />}
      {activeTab === 'equity' && <><ChartPanel chartData={chartData} trades={report?.trades ?? []} title="Equity, Benchmark & Drawdown" /><EquityDistribution report={report} /></>}
      {activeTab === 'diagnostics' && <WorkerDiagnosticsPanel report={report} />}
      {activeTab === 'config' && <JsonPanel title="Run Config" data={report?.config ?? getRecord(report?.summary).config ?? { message: 'No config found in report.' }} />}
      {activeTab === 'model' && <JsonPanel title="Model Metadata" data={report?.model ?? getRecord(report?.summary).model ?? getRecord(report?.summary).artifact ?? { message: 'No model metadata found in report.' }} />}
    </section>
  );
}

function LivePage() {
  const { modelArtifactPath, helformerArtifactPath, refreshRuns } = useOutletContext<DashboardContext>();
  return <LiveReplayPanel modelArtifactPath={modelArtifactPath} helformerArtifactPath={helformerArtifactPath} onCompleted={refreshRuns} />;
}

function JsonPanel({ title, data }: { title: string; data: unknown }) {
  return (
    <section className="panel json-panel">
      <div className="panel-head">
        <div><span className="kicker">Structured data</span><h2>{title}</h2></div>
        <button type="button" onClick={() => downloadTextFile(`${title}.json`, JSON.stringify(data, null, 2), 'application/json;charset=utf-8')}>Export JSON</button>
      </div>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </section>
  );
}

function EquityDistribution({ report }: { report: ReportResponse | null }) {
  const trades = report?.trades ?? [];
  const buckets = buildReturnBuckets(trades.map((trade) => Number(trade.return_pct ?? 0)).filter(Number.isFinite));
  return (
    <section className="panel distribution-panel">
      <div className="panel-head"><div><span className="kicker">Distribution</span><h2>Trade Return Distribution</h2><p>Lightweight histogram from trade returns.</p></div></div>
      <div className="histogram">
        {buckets.map((bucket) => <div key={bucket.label} style={{ height: `${Math.max(8, bucket.weight * 100)}%` }}><span>{bucket.count}</span><small>{bucket.label}</small></div>)}
      </div>
    </section>
  );
}

function ArtifactPicker({
  title = 'Select model from outputs/models',
  description,
  artifacts,
  selectedPath,
  isLoading,
  error,
  onRefresh,
  onSelect,
}: {
  title?: string;
  description?: string;
  artifacts: ModelArtifactItem[];
  selectedPath: string;
  isLoading: boolean;
  error: unknown;
  onRefresh: () => void;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const selectedArtifact = artifacts.find((artifact) => artifact.path === selectedPath);
  const errorMessage = error instanceof Error ? error.message : null;

  function selectArtifact(path: string) {
    onSelect(path);
    setOpen(false);
  }

  return (
    <section className="artifact-picker">
      <div className="artifact-picker-head">
        <div>
          <span className="kicker">Model Artifact</span>
          <h3>{title}</h3>
          {description && <p>{description}</p>}
        </div>

        <button type="button" onClick={onRefresh}>
          Refresh
        </button>
      </div>

      <button
        type="button"
        className={`artifact-select ${open ? 'open' : ''} ${selectedPath ? 'selected' : ''}`}
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className="artifact-select-icon">M</span>

        <span className="artifact-select-text">
          <b>{selectedArtifact?.name ?? 'Choose model artifact'}</b>
          <small>{selectedArtifact?.path ?? 'Click to open artifact list'}</small>
        </span>

        <span className="artifact-select-caret">⌄</span>
      </button>

      {open && (
        <div className="artifact-dropdown">
          {isLoading && (
            <div className="artifact-loading">
              <span />
              <p>Loading model artifacts…</p>
            </div>
          )}

          {errorMessage && (
            <div className="notice error">
              <strong>Could not load artifacts</strong>
              <p>{errorMessage}</p>
            </div>
          )}

          {!isLoading && !errorMessage && artifacts.length === 0 && (
            <div className="empty-state small">
              No model artifacts found in outputs/models.
            </div>
          )}

          {!isLoading && !errorMessage && artifacts.length > 0 && (
            <div className="artifact-list">
              {artifacts.map((artifact) => (
                <button
                  key={artifact.path}
                  type="button"
                  className={artifact.path === selectedPath ? 'active' : ''}
                  onClick={() => selectArtifact(artifact.path)}
                >
                  <span className="artifact-icon">M</span>

                  <span className="artifact-meta">
                    <b>{artifact.name}</b>
                    <small>{artifact.path}{artifact.artifact_role ? ` · ${artifact.artifact_role}` : ''}</small>
                  </span>

                  <em>{formatFileSize(artifact.size_bytes)}</em>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {selectedPath && (
        <div className="selected-artifact">
          <span>Selected artifact</span>
          <b>{selectedPath}</b>
        </div>
      )}
    </section>
  );
}

function filterArtifacts(artifacts: ModelArtifactItem[], role: string | null): ModelArtifactItem[] {
  if (!role) return artifacts;
  const exact = artifacts.filter((artifact) => artifact.artifact_role === role || artifact.path.includes(role));
  return exact.length > 0 ? exact : artifacts;
}

function formatFileSize(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB'];
  let size = value;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }

  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function RunProgress() {
  return (
    <div className="job-progress">
      <span>Submitting job → monitor starts automatically</span>
      <div><i /></div>
    </div>
  );
}

function formatDuration(seconds: unknown): string {
  const value = Number(seconds ?? 0);
  if (!Number.isFinite(value) || value <= 0) return '0s';
  const minutes = Math.floor(value / 60);
  const rest = Math.floor(value % 60);
  if (minutes <= 0) return `${rest}s`;
  const hours = Math.floor(minutes / 60);
  if (hours <= 0) return `${minutes}m ${rest}s`;
  return `${hours}h ${minutes % 60}m`;
}

function normalizeReport(report?: ReportResponse | null): ReportResponse | null {
  if (!report) return null;
  return {
    ...report,
    summary: report.summary ?? {},
    trades: Array.isArray(report.trades) ? report.trades : [],
    execution_log: Array.isArray(report.execution_log) ? report.execution_log : [],
    orders: Array.isArray(report.orders) ? report.orders : [],
  };
}

function normalizeChart(chart?: ChartDataResponse | null): ChartDataResponse | null {
  if (!chart) return null;
  return {
    candles: Array.isArray(chart.candles) ? chart.candles : [],
    equity_curve: Array.isArray(chart.equity_curve) ? chart.equity_curve : [],
    benchmark_curve: Array.isArray(chart.benchmark_curve) ? chart.benchmark_curve : [],
    markers: Array.isArray(chart.markers) ? chart.markers : [],
  };
}

function buildReturnBuckets(values: number[]) {
  if (values.length === 0) return [{ label: 'No data', count: 0, weight: 0.08 }];
  const labels = ['<-2%', '-2/-1%', '-1/0%', '0/1%', '1/2%', '>2%'];
  const counts = [0, 0, 0, 0, 0, 0];
  for (const value of values) {
    const pct = Math.abs(value) > 1.5 ? value / 100 : value;
    if (pct < -0.02) counts[0] += 1;
    else if (pct < -0.01) counts[1] += 1;
    else if (pct < 0) counts[2] += 1;
    else if (pct < 0.01) counts[3] += 1;
    else if (pct < 0.02) counts[4] += 1;
    else counts[5] += 1;
  }
  const max = Math.max(...counts, 1);
  return labels.map((label, index) => ({ label, count: counts[index], weight: counts[index] / max }));
}
