import { useEffect, useMemo, useState } from "react";

import { getChartData, getReport, listBacktests, runBacktest } from "./api/api";
import { ChartPanel } from "./components/ChartPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { LiveReplayPanel } from "./components/LiveReplayPanel";
import { SummaryCards } from "./components/SummaryCards";
import { TradesTable } from "./components/TradesTable";
import type {
  BacktestRunListItem,
  ChartDataResponse,
  ReportResponse,
  StrategyName,
} from "./types";
import { formatPercent } from "./utils/formatters";

const STRATEGY_OPTIONS: Array<{
  value: StrategyName;
  label: string;
  badge: string;
  description: string;
  requiresArtifact: boolean;
}> = [
  {
    value: "mean_reversion",
    label: "Mean Reversion",
    badge: "Classic",
    description: "Z-score reversal around rolling mean.",
    requiresArtifact: false,
  },
  {
    value: "adaptive_trend_breakout",
    label: "Adaptive Trend Breakout",
    badge: "Trend",
    description: "EMA regime + Donchian breakout + ATR expansion.",
    requiresArtifact: false,
  },
  {
    value: "liquidity_sweep_reversal",
    label: "Liquidity Sweep Reversal",
    badge: "Sweep",
    description: "False-breakout reversal with wick and volume confirmation.",
    requiresArtifact: false,
  },
  {
    value: "ml_momentum",
    label: "ML Momentum",
    badge: "ML",
    description: "Causal feature classifier with probability thresholds.",
    requiresArtifact: true,
  },
  {
    value: "ml_regime_meta_label",
    label: "ML Regime Meta Label",
    badge: "Meta",
    description: "Regime-aware meta-labeling model.",
    requiresArtifact: true,
  },
  {
    value: "dl_temporal_fusion_momentum",
    label: "DL Temporal Fusion Momentum",
    badge: "DL",
    description: "TorchScript sequence model for temporal edge detection.",
    requiresArtifact: true,
  },
];

function strategyRequiresArtifact(strategy: StrategyName): boolean {
  return STRATEGY_OPTIONS.some(
    (item) => item.value === strategy && item.requiresArtifact,
  );
}

function normalizeReport(report: ReportResponse | null): ReportResponse | null {
  if (!report) return null;

  return {
    ...report,
    summary: report.summary ?? {},
    trades: Array.isArray(report.trades) ? report.trades : [],
    execution_log: Array.isArray(report.execution_log) ? report.execution_log : [],
    equity_curve: Array.isArray(report.equity_curve) ? report.equity_curve : [],
    benchmark_curve: Array.isArray(report.benchmark_curve) ? report.benchmark_curve : [],
  };
}

function normalizeChartData(chartData: ChartDataResponse | null): ChartDataResponse | null {
  if (!chartData) return null;

  return {
    candles: Array.isArray(chartData.candles) ? chartData.candles : [],
    equity_curve: Array.isArray(chartData.equity_curve) ? chartData.equity_curve : [],
    benchmark_curve: Array.isArray(chartData.benchmark_curve) ? chartData.benchmark_curve : [],
    markers: Array.isArray(chartData.markers) ? chartData.markers : [],
  };
}

export default function App() {
  const [runs, setRuns] = useState<BacktestRunListItem[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [chartData, setChartData] = useState<ChartDataResponse | null>(null);
  const [strategy, setStrategy] = useState<StrategyName>("adaptive_trend_breakout");
  const [modelArtifactPath, setModelArtifactPath] = useState("");
  const [activeTab, setActiveTab] = useState<"live" | "reports">("reports");
  const [isRunning, setIsRunning] = useState(false);
  const [isLoadingRun, setIsLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const currentStrategy = useMemo(
    () => STRATEGY_OPTIONS.find((item) => item.value === strategy) ?? STRATEGY_OPTIONS[0],
    [strategy],
  );

  const aggregateStats = useMemo(() => {
    const totalTrades = report?.trades?.length ?? 0;
    const winningTrades =
      report?.trades?.filter((trade) => typeof trade.net_pnl === "number" && trade.net_pnl > 0)
        .length ?? 0;
    const openRun = runs.find((run) => run.run_id === selectedRunId);

    return {
      totalTrades,
      winningTrades,
      winRate: totalTrades > 0 ? winningTrades / totalTrades : null,
      selectedReturn: openRun?.total_return,
      selectedDrawdown: openRun?.max_drawdown,
    };
  }, [report, runs, selectedRunId]);

  async function refreshRuns() {
    const data = await listBacktests();
    setRuns(Array.isArray(data) ? data : []);
  }

  async function loadRun(runId: string) {
    setSelectedRunId(runId);
    setIsLoadingRun(true);
    setError(null);

    try {
      const [reportData, chartResponse] = await Promise.all([
        getReport(runId),
        getChartData(runId),
      ]);

      setReport(normalizeReport(reportData));
      setChartData(normalizeChartData(chartResponse));
      setActiveTab("reports");
    } catch (err) {
      setReport(null);
      setChartData(null);
      setSelectedRunId(null);

      const message =
        err instanceof Error
          ? err.message
          : `Run not found or report files are invalid: ${runId}`;

      setError(message);
      await refreshRuns();
    } finally {
      setIsLoadingRun(false);
    }
  }

  async function handleRunBacktest() {
    setIsRunning(true);
    setError(null);

    try {
      const requiresArtifact = strategyRequiresArtifact(strategy);
      const artifactPath = modelArtifactPath.trim();

      if (requiresArtifact && artifactPath.length === 0) {
        throw new Error(`${strategy} requires a model artifact path.`);
      }

      const response = await runBacktest({
        strategy,
        config_path: "configs/backtest.yaml",
        model_artifact_path: requiresArtifact ? artifactPath : null,
      });

      await refreshRuns();
      await loadRun(response.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsRunning(false);
    }
  }

  useEffect(() => {
    refreshRuns().catch((err) => {
      setError(err instanceof Error ? err.message : "Failed to load runs");
    });
  }, []);

  return (
    <main className="app-shell">
      <div className="aurora-bg" />

      <header className="terminal-header">
        <div className="brand-block">
          <div className="brand-mark">
            <span>Q</span>
          </div>
          <div>
            <div className="eyebrow">Quant Research Terminal</div>
            <h1>Trading System Control Center</h1>
            <p>
              Robot backtesting, TradingView-style replay, ML/DL artifacts,
              execution analytics, and strategy-level trade attribution.
            </p>
          </div>
        </div>

        <div className="header-metrics">
          <div className="header-metric">
            <span>Mode</span>
            <strong>{activeTab === "live" ? "Live Replay" : "Reports"}</strong>
          </div>
          <div className="header-metric">
            <span>Runs</span>
            <strong>{runs.length}</strong>
          </div>
          <div className="header-metric">
            <span>Status</span>
            <strong>{isRunning ? "Running" : isLoadingRun ? "Loading" : "Ready"}</strong>
          </div>
          <div className="header-metric highlight">
            <span>Trades</span>
            <strong>{aggregateStats.totalTrades}</strong>
          </div>
        </div>
      </header>

      <nav className="terminal-tabs">
        <button
          className={activeTab === "reports" ? "terminal-tab active" : "terminal-tab"}
          onClick={() => setActiveTab("reports")}
          type="button"
        >
          Reports & Replay
        </button>
        <button
          className={activeTab === "live" ? "terminal-tab active" : "terminal-tab"}
          onClick={() => setActiveTab("live")}
          type="button"
        >
          Live Robot Lab
        </button>
      </nav>

      <section className="terminal-layout">
        <aside className="control-rail">
          <section className="glass-card hero-control-card">
            <div className="panel-header">
              <div>
                <div className="section-kicker">Strategy Backtest</div>
                <h2>Run Single Strategy</h2>
                <p>Runs against <code>POST /backtests/run</code>.</p>
              </div>
              <span className="pulse-dot" />
            </div>

            <div className="strategy-picker">
              {STRATEGY_OPTIONS.map((item) => (
                <button
                  key={item.value}
                  className={item.value === strategy ? "strategy-chip active" : "strategy-chip"}
                  onClick={() => setStrategy(item.value)}
                  disabled={isRunning}
                  type="button"
                >
                  <span className="strategy-chip-top">
                    <strong>{item.label}</strong>
                    <em>{item.badge}</em>
                  </span>
                  <small>{item.description}</small>
                </button>
              ))}
            </div>

            <label className="field">
              <span>Model Artifact Path</span>
              <input
                value={modelArtifactPath}
                onChange={(event) => setModelArtifactPath(event.target.value)}
                placeholder={
                  currentStrategy.requiresArtifact
                    ? "outputs/models/model_or_artifact_path"
                    : "Only required for ML/DL strategies"
                }
                disabled={isRunning}
              />
            </label>

            {currentStrategy.requiresArtifact && (
              <div className="info-box">
                {currentStrategy.label} needs a valid artifact path before running.
              </div>
            )}

            <button
              className="primary-action"
              onClick={handleRunBacktest}
              disabled={isRunning}
              type="button"
            >
              <span>{isRunning ? "Running Backtest..." : "Launch Backtest"}</span>
              <strong>↗</strong>
            </button>

            {error && <div className="error-box">{error}</div>}
          </section>

          <section className="glass-card">
            <div className="panel-header">
              <div>
                <div className="section-kicker">Research Archive</div>
                <h2>Saved Runs</h2>
                <p>Click any run to load report, chart, trades and replay.</p>
              </div>
            </div>

            <div className="run-list">
              {runs.length === 0 ? (
                <div className="empty-state">No backtest runs found yet.</div>
              ) : (
                runs.map((run) => (
                  <button
                    key={run.run_id}
                    className={run.run_id === selectedRunId ? "run-item active" : "run-item"}
                    onClick={() => loadRun(run.run_id)}
                    type="button"
                  >
                    <span className="run-title-row">
                      <strong>{run.run_id}</strong>
                      <em>{run.strategy ?? "unknown"}</em>
                    </span>
                    <span className="run-mini-metrics">
                      <small>Return {formatPercent(run.total_return)}</small>
                      <small>DD {formatPercent(run.max_drawdown)}</small>
                      <small>BM {formatPercent(run.benchmark_return)}</small>
                    </span>
                  </button>
                ))
              )}
            </div>
          </section>
        </aside>

        <section className="terminal-content">
          {activeTab === "live" ? (
            <ErrorBoundary fallbackTitle="Live replay panel failed">
              <LiveReplayPanel modelArtifactPath={modelArtifactPath} onCompleted={refreshRuns} />
            </ErrorBoundary>
          ) : (
            <ErrorBoundary fallbackTitle="Report panel failed">
              <SummaryCards summary={report?.summary ?? null} />
              <ChartPanel chartData={chartData} trades={report?.trades ?? []} />
              <TradesTable trades={report?.trades ?? []} />
            </ErrorBoundary>
          )}
        </section>
      </section>
    </main>
  );
}
