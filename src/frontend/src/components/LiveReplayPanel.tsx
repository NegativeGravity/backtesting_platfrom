import { useMemo, useRef, useState } from "react";

import {
  createLiveReplaySocket,
  runRobotBacktest,
  startLiveReplay,
  type LiveEvent,
  type LiveRobotConfig,
  type LiveStrategyWorkerConfig,
} from "../api/liveApi";
import type { StrategyName } from "../types";
import { formatMoney, formatPercent } from "../utils/formatters";

interface LiveReplayPanelProps {
  modelArtifactPath: string;
  onCompleted: () => void | Promise<void>;
}

const STRATEGIES: Array<{ value: StrategyName; label: string; artifact: boolean }> = [
  { value: "mean_reversion", label: "Mean Reversion", artifact: false },
  { value: "adaptive_trend_breakout", label: "Adaptive Trend Breakout", artifact: false },
  { value: "liquidity_sweep_reversal", label: "Liquidity Sweep Reversal", artifact: false },
  { value: "ml_momentum", label: "ML Momentum", artifact: true },
  { value: "ml_regime_meta_label", label: "ML Regime Meta Label", artifact: true },
  { value: "dl_temporal_fusion_momentum", label: "DL Temporal Fusion Momentum", artifact: true },
];

export function LiveReplayPanel({ modelArtifactPath, onCompleted }: LiveReplayPanelProps) {
  const socketRef = useRef<WebSocket | null>(null);
  const pendingEventsRef = useRef<LiveEvent[]>([]);
  const animationFrameRef = useRef<number | null>(null);

  const [robotId, setRobotId] = useState("alpha_robot");
  const [displayName, setDisplayName] = useState("Alpha Purple Robot");
  const [workers, setWorkers] = useState<LiveStrategyWorkerConfig[]>([
    { worker_id: "trend_1", strategy: "adaptive_trend_breakout", model_artifact_path: null },
    { worker_id: "sweep_1", strategy: "liquidity_sweep_reversal", model_artifact_path: null },
  ]);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [delay, setDelay] = useState(0.02);
  const [error, setError] = useState<string | null>(null);

  const latestPortfolio = useMemo(
    () => events.find((event) => event.event_type === "PORTFOLIO_UPDATED"),
    [events],
  );

  const stats = useMemo(() => {
    let fills = 0;
    let signals = 0;
    let opens = 0;
    let closes = 0;

    for (const event of events) {
      if (event.event_type === "ORDER_FILLED") fills += 1;
      if (event.event_type === "SIGNAL_GENERATED") signals += 1;
      if (event.event_type === "POSITION_OPENED") opens += 1;
      if (event.event_type === "POSITION_CLOSED") closes += 1;
    }

    return { fills, signals, opens, closes };
  }, [events]);

  function flushEvents() {
    animationFrameRef.current = null;
    const pending = pendingEventsRef.current.splice(0);
    if (pending.length === 0) return;

    setEvents((current) => [...pending.reverse(), ...current].slice(0, 2500));
  }

  function enqueueEvent(event: LiveEvent) {
    pendingEventsRef.current.push(event);

    if (animationFrameRef.current === null) {
      animationFrameRef.current = window.requestAnimationFrame(flushEvents);
    }
  }

  function updateWorker(index: number, patch: Partial<LiveStrategyWorkerConfig>) {
    setWorkers((current) =>
      current.map((worker, workerIndex) =>
        workerIndex === index ? { ...worker, ...patch } : worker,
      ),
    );
  }

  function addWorker() {
    setWorkers((current) => [
      ...current,
      {
        worker_id: `worker_${current.length + 1}`,
        strategy: "adaptive_trend_breakout",
        model_artifact_path: null,
      },
    ]);
  }

  function removeWorker(index: number) {
    setWorkers((current) => current.filter((_, workerIndex) => workerIndex !== index));
  }

  function buildRobot(): LiveRobotConfig {
    const strategyWorkers = workers.map((worker) => {
      const requiresArtifact = STRATEGIES.some(
        (item) => item.value === worker.strategy && item.artifact,
      );

      return {
        ...worker,
        model_artifact_path: requiresArtifact
          ? worker.model_artifact_path || modelArtifactPath || null
          : null,
      };
    });

    return {
      robot_id: robotId.trim(),
      display_name: displayName.trim(),
      strategy_workers: strategyWorkers,
    };
  }

  async function handleStartLiveReplay() {
    setError(null);
    setEvents([]);
    setIsRunning(true);

    try {
      socketRef.current?.close();

      const response = await startLiveReplay({
        config_path: "configs/backtest.yaml",
        robots: [buildRobot()],
        replay_delay_seconds: delay,
      });

      const socket = createLiveReplaySocket(response.session_id);
      socketRef.current = socket;

      socket.onmessage = (message) => {
        const event = JSON.parse(message.data) as LiveEvent;
        enqueueEvent(event);

        if (
          event.event_type === "LIVE_SESSION_COMPLETED" ||
          event.event_type === "LIVE_SESSION_ERROR"
        ) {
          setIsRunning(false);
          void onCompleted();
        }
      };

      socket.onerror = () => {
        setError("Live replay websocket failed.");
        setIsRunning(false);
      };

      socket.onclose = () => {
        setIsRunning(false);
      };
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start live replay.");
      setIsRunning(false);
    }
  }

  async function handleRobotBacktest() {
    setError(null);
    setIsRunning(true);

    try {
      await runRobotBacktest({
        config_path: "configs/backtest.yaml",
        robot: buildRobot(),
      });
      await onCompleted();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run robot backtest.");
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <section className="live-grid">
      <section className="glass-card">
        <div className="panel-header">
          <div>
            <div className="section-kicker">Robot Builder</div>
            <h2>Multi-Strategy Robot</h2>
            <p>Define a robot with independent strategy workers.</p>
          </div>
        </div>

        <label className="field">
          <span>Robot ID</span>
          <input value={robotId} onChange={(event) => setRobotId(event.target.value)} />
        </label>

        <label className="field">
          <span>Display Name</span>
          <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
        </label>

        <label className="field">
          <span>Replay Delay Seconds</span>
          <input
            type="number"
            min="0"
            step="0.01"
            value={delay}
            onChange={(event) => setDelay(Number(event.target.value))}
          />
        </label>

        <div className="worker-list">
          {workers.map((worker, index) => (
            <div key={index} className="worker-card">
              <div className="worker-card-header">
                <strong>Worker {index + 1}</strong>
                <button type="button" className="toolbar-button" onClick={() => removeWorker(index)}>
                  Remove
                </button>
              </div>

              <label className="field">
                <span>Worker ID</span>
                <input
                  value={worker.worker_id ?? ""}
                  onChange={(event) => updateWorker(index, { worker_id: event.target.value })}
                />
              </label>

              <label className="field">
                <span>Strategy</span>
                <select
                  value={worker.strategy}
                  onChange={(event) =>
                    updateWorker(index, { strategy: event.target.value as StrategyName })
                  }
                >
                  {STRATEGIES.map((strategy) => (
                    <option key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Artifact Path</span>
                <input
                  value={worker.model_artifact_path ?? ""}
                  placeholder="Only for ML/DL workers"
                  onChange={(event) =>
                    updateWorker(index, { model_artifact_path: event.target.value || null })
                  }
                />
              </label>
            </div>
          ))}
        </div>

        <button type="button" className="toolbar-button" onClick={addWorker}>
          Add Worker
        </button>

        <div className="action-row">
          <button type="button" className="primary-action" onClick={handleStartLiveReplay} disabled={isRunning}>
            <span>{isRunning ? "Running..." : "Start Live Replay"}</span>
            <strong>▶</strong>
          </button>
          <button type="button" className="primary-action secondary" onClick={handleRobotBacktest} disabled={isRunning}>
            <span>Run Robot Backtest</span>
            <strong>↗</strong>
          </button>
        </div>

        {error && <div className="error-box">{error}</div>}
      </section>

      <section className="glass-card">
        <div className="panel-header">
          <div>
            <div className="section-kicker">Live Telemetry</div>
            <h2>Event Stream</h2>
            <p>Realtime robot, order, signal and portfolio events.</p>
          </div>
        </div>

        <section className="summary-grid live-summary-grid">
          <article className="summary-card">
            <span>Signals</span>
            <strong>{stats.signals}</strong>
          </article>
          <article className="summary-card">
            <span>Fills</span>
            <strong>{stats.fills}</strong>
          </article>
          <article className="summary-card">
            <span>Opened</span>
            <strong>{stats.opens}</strong>
          </article>
          <article className="summary-card">
            <span>Closed</span>
            <strong>{stats.closes}</strong>
          </article>
          <article className="summary-card">
            <span>Equity</span>
            <strong>{formatMoney(latestPortfolio?.payload?.equity)}</strong>
          </article>
          <article className="summary-card">
            <span>Return</span>
            <strong>{formatPercent(latestPortfolio?.payload?.total_return)}</strong>
          </article>
        </section>

        <div className="event-stream">
          {events.length === 0 ? (
            <div className="empty-state">No events yet.</div>
          ) : (
            events.slice(0, 160).map((event, index) => (
              <article key={`${event.event_id ?? index}-${event.timestamp}`} className="event-card">
                <div>
                  <strong>{event.event_type}</strong>
                  <span>{event.source}</span>
                </div>
                <small>{event.timestamp ?? ""}</small>
              </article>
            ))
          )}
        </div>
      </section>
    </section>
  );
}
