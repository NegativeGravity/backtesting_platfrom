import { useEffect, useMemo, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { createLiveReplaySocket, runLiveRobotBacktest, startLiveReplay, waitForBacktestJob } from '../api/liveApi';
import type { LiveEvent, LiveRobotConfig, LiveStrategyWorkerConfig, StrategyName } from '../types';
import { formatMoney, formatPercent } from '../utils/formatters';
import { LiveTradingChart } from './LiveTradingChart';
import { LiveTradesTable } from './LiveTradesTable';
import { StatusPill } from './StatusPill';
import { VirtualEventStream } from './VirtualEventStream';
import { WorkerDiagnosticsPanel } from './WorkerDiagnosticsPanel';

interface LiveReplayPanelProps {
  modelArtifactPath: string;
  onCompleted: () => void | Promise<void>;
}

const STRATEGIES: Array<{ value: StrategyName; label: string; artifact: boolean }> = [
  { value: 'mean_reversion', label: 'Mean Reversion', artifact: false },
  { value: 'adaptive_trend_breakout', label: 'Adaptive Trend Breakout', artifact: false },
  { value: 'liquidity_sweep_reversal', label: 'Liquidity Sweep Reversal', artifact: false },
  { value: 'ml_momentum', label: 'ML Momentum', artifact: true },
  { value: 'ml_regime_meta_label', label: 'ML Regime Meta Label', artifact: true },
  { value: 'dl_temporal_fusion_momentum', label: 'DL Temporal Fusion Momentum', artifact: true },
];

type SocketStatus = 'idle' | 'connecting' | 'live' | 'paused' | 'closed' | 'error';

export function LiveReplayPanel({ modelArtifactPath, onCompleted }: LiveReplayPanelProps) {
  const socketRef = useRef<WebSocket | null>(null);
  const pendingRef = useRef<LiveEvent[]>([]);
  const frameRef = useRef<number | null>(null);
  const manualCloseRef = useRef(false);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [websocketUrl, setWebsocketUrl] = useState<string | null>(null);
  const [robotId, setRobotId] = useState('alpha_robot');
  const [displayName, setDisplayName] = useState('Alpha Robot');
  const [delay, setDelay] = useState(0.02);
  const [workers, setWorkers] = useState<LiveStrategyWorkerConfig[]>([
    { worker_id: 'trend_1', strategy: 'adaptive_trend_breakout', model_artifact_path: null },
    { worker_id: 'sweep_1', strategy: 'liquidity_sweep_reversal', model_artifact_path: null },
  ]);
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [status, setStatus] = useState<SocketStatus>('idle');
  const [paused, setPaused] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const stats = useMemo(() => buildStats(events), [events]);
  const latestPortfolio = useMemo(
    () => events.find((event) => event.event_type === 'PORTFOLIO_UPDATED'),
    [events],
  );

  function flushEvents() {
    frameRef.current = null;

    if (paused) return;

    const pending = pendingRef.current.splice(0);
    if (pending.length === 0) return;

    setEvents((current) => [...pending.reverse(), ...current].slice(0, 6000));
  }

  function enqueue(event: LiveEvent) {
    pendingRef.current.push(event);

    if (pendingRef.current.length > 6000) {
      pendingRef.current.splice(0, pendingRef.current.length - 6000);
    }

    if (frameRef.current === null && !paused) {
      frameRef.current = window.requestAnimationFrame(flushEvents);
    }
  }

  function connect(nextWebsocketUrl: string) {
    manualCloseRef.current = false;
    socketRef.current?.close();

    setStatus('connecting');
    setError(null);

    const socket = createLiveReplaySocket(nextWebsocketUrl);
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus(paused ? 'paused' : 'live');
    };

    socket.onmessage = (message) => {
      try {
        const event = JSON.parse(message.data) as LiveEvent;
        enqueue(event);

        if (
          event.event_type === 'LIVE_SESSION_COMPLETED' ||
          event.event_type === 'LIVE_SESSION_ERROR' ||
          event.event_type === 'LIVE_SESSION_STOPPED'
        ) {
          setStatus(event.event_type === 'LIVE_SESSION_ERROR' ? 'error' : 'closed');
          void onCompleted();
        }
      } catch (parseError) {
        console.error(parseError);
      }
    };

    socket.onerror = () => {
      setStatus('error');
      setError(`WebSocket connection failed: ${nextWebsocketUrl}`);
    };

    socket.onclose = () => {
      if (!manualCloseRef.current) {
        setStatus((current) => (current === 'error' ? 'error' : 'closed'));
      }
    };
  }

  const startMutation = useMutation({
    mutationFn: () => startLiveReplay({
      config_path: 'configs/backtest.yaml',
      robots: [buildRobot()],
      replay_delay_seconds: delay,
    }),
    onMutate: () => {
      setError(null);
      setEvents([]);
      pendingRef.current = [];
      setStatus('connecting');
    },
    onSuccess: (response) => {
      const nextWebsocketUrl = response.websocket_url ?? `/ws/live-replay/${response.session_id}`;

      setSessionId(response.session_id);
      setWebsocketUrl(nextWebsocketUrl);
      connect(nextWebsocketUrl);
    },
    onError: (mutationError) => {
      setStatus('error');
      setError(mutationError instanceof Error ? mutationError.message : 'Failed to start live replay.');
    },
  });

  const robotBacktestMutation = useMutation({
    mutationFn: async () => {
      const response = await runLiveRobotBacktest({
        config_path: 'configs/backtest.yaml',
        robot: buildRobot(),
      });

      return response.run_id || !response.job_id ? response : waitForBacktestJob(response.job_id);
    },
    onSuccess: async () => {
      await onCompleted();
    },
    onError: (mutationError) => {
      setError(mutationError instanceof Error ? mutationError.message : 'Failed to run robot backtest.');
    },
  });

  useEffect(() => {
    return () => {
      manualCloseRef.current = true;
      socketRef.current?.close();

      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!paused && pendingRef.current.length > 0 && frameRef.current === null) {
      frameRef.current = window.requestAnimationFrame(flushEvents);
    }

    setStatus((current) => {
      if (paused && current === 'live') return 'paused';
      if (!paused && current === 'paused') return 'live';
      return current;
    });
  }, [paused]);

  function updateWorker(index: number, patch: Partial<LiveStrategyWorkerConfig>) {
    setWorkers((current) => current.map((worker, workerIndex) => (
      workerIndex === index ? { ...worker, ...patch } : worker
    )));
  }

  function addWorker() {
    setWorkers((current) => [
      ...current,
      {
        worker_id: `worker_${current.length + 1}`,
        strategy: 'adaptive_trend_breakout',
        model_artifact_path: null,
      },
    ]);
  }

  function removeWorker(index: number) {
    setWorkers((current) => current.filter((_, workerIndex) => workerIndex !== index));
  }

  function buildRobot(): LiveRobotConfig {
    return {
      robot_id: robotId.trim() || 'alpha_robot',
      display_name: displayName.trim() || 'Alpha Robot',
      strategy_workers: workers.map((worker) => {
        const requiresArtifact = STRATEGIES.some((strategy) => (
          strategy.value === worker.strategy && strategy.artifact
        ));

        return {
          ...worker,
          worker_id: worker.worker_id?.trim() || null,
          model_artifact_path: requiresArtifact
            ? worker.model_artifact_path || modelArtifactPath || null
            : null,
        };
      }),
    };
  }

  function stopSocket() {
    manualCloseRef.current = true;
    socketRef.current?.close();
    setStatus('closed');
  }

  return (
    <section className="live-layout">
      <aside className="panel live-builder">
        <div className="panel-head">
          <div>
            <span className="kicker">Robot builder</span>
            <h2>Live Replay</h2>
            <p>Multi-worker robot with websocket telemetry.</p>
          </div>

          <StatusPill
            status={status}
            tone={
              status === 'live'
                ? 'live'
                : status === 'error'
                  ? 'bad'
                  : status === 'paused'
                    ? 'warn'
                    : 'neutral'
            }
          />
        </div>

        <label>
          Robot ID
          <input value={robotId} onChange={(event) => setRobotId(event.target.value)} />
        </label>

        <label>
          Display Name
          <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
        </label>

        <label>
          Replay Delay Seconds
          <input
            type="number"
            min="0"
            step="0.01"
            value={delay}
            onChange={(event) => setDelay(Number(event.target.value))}
          />
        </label>

        <div className="worker-stack">
          {workers.map((worker, index) => (
            <article className="worker-card" key={`${worker.worker_id}-${index}`}>
              <header>
                <b>Worker {index + 1}</b>
                <button type="button" onClick={() => removeWorker(index)}>
                  Remove
                </button>
              </header>

              <label>
                Worker ID
                <input
                  value={worker.worker_id ?? ''}
                  onChange={(event) => updateWorker(index, { worker_id: event.target.value })}
                />
              </label>

              <label>
                Strategy
                <select
                  value={worker.strategy}
                  onChange={(event) => updateWorker(index, { strategy: event.target.value as StrategyName })}
                >
                  {STRATEGIES.map((strategy) => (
                    <option key={strategy.value} value={strategy.value}>
                      {strategy.label}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Artifact
                <input
                  value={worker.model_artifact_path ?? ''}
                  placeholder="Only for ML/DL"
                  onChange={(event) => updateWorker(index, {
                    model_artifact_path: event.target.value || null,
                  })}
                />
              </label>
            </article>
          ))}
        </div>

        <button type="button" onClick={addWorker}>
          Add Worker
        </button>

        <div className="action-row">
          <button
            type="button"
            className="primary"
            disabled={startMutation.isPending}
            onClick={() => startMutation.mutate()}
          >
            {startMutation.isPending ? 'Starting…' : 'Start Live'}
          </button>

          <button type="button" onClick={() => setPaused((value) => !value)}>
            {paused ? 'Resume UI' : 'Pause UI'}
          </button>

          <button type="button" onClick={stopSocket}>
            Stop
          </button>

          {websocketUrl && (
            <button type="button" onClick={() => connect(websocketUrl)}>
              Reconnect
            </button>
          )}

          <button
            type="button"
            className="secondary"
            disabled={robotBacktestMutation.isPending}
            onClick={() => robotBacktestMutation.mutate()}
          >
            {robotBacktestMutation.isPending ? 'Running…' : 'Robot Backtest'}
          </button>
        </div>

        {sessionId && (
          <div className="selected-artifact">
            <span>Live session</span>
            <b>{sessionId}</b>
          </div>
        )}

        {websocketUrl && (
          <div className="selected-artifact">
            <span>WebSocket URL</span>
            <b>{websocketUrl}</b>
          </div>
        )}

        {error && (
          <div className="notice error">
            <strong>Live replay error</strong>
            <p>{error}</p>
          </div>
        )}
      </aside>

      <section className="live-main">
        <section className="summary-grid live-summary-grid">
          <article className="metric-card">
            <span>Signals</span>
            <strong>{stats.signals}</strong>
            <small>Generated strategy signals</small>
          </article>

          <article className="metric-card">
            <span>Fills</span>
            <strong>{stats.fills}</strong>
            <small>Filled orders</small>
          </article>

          <article className="metric-card">
            <span>Opened</span>
            <strong>{stats.opens}</strong>
            <small>Position opened events</small>
          </article>

          <article className="metric-card">
            <span>Closed</span>
            <strong>{stats.closes}</strong>
            <small>Position closed events</small>
          </article>

          <article className="metric-card">
            <span>Equity</span>
            <strong>{formatMoney(latestPortfolio?.payload.equity)}</strong>
            <small>Latest portfolio update</small>
          </article>

          <article className="metric-card">
            <span>Return</span>
            <strong>{formatPercent(latestPortfolio?.payload.total_return)}</strong>
            <small>Latest total return</small>
          </article>
        </section>

        <LiveTradingChart events={events} />
        <LiveTradesTable events={events} />
        <WorkerDiagnosticsPanel events={events} />

        <section className="panel">
          <div className="panel-head">
            <div>
              <span className="kicker">Telemetry</span>
              <h2>Virtualized Event Stream</h2>
            </div>

            <span className="badge">{events.length} events</span>
          </div>

          <VirtualEventStream events={events} />
        </section>
      </section>
    </section>
  );
}

function buildStats(events: LiveEvent[]) {
  return events.reduce((stats, event) => ({
    signals: stats.signals + (event.event_type === 'SIGNAL_GENERATED' ? 1 : 0),
    fills: stats.fills + (event.event_type === 'ORDER_FILLED' ? 1 : 0),
    opens: stats.opens + (event.event_type === 'POSITION_OPENED' ? 1 : 0),
    closes: stats.closes + (event.event_type === 'POSITION_CLOSED' ? 1 : 0),
  }), {
    signals: 0,
    fills: 0,
    opens: 0,
    closes: 0,
  });
}