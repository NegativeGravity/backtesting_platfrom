import type { LiveEvent } from "../api/liveApi";

interface WorkerDiagnosticsPanelProps {
  events: LiveEvent[];
}

interface WorkerState {
  workerId: string;
  robotName: string;
  strategy: string;
  started: boolean;
  errors: number;
  holdSignals: number;
  longSignals: number;
  shortSignals: number;
  exitSignals: number;
  scheduledOrders: number;
  fills: number;
  cancelled: number;
  openPositions: number;
  closedPositions: number;
  latestSignal: string;
  latestReason: string;
  latestProbability: string;
}

export function WorkerDiagnosticsPanel({ events }: WorkerDiagnosticsPanelProps) {
  const workers = buildWorkerStates(events);

  return (
    <section className="card diagnostics-card">
      <div className="panel-header">
        <div>
          <h3>Strategy Worker Diagnostics</h3>
          <p>
            Shows worker state, long/short signals, fills, cancellations, and
            position ownership.
          </p>
        </div>
        <span className="status-pill">{workers.length} workers</span>
      </div>

      {workers.length === 0 ? (
        <div className="empty-table-state">
          <strong>No worker events yet</strong>
          <span>Start live replay to see worker diagnostics.</span>
        </div>
      ) : (
        <div className="worker-diagnostics-grid">
          {workers.map((worker) => (
            <article key={worker.workerId} className="worker-diagnostic-card">
              <div className="worker-diagnostic-top">
                <div>
                  <strong>{worker.workerId}</strong>
                  <span>{worker.robotName}</span>
                </div>
                <span className="strategy-chip">{worker.strategy}</span>
              </div>

              <div className="worker-status-row">
                <StatusPill label="Started" value={worker.started ? "YES" : "NO"} good={worker.started} />
                <StatusPill label="Errors" value={String(worker.errors)} bad={worker.errors > 0} />
                <StatusPill label="Fills" value={String(worker.fills)} good={worker.fills > 0} />
                <StatusPill label="Cancelled" value={String(worker.cancelled)} bad={worker.cancelled > 0} />
              </div>

              <div className="signal-summary-grid">
                <MiniMetric label="LONG" value={worker.longSignals} />
                <MiniMetric label="SHORT" value={worker.shortSignals} />
                <MiniMetric label="EXIT" value={worker.exitSignals} />
                <MiniMetric label="HOLD" value={worker.holdSignals} />
              </div>

              <div className="signal-summary-grid">
                <MiniMetric label="Scheduled" value={worker.scheduledOrders} />
                <MiniMetric label="Open Pos" value={worker.openPositions} />
                <MiniMetric label="Closed Pos" value={worker.closedPositions} />
                <MiniMetric label="Errors" value={worker.errors} />
              </div>

              <div className="latest-signal-box">
                <span>Latest Signal</span>
                <strong>{worker.latestSignal}</strong>
                <small>{worker.latestReason}</small>
                <small>Probability: {worker.latestProbability}</small>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function StatusPill({
  label,
  value,
  good,
  bad,
}: {
  label: string;
  value: string;
  good?: boolean;
  bad?: boolean;
}) {
  return (
    <div className={bad ? "worker-pill bad" : good ? "worker-pill good" : "worker-pill"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MiniMetric({ label, value }: { label: string; value: number }) {
  return (
    <div className="mini-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function buildWorkerStates(events: LiveEvent[]): WorkerState[] {
  const states = new Map<string, WorkerState>();

  for (const event of [...events].reverse()) {
    const payload = event.payload;
    const workerId = String(payload.worker_id ?? payload.bot_id ?? "");

    if (!workerId) {
      continue;
    }

    if (!states.has(workerId)) {
      states.set(workerId, {
        workerId,
        robotName: String(payload.robot_name ?? payload.robot_id ?? "-"),
        strategy: String(payload.strategy ?? "-"),
        started: false,
        errors: 0,
        holdSignals: 0,
        longSignals: 0,
        shortSignals: 0,
        exitSignals: 0,
        scheduledOrders: 0,
        fills: 0,
        cancelled: 0,
        openPositions: 0,
        closedPositions: 0,
        latestSignal: "-",
        latestReason: "-",
        latestProbability: "-",
      });
    }

    const state = states.get(workerId)!;

    if (event.event_type === "BOT_STARTED") {
      state.started = true;
      state.robotName = String(payload.robot_name ?? state.robotName);
      state.strategy = String(payload.strategy ?? state.strategy);
    }

    if (event.event_type === "BOT_ERROR") {
      state.errors += 1;
      state.latestReason = String(payload.error ?? "Worker error");
    }

    if (event.event_type === "SIGNAL_GENERATED") {
      const signalType = String(payload.signal_type ?? "-");
      const metadata = payload.metadata as Record<string, unknown> | undefined;

      state.latestSignal = signalType;
      state.latestReason = String(payload.reason ?? "-");

      const probability =
        metadata?.probability ??
        metadata?.long_probability ??
        metadata?.short_probability;

      state.latestProbability =
        typeof probability === "number" ? probability.toFixed(4) : "-";

      if (signalType === "HOLD") state.holdSignals += 1;
      if (signalType === "LONG") state.longSignals += 1;
      if (signalType === "SHORT") state.shortSignals += 1;
      if (signalType === "EXIT") state.exitSignals += 1;
    }

    if (event.event_type === "ORDER_SCHEDULED") {
      state.scheduledOrders += 1;
    }

    if (event.event_type === "ORDER_FILLED") {
      state.fills += 1;
    }

    if (
      event.event_type === "ORDER_CANCELLED" ||
      event.event_type === "ORDER_REJECTED"
    ) {
      state.cancelled += 1;
      state.latestReason = String(payload.reason ?? "Cancelled");
    }

    if (event.event_type === "POSITION_OPENED") {
      state.openPositions += 1;
    }

    if (event.event_type === "POSITION_CLOSED") {
      state.closedPositions += 1;
      state.openPositions = Math.max(state.openPositions - 1, 0);
    }
  }

  return [...states.values()].sort((left, right) =>
    left.workerId.localeCompare(right.workerId),
  );
}