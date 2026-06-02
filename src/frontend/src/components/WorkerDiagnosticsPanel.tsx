import type { LiveEvent, ReportResponse, TradeRecord } from '../types';
import { formatMoney, formatNumber, formatPercent, getRecord, toNumber } from '../utils/formatters';

interface WorkerDiagnosticsPanelProps {
  report?: ReportResponse | null;
  events?: LiveEvent[];
}

export function WorkerDiagnosticsPanel({ report, events = [] }: WorkerDiagnosticsPanelProps) {
  const trades = report?.trades ?? [];
  const byWorker = buildWorkerStats(trades, events);
  const diagnostics = getRecord(report?.diagnostics ?? getRecord(report?.summary).diagnostics);

  return (
    <section className="panel diagnostics-panel">
      <div className="panel-head">
        <div>
          <span className="kicker">Diagnostics</span>
          <h2>Worker & Engine Performance</h2>
          <p>Latency, fill quality, strategy attribution and per-worker PnL.</p>
        </div>
      </div>

      <div className="diag-grid">
        {byWorker.map((worker) => (
          <article key={worker.id} className="diag-card">
            <span>{worker.id}</span>
            <strong className={worker.pnl >= 0 ? 'pos' : 'neg'}>{formatMoney(worker.pnl)}</strong>
            <small>{worker.trades} trades · win {formatPercent(worker.winRate)}</small>
            <small>Signals {worker.signals} · Fills {worker.fills}</small>
          </article>
        ))}
        {byWorker.length === 0 && <div className="empty-state">No worker-level trade or event data available.</div>}
      </div>

      <div className="diag-raw">
        <h3>Raw diagnostics</h3>
        <pre>{JSON.stringify(diagnostics, null, 2)}</pre>
      </div>
    </section>
  );
}

function buildWorkerStats(trades: TradeRecord[], events: LiveEvent[]) {
  const map = new Map<string, { id: string; trades: number; wins: number; pnl: number; signals: number; fills: number }>();

  function ensure(id: string) {
    if (!map.has(id)) map.set(id, { id, trades: 0, wins: 0, pnl: 0, signals: 0, fills: 0 });
    return map.get(id)!;
  }

  for (const trade of trades) {
    const id = trade.worker_id ?? trade.strategy ?? 'unknown';
    const stats = ensure(id);
    const pnl = toNumber(trade.net_pnl) ?? 0;
    stats.trades += 1;
    stats.pnl += pnl;
    if (pnl > 0) stats.wins += 1;
  }

  for (const event of events) {
    const id = String(event.payload.worker_id ?? event.source ?? 'unknown');
    const stats = ensure(id);
    if (event.event_type === 'SIGNAL_GENERATED') stats.signals += 1;
    if (event.event_type === 'ORDER_FILLED') stats.fills += 1;
  }

  return [...map.values()]
    .map((item) => ({ ...item, winRate: item.trades > 0 ? item.wins / item.trades : null as number | null }))
    .sort((left, right) => right.pnl - left.pnl);
}
