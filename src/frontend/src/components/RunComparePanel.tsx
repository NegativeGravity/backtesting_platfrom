import { useQueries } from '@tanstack/react-query';
import type { BacktestRunListItem, ReportResponse } from '../types';
import { getReport } from '../api/api';
import { useUiStore } from '../stores/uiStore';
import { formatDate, formatMoney, formatPercent, getRecord } from '../utils/formatters';

interface RunComparePanelProps {
  runs: BacktestRunListItem[];
}

export function RunComparePanel({ runs }: RunComparePanelProps) {
  const selected = useUiStore((state) => state.compareRunIds);
  const toggle = useUiStore((state) => state.toggleCompareRun);
  const clear = useUiStore((state) => state.clearCompare);

  const reportQueries = useQueries({
    queries: selected.map((runId) => ({
      queryKey: ['report', runId],
      queryFn: () => getReport(runId),
      staleTime: 60_000,
    })),
  });

  return (
    <section className="panel compare-panel">
      <div className="panel-head">
        <div>
          <span className="kicker">Compare mode</span>
          <h2>Multi-run Comparison</h2>
          <p>Select up to four runs and compare return, drawdown, Sharpe and trade count side-by-side.</p>
        </div>
        {selected.length > 0 && <button type="button" onClick={clear}>Clear</button>}
      </div>

      <div className="compare-picker">
        {runs.slice(0, 24).map((run) => (
          <button key={run.run_id} type="button" className={selected.includes(run.run_id) ? 'active' : ''} onClick={() => toggle(run.run_id)}>
            <b>{run.run_id}</b>
            <small>{run.strategy ?? 'unknown'} · {formatDate(run.created_at)}</small>
            <small>Return {formatPercent(run.total_return)} · DD {formatPercent(run.max_drawdown)}</small>
          </button>
        ))}
      </div>

      {selected.length > 0 && (
        <div className="compare-grid">
          {selected.map((runId, index) => {
            const report = reportQueries[index]?.data as ReportResponse | undefined;
            const summary = getRecord(getRecord(report?.summary).metrics ?? report?.summary ?? {});
            const run = runs.find((item) => item.run_id === runId);
            return (
              <article key={runId} className="compare-card">
                <span>{run?.strategy ?? 'Run'}</span>
                <strong>{runId}</strong>
                <small>Final {formatMoney(summary.final_equity ?? run?.final_equity)}</small>
                <small>Return {formatPercent(summary.total_return ?? run?.total_return)}</small>
                <small>Max DD {formatPercent(summary.max_drawdown ?? run?.max_drawdown)}</small>
                <small>Sharpe {String(summary.sharpe ?? '-')}</small>
                <small>Trades {String(summary.number_of_trades ?? report?.trades?.length ?? '-')}</small>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}