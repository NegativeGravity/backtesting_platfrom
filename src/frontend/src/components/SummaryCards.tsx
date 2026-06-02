import { formatMoney, formatNumber, formatPercent, getRecord, toNumber } from '../utils/formatters';

interface SummaryCardsProps {
  summary: Record<string, unknown> | null;
}

const METRICS = [
  { key: 'final_equity', label: 'Final Equity', type: 'money', hint: 'Portfolio value at the end of this run.' },
  { key: 'total_return', label: 'Total Return', type: 'percent', hint: 'Run-period return, not necessarily annualized.' },
  { key: 'benchmark_return', label: 'Benchmark', type: 'percent', hint: 'Buy-and-hold or benchmark return for the same period.' },
  { key: 'excess_return', label: 'Excess', type: 'percent', hint: 'Strategy return minus benchmark return.' },
  { key: 'max_drawdown', label: 'Max DD', type: 'percent', hint: 'Worst peak-to-trough equity decline.' },
  { key: 'number_of_trades', label: 'Trades', type: 'number', hint: 'Closed trades in the report.' },
  { key: 'win_rate', label: 'Win Rate', type: 'percent', hint: 'Winning trades divided by all closed trades.' },
  { key: 'profit_factor', label: 'Profit Factor', type: 'number', hint: 'Gross wins divided by gross losses.' },
  { key: 'sharpe', label: 'Sharpe', type: 'number', hint: 'Risk-adjusted return. Confirm backend annualization/timeframe.' },
  { key: 'sortino', label: 'Sortino', type: 'number', hint: 'Downside-risk adjusted return. Confirm backend annualization/timeframe.' },
  { key: 'fees_paid', label: 'Fees', type: 'money', hint: 'Total commissions and exchange fees.' },
  { key: 'slippage_cost', label: 'Slippage', type: 'money', hint: 'Estimated execution slippage cost.' },
];

export function SummaryCards({ summary }: SummaryCardsProps) {
  const metrics = getRecord(getRecord(summary).metrics ?? summary ?? {});

  if (!summary) {
    return (
      <section className="summary-grid">
        <article className="metric-card wide">
          <span>No report selected</span>
          <strong>Select or run a backtest</strong>
          <small>Metrics, charts, trades and diagnostics will appear here.</small>
        </article>
      </section>
    );
  }

  return (
    <section className="summary-grid">
      {METRICS.map((metric) => {
        const raw = metrics[metric.key];
        const value = formatMetric(raw, metric.type);
        const numeric = toNumber(raw);
        const tone = metric.type === 'percent' || metric.key.includes('return') || metric.key === 'sharpe'
          ? numeric === null
            ? ''
            : numeric >= 0
              ? 'positive'
              : 'negative'
          : '';

        return (
          <article key={metric.key} className={`metric-card ${tone}`} title={metric.hint}>
            <span>{metric.label}</span>
            <strong>{value}</strong>
            <small>{metric.hint}</small>
          </article>
        );
      })}
    </section>
  );
}

function formatMetric(value: unknown, type: string): string {
  if (type === 'money') return formatMoney(value);
  if (type === 'percent') return formatPercent(value);
  if (type === 'number') return formatNumber(value, 3);
  return String(value ?? '-');
}
