import { formatMoney, formatPercent, formatNumber } from "../utils/formatters";

interface SummaryCardsProps {
  summary: Record<string, unknown> | null;
}

const METRICS = [
  { key: "final_equity", label: "Final Equity", type: "money" },
  { key: "total_return", label: "Total Return", type: "percent" },
  { key: "benchmark_return", label: "Benchmark", type: "percent" },
  { key: "excess_return", label: "Excess", type: "percent" },
  { key: "max_drawdown", label: "Max DD", type: "percent" },
  { key: "number_of_trades", label: "Trades", type: "number" },
  { key: "win_rate", label: "Win Rate", type: "percent" },
  { key: "profit_factor", label: "Profit Factor", type: "number" },
  { key: "sharpe", label: "Sharpe", type: "number" },
  { key: "sortino", label: "Sortino", type: "number" },
  { key: "fees_paid", label: "Fees", type: "money" },
  { key: "slippage_cost", label: "Slippage", type: "money" },
];

export function SummaryCards({ summary }: SummaryCardsProps) {
  const metrics = (summary?.metrics ?? summary ?? {}) as Record<string, unknown>;

  if (!summary) {
    return (
      <section className="summary-grid">
        <div className="summary-card summary-card-wide">
          <span>No report selected</span>
          <strong>Run or select a backtest</strong>
          <p>Performance metrics will appear here after loading a report.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="summary-grid">
      {METRICS.map((metric) => (
        <article key={metric.key} className="summary-card">
          <span>{metric.label}</span>
          <strong>{formatMetric(metrics[metric.key], metric.type)}</strong>
          <small>{metric.key.replaceAll("_", " ")}</small>
        </article>
      ))}
    </section>
  );
}

function formatMetric(value: unknown, type: string): string {
  if (type === "money") return formatMoney(value);
  if (type === "percent") return formatPercent(value);
  if (type === "number") return formatNumber(value, 3);
  return String(value ?? "-");
}
