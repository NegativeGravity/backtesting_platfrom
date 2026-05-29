import type { ReportResponse, TradeRecord } from '../types';
import { formatDateTime } from './formatters';

export function downloadTextFile(filename: string, content: string, mime = 'text/plain;charset=utf-8'): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function tradesToCsv(trades: TradeRecord[]): string {
  const columns: Array<keyof TradeRecord> = [
    'trade_id',
    'symbol',
    'strategy',
    'worker_id',
    'side',
    'entry_time',
    'exit_time',
    'entry_price',
    'exit_price',
    'quantity',
    'net_pnl',
    'return_pct',
    'exit_reason',
  ];
  const rows = trades.map((trade) => columns.map((column) => escapeCsv(trade[column])).join(','));
  return [columns.join(','), ...rows].join('\n');
}

export function reportToHtml(runId: string, report: ReportResponse): string {
  const summaryRows = Object.entries(report.summary ?? {})
    .map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(String(value ?? '-'))}</td></tr>`)
    .join('');

  const tradeRows = (report.trades ?? []).slice(0, 1000).map((trade) => `
    <tr>
      <td>${escapeHtml(trade.symbol ?? '-')}</td>
      <td>${escapeHtml(trade.side ?? '-')}</td>
      <td>${escapeHtml(trade.strategy ?? '-')}</td>
      <td>${escapeHtml(formatDateTime(trade.entry_time))}</td>
      <td>${escapeHtml(formatDateTime(trade.exit_time))}</td>
      <td>${escapeHtml(String(trade.net_pnl ?? '-'))}</td>
    </tr>`).join('');

  return `<!doctype html>
<html><head><meta charset="utf-8"><title>${escapeHtml(runId)} Report</title>
<style>body{font-family:Inter,system-ui;background:#0b1220;color:#e5edf8;padding:32px}table{border-collapse:collapse;width:100%;margin:20px 0;background:#111827}td,th{border:1px solid #25324a;padding:8px;text-align:left}th{color:#8bd3ff}</style>
</head><body><h1>${escapeHtml(runId)}</h1><p>Generated ${escapeHtml(new Date().toISOString())}</p><h2>Summary</h2><table>${summaryRows}</table><h2>Trades</h2><table><thead><tr><th>Symbol</th><th>Side</th><th>Strategy</th><th>Entry</th><th>Exit</th><th>Net PnL</th></tr></thead><tbody>${tradeRows}</tbody></table></body></html>`;
}

function escapeCsv(value: unknown): string {
  const text = String(value ?? '');
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function escapeHtml(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}