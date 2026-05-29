import type { ExecutionLogRecord } from '../types';
import { formatDateTime, formatMoney, formatNumber } from '../utils/formatters';

interface TradeBlotterProps {
  title?: string;
  records: ExecutionLogRecord[];
}

export function TradeBlotter({ title = 'Order Blotter', records }: TradeBlotterProps) {
  const sorted = [...records].sort((left, right) => {
    const leftTime = new Date(left.filled_bar_time ?? left.timestamp ?? 0).getTime() || 0;
    const rightTime = new Date(right.filled_bar_time ?? right.timestamp ?? 0).getTime() || 0;
    return rightTime - leftTime;
  });

  return (
    <section className="panel table-card">
      <div className="panel-head">
        <div>
          <span className="kicker">Execution analytics</span>
          <h2>{title}</h2>
          <p>Orders, fills, slippage, fees and engine reasons.</p>
        </div>
        <span className="badge">{sorted.length} records</span>
      </div>

      {sorted.length === 0 ? (
        <div className="empty-state">No execution records found in this report.</div>
      ) : (
        <div className="blotter-list">
          {sorted.slice(0, 500).map((record, index) => (
            <article key={`${record.timestamp ?? record.filled_bar_time ?? index}-${index}`} className="blotter-row">
              <div><b>{record.action ?? record.status ?? 'ORDER'}</b><small>{formatDateTime(record.filled_bar_time ?? record.timestamp)}</small></div>
              <div><b>{record.symbol ?? '-'}</b><small>{record.side ?? '-'}</small></div>
              <div><b>{record.worker_id ?? '-'}</b><small>{record.strategy ?? '-'}</small></div>
              <div><b>{formatNumber(record.quantity, 6)}</b><small>qty</small></div>
              <div><b>{formatMoney(record.fill_price)}</b><small>fill</small></div>
              <div><b>{formatMoney(record.fee)}</b><small>fee</small></div>
              <div><b>{formatMoney(record.slippage_cost)}</b><small>slippage</small></div>
              <div><b>{record.reason ?? '-'}</b><small>{record.robot_name ?? record.robot_id ?? ''}</small></div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}