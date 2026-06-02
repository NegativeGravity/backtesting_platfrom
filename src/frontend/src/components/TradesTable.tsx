import { useMemo, useRef, useState } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import type { TradeRecord } from '../types';
import { downloadTextFile, tradesToCsv } from '../utils/exports';
import { formatDateTime, formatMoney, formatNumber, formatPercent, toNumber } from '../utils/formatters';

interface TradesTableProps {
  trades: TradeRecord[];
}

export function TradesTable({ trades }: TradesTableProps) {
  const [query, setQuery] = useState('');
  const [side, setSide] = useState('all');
  const [worker, setWorker] = useState('all');
  const [reason, setReason] = useState('all');
  const [sort, setSort] = useState('newest');
  const parentRef = useRef<HTMLDivElement | null>(null);

  const workers = useMemo(() => unique(trades.map((trade) => trade.worker_id ?? trade.strategy ?? 'unknown')), [trades]);
  const reasons = useMemo(() => unique(trades.map((trade) => trade.exit_reason ?? 'unknown')), [trades]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return trades
      .filter((trade) => {
        const haystack = [trade.symbol, trade.strategy, trade.worker_id, trade.side, trade.exit_reason, trade.trade_id, trade.position_id]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        const matchesQuery = !needle || haystack.includes(needle);
        const matchesSide = side === 'all' || String(trade.side ?? '').toLowerCase().includes(side.toLowerCase());
        const matchesWorker = worker === 'all' || (trade.worker_id ?? trade.strategy ?? 'unknown') === worker;
        const matchesReason = reason === 'all' || (trade.exit_reason ?? 'unknown') === reason;
        return matchesQuery && matchesSide && matchesWorker && matchesReason;
      })
      .sort((left, right) => compareTrades(left, right, sort));
  }, [trades, query, side, worker, reason, sort]);

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 68,
    overscan: 12,
  });

  const totalPnl = filtered.reduce((sum, trade) => sum + (toNumber(trade.net_pnl) ?? 0), 0);

  return (
    <section className="panel table-card">
      <div className="panel-head">
        <div>
          <span className="kicker">Trade ledger</span>
          <h2>Executed Positions</h2>
          <p>Search and filter by worker, strategy, side and exit reason.</p>
        </div>
        <div className="table-actions">
          <span className={totalPnl >= 0 ? 'badge good' : 'badge bad'}>{formatMoney(totalPnl)} filtered PnL</span>
          <button type="button" onClick={() => downloadTextFile('trades.csv', tradesToCsv(filtered), 'text/csv;charset=utf-8')}>Export CSV</button>
        </div>
      </div>

      <div className="filters">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search symbol, worker, reason…" />
        <select value={side} onChange={(event) => setSide(event.target.value)}>
          <option value="all">All sides</option>
          <option value="long">Long / Buy</option>
          <option value="short">Short / Sell</option>
        </select>
        <select value={worker} onChange={(event) => setWorker(event.target.value)}>
          <option value="all">All workers</option>
          {workers.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <select value={reason} onChange={(event) => setReason(event.target.value)}>
          <option value="all">All reasons</option>
          {reasons.map((item) => <option key={item} value={item}>{item}</option>)}
        </select>
        <select value={sort} onChange={(event) => setSort(event.target.value)}>
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
          <option value="best">Best PnL</option>
          <option value="worst">Worst PnL</option>
          <option value="size">Largest size</option>
        </select>
      </div>

      <div className="table-head trades-head">
        <span>Symbol</span><span>Side</span><span>Worker / Strategy</span><span>Entry</span><span>Exit</span><span>Size</span><span>PnL</span><span>Reason</span>
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state">No trades match the selected filters.</div>
      ) : (
        <div ref={parentRef} className="virtual-table" style={{ height: Math.min(620, filtered.length * 72 + 12) }}>
          <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
            {virtualizer.getVirtualItems().map((item) => {
              const trade = filtered[item.index];
              const pnl = toNumber(trade.net_pnl) ?? 0;
              return (
                <article
                  key={`${trade.trade_id ?? trade.position_id ?? item.index}`}
                  className="trade-row"
                  style={{ transform: `translateY(${item.start}px)` }}
                >
                  <span><b>{trade.symbol ?? '-'}</b><small>{trade.trade_id ?? trade.position_id ?? ''}</small></span>
                  <span className={`side-badge ${isShort(trade.side) ? 'sell' : 'buy'}`}>{trade.side ?? '-'}</span>
                  <span><b>{trade.worker_id ?? '-'}</b><small>{trade.strategy ?? '-'}</small></span>
                  <span><b>{formatMoney(trade.entry_price)}</b><small>{formatDateTime(trade.entry_time)}</small></span>
                  <span><b>{formatMoney(trade.exit_price)}</b><small>{formatDateTime(trade.exit_time)}</small></span>
                  <span>{formatNumber(trade.quantity, 6)}</span>
                  <span className={pnl >= 0 ? 'pos' : 'neg'}><b>{formatMoney(pnl)}</b><small>{formatPercent(trade.return_pct)}</small></span>
                  <span>{trade.exit_reason ?? '-'}</span>
                </article>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

function compareTrades(left: TradeRecord, right: TradeRecord, sort: string): number {
  if (sort === 'best') return (toNumber(right.net_pnl) ?? 0) - (toNumber(left.net_pnl) ?? 0);
  if (sort === 'worst') return (toNumber(left.net_pnl) ?? 0) - (toNumber(right.net_pnl) ?? 0);
  if (sort === 'size') return (toNumber(right.quantity) ?? 0) - (toNumber(left.quantity) ?? 0);
  const leftTime = new Date(left.exit_time ?? left.entry_time ?? 0).getTime() || 0;
  const rightTime = new Date(right.exit_time ?? right.entry_time ?? 0).getTime() || 0;
  return sort === 'oldest' ? leftTime - rightTime : rightTime - leftTime;
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function isShort(side: unknown): boolean {
  const value = String(side ?? '').toLowerCase();
  return value.includes('short') || value.includes('sell');
}
