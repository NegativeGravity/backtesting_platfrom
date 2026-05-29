import { useEffect, useRef, useState } from 'react';
import type { LiveEvent, LivePositionPayload } from '../types';
import { formatDateTime, formatMoney, formatNumber, formatPercent } from '../utils/formatters';

interface LiveTradesTableProps {
  events: LiveEvent[];
}

export function LiveTradesTable({ events }: LiveTradesTableProps) {
  const processedRef = useRef<Set<string>>(new Set());
  const openRef = useRef<Map<string, LivePositionPayload>>(new Map());
  const closedRef = useRef<Map<string, LivePositionPayload>>(new Map());
  const [positions, setPositions] = useState(() => ({ openPositions: [] as LivePositionPayload[], closedPositions: [] as LivePositionPayload[] }));
  const { openPositions, closedPositions } = positions;

  useEffect(() => {
    if (events.length === 0) {
      processedRef.current.clear();
      openRef.current.clear();
      closedRef.current.clear();
      setPositions({ openPositions: [], closedPositions: [] });
      return;
    }

    const nextEvents = [...events].reverse().filter((event) => !processedRef.current.has(eventKey(event)));
    if (nextEvents.length === 0) return;

    let changed = false;
    for (const event of nextEvents) {
      processedRef.current.add(eventKey(event));
      changed = applyPositionEvent(event, openRef.current, closedRef.current) || changed;
    }

    if (changed) {
      setPositions({
        openPositions: [...openRef.current.values()].reverse(),
        closedPositions: [...closedRef.current.values()].reverse(),
      });
    }
  }, [events]);

  return (
    <section className="panel live-positions">
      <div className="panel-head">
        <div>
          <span className="kicker">Live blotter</span>
          <h2>Positions</h2>
          <p>Open and closed positions reconstructed from websocket events.</p>
        </div>
        <span className="badge live">{openPositions.length} open</span>
      </div>

      <div className="mini-grid">
        <Stat label="Open" value={String(openPositions.length)} />
        <Stat label="Closed" value={String(closedPositions.length)} />
        <Stat label="Unrealized" value={formatMoney(openPositions.reduce((sum, p) => sum + Number(p.unrealized_pnl ?? 0), 0))} />
        <Stat label="Realized" value={formatMoney(closedPositions.reduce((sum, p) => sum + Number(p.net_pnl ?? 0), 0))} />
      </div>

      <h3>Open Positions</h3>
      <PositionList positions={openPositions} closed={false} />
      <h3>Closed Positions</h3>
      <PositionList positions={closedPositions} closed />
    </section>
  );
}

function PositionList({ positions, closed }: { positions: LivePositionPayload[]; closed: boolean }) {
  if (positions.length === 0) return <div className="empty-state small">No {closed ? 'closed' : 'open'} positions.</div>;
  return (
    <div className="position-list">
      {positions.slice(0, 80).map((position) => {
        const pnl = Number(closed ? position.net_pnl ?? 0 : position.unrealized_pnl ?? 0);
        return (
          <article key={`${position.position_id}-${closed ? 'closed' : 'open'}`} className="position-card">
            <span><b>{position.symbol}</b><small>{position.position_id}</small></span>
            <span className={`side-badge ${String(position.side).toLowerCase().includes('short') ? 'sell' : 'buy'}`}>{position.side}</span>
            <span><b>{position.worker_id}</b><small>{position.strategy}</small></span>
            <span><b>{formatMoney(position.entry_price)}</b><small>{formatDateTime(position.entry_time)}</small></span>
            <span><b>{formatNumber(position.quantity, 6)}</b><small>quantity</small></span>
            <span className={pnl >= 0 ? 'pos' : 'neg'}><b>{formatMoney(pnl)}</b><small>{formatPercent(closed ? position.return_pct : position.unrealized_return_pct)}</small></span>
          </article>
        );
      })}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <article className="stat-tile"><span>{label}</span><strong>{value}</strong></article>;
}

function applyPositionEvent(event: LiveEvent, openMap: Map<string, LivePositionPayload>, closedMap: Map<string, LivePositionPayload>): boolean {
  const payload = event.payload as unknown as LivePositionPayload;
  if (!payload?.position_id) return false;

  if (event.event_type === 'POSITION_OPENED') {
    openMap.set(payload.position_id, payload);
    closedMap.delete(payload.position_id);
    return true;
  }

  if (event.event_type === 'POSITION_UPDATED' && openMap.has(payload.position_id)) {
    openMap.set(payload.position_id, { ...openMap.get(payload.position_id)!, ...payload });
    return true;
  }

  if (event.event_type === 'POSITION_CLOSED') {
    openMap.delete(payload.position_id);
    closedMap.set(payload.position_id, payload);
    return true;
  }

  return false;
}

function eventKey(event: LiveEvent): string {
  if (event.event_id) return event.event_id;
  return `${event.event_type}:${event.timestamp ?? ''}:${JSON.stringify(event.payload)}`;
}
