import type { LiveEvent } from "../api/liveApi";
import type { LivePositionPayload } from "../types";
import {
  formatDateTime,
  formatMoney,
  formatNumber,
  formatPercent,
} from "../utils/formatters";

interface LiveTradesTableProps {
  events: LiveEvent[];
}

export function LiveTradesTable({ events }: LiveTradesTableProps) {
  const { openPositions, closedPositions } = buildPositionState(events);

  return (
    <section className="trade-blotter">
      <div className="trade-blotter-header">
        <div>
          <h2>Live Positions</h2>
          <p>
            Worker-owned long/short positions with live unrealized PnL and closed
            trade results.
          </p>
        </div>

        <div className="trade-blotter-count">
          <strong>{openPositions.length}</strong>
          <span>open</span>
        </div>
      </div>

      <div className="trade-stats-grid">
        <StatTile label="Open" value={String(openPositions.length)} tone="neutral" />
        <StatTile label="Closed" value={String(closedPositions.length)} tone="neutral" />
        <StatTile
          label="Unrealized"
          value={formatMoney(
            openPositions.reduce(
              (sum, position) => sum + Number(position.unrealized_pnl ?? 0),
              0,
            ),
          )}
          tone={
            openPositions.reduce(
              (sum, position) => sum + Number(position.unrealized_pnl ?? 0),
              0,
            ) >= 0
              ? "good"
              : "bad"
          }
        />
        <StatTile
          label="Realized"
          value={formatMoney(
            closedPositions.reduce(
              (sum, position) => sum + Number(position.net_pnl ?? 0),
              0,
            ),
          )}
          tone={
            closedPositions.reduce(
              (sum, position) => sum + Number(position.net_pnl ?? 0),
              0,
            ) >= 0
              ? "good"
              : "bad"
          }
        />
      </div>

      <div className="positions-section-title">Open Positions</div>

      {openPositions.length === 0 ? (
        <div className="trade-empty-state">
          <strong>No open positions</strong>
          <span>Positions will appear here after live orders are filled.</span>
        </div>
      ) : (
        <div className="trade-list">
          {openPositions.map((position, index) => (
            <PositionCard
              key={position.position_id}
              position={position}
              index={index}
              closed={false}
            />
          ))}
        </div>
      )}

      <div className="positions-section-title">Closed Positions</div>

      {closedPositions.length === 0 ? (
        <div className="trade-empty-state">
          <strong>No closed positions yet</strong>
          <span>Closed long/short positions will appear here.</span>
        </div>
      ) : (
        <div className="trade-list">
          {closedPositions.map((position, index) => (
            <PositionCard
              key={`${position.position_id}_closed`}
              position={position}
              index={index}
              closed
            />
          ))}
        </div>
      )}
    </section>
  );
}

function PositionCard({
  position,
  index,
  closed,
}: {
  position: LivePositionPayload;
  index: number;
  closed: boolean;
}) {
  const pnl = closed
    ? Number(position.net_pnl ?? 0)
    : Number(position.unrealized_pnl ?? 0);

  const pnlPct = closed
    ? Number(position.return_pct ?? 0)
    : Number(position.unrealized_return_pct ?? 0);

  const resultStatus = pnl >= 0 ? "profit" : "loss";

  return (
    <article className={`trade-row-card ${closed ? resultStatus : "open"}`}>
      <div className="trade-row-index">{index + 1}</div>

      <div className="trade-main-cell">
        <div className="trade-symbol-line">
          <strong>{position.symbol}</strong>
          <span className={position.side === "LONG" ? "side-badge buy" : "side-badge sell"}>
            {position.side}
          </span>
          <span className={`result-badge ${closed ? resultStatus : "open"}`}>
            {closed ? formatMoney(pnl) : "OPEN"}
          </span>
        </div>

        <div className="trade-meta-line">
          <span>{position.robot_name}</span>
          <span>{position.worker_id}</span>
          <span>{position.strategy}</span>
          {closed && position.exit_reason && <span>{position.exit_reason}</span>}
        </div>
      </div>

      <div className="trade-detail-grid">
        <InfoCell label="Entry" value={formatMoney(position.entry_price)} />
        <InfoCell
          label={closed ? "Exit" : "Current"}
          value={formatMoney(closed ? position.exit_price : position.current_price)}
        />
        <InfoCell label="Quantity" value={formatNumber(position.quantity, 6)} />
        <InfoCell label="PnL" value={formatMoney(pnl)} />
        <InfoCell label="Return" value={formatPercent(pnlPct)} />
        <InfoCell label="Stop Loss" value={formatMoney(position.stop_loss)} />
        <InfoCell label="Take Profit" value={formatMoney(position.take_profit)} />
        <InfoCell
          label={closed ? "Exit Time" : "Entry Time"}
          value={formatDateTime(closed ? position.exit_time : position.entry_time)}
        />
      </div>
    </article>
  );
}

function InfoCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="trade-info-cell">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StatTile({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone: "good" | "bad" | "neutral";
}) {
  return (
    <div className={`trade-stat-tile ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function buildPositionState(events: LiveEvent[]) {
  const openMap = new Map<string, LivePositionPayload>();
  const closedMap = new Map<string, LivePositionPayload>();

  for (const event of [...events].reverse()) {
    if (event.event_type === "POSITION_OPENED") {
      const payload = event.payload as unknown as LivePositionPayload;
      openMap.set(payload.position_id, payload);
      closedMap.delete(payload.position_id);
    }

    if (event.event_type === "POSITION_UPDATED") {
      const payload = event.payload as unknown as LivePositionPayload;

      if (openMap.has(payload.position_id)) {
        openMap.set(payload.position_id, {
          ...openMap.get(payload.position_id)!,
          ...payload,
        });
      }
    }

    if (event.event_type === "POSITION_CLOSED") {
      const payload = event.payload as unknown as LivePositionPayload;
      openMap.delete(payload.position_id);
      closedMap.set(payload.position_id, payload);
    }
  }

  return {
    openPositions: [...openMap.values()].reverse(),
    closedPositions: [...closedMap.values()].reverse(),
  };
}