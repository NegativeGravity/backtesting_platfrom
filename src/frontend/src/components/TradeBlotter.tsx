import { useMemo, useState } from "react";

import {
  formatDateTime,
  formatMoney,
  formatNumber,
  formatPercent,
} from "../utils/formatters";

export type TradeBlotterMode = "closed_trades" | "live_executions";

export interface TradeBlotterRow {
  id: string;
  robotName?: string;
  robotId?: string;
  workerId?: string;
  strategy?: string;
  symbol: string;
  side?: "BUY" | "SELL" | string;
  quantity: number;
  entryTime?: string;
  exitTime?: string;
  signalTime?: string;
  fillTime?: string;
  entryPrice?: number;
  exitPrice?: number;
  fillPrice?: number;
  grossPnl?: number | null;
  netPnl?: number | null;
  returnPct?: number | null;
  fee?: number | null;
  slippageCost?: number | null;
  resultStatus: "profit" | "loss" | "open" | "neutral";
  resultLabel: string;
  reason?: string;
}

interface TradeBlotterProps {
  title: string;
  subtitle: string;
  mode: TradeBlotterMode;
  rows: TradeBlotterRow[];
}

type SortMode = "newest" | "oldest" | "best_pnl" | "worst_pnl" | "largest_size";

export function TradeBlotter({ title, subtitle, mode, rows }: TradeBlotterProps) {
  const [query, setQuery] = useState("");
  const [sideFilter, setSideFilter] = useState<"all" | "BUY" | "SELL">("all");
  const [resultFilter, setResultFilter] = useState<
    "all" | "profit" | "loss" | "open"
  >("all");
  const [sortMode, setSortMode] = useState<SortMode>("newest");

  const stats = useMemo(() => calculateStats(rows), [rows]);

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    return rows
      .filter((row) => {
        if (sideFilter !== "all" && row.side !== sideFilter) {
          return false;
        }

        if (resultFilter !== "all" && row.resultStatus !== resultFilter) {
          return false;
        }

        if (!normalizedQuery) {
          return true;
        }

        const searchable = [
          row.robotName,
          row.robotId,
          row.workerId,
          row.strategy,
          row.symbol,
          row.side,
          row.reason,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return searchable.includes(normalizedQuery);
      })
      .sort((left, right) => compareRows(left, right, sortMode));
  }, [rows, query, sideFilter, resultFilter, sortMode]);

  return (
    <section className="trade-blotter">
      <div className="trade-blotter-header">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>

        <div className="trade-blotter-count">
          <strong>{filteredRows.length}</strong>
          <span>shown</span>
        </div>
      </div>

      <div className="trade-stats-grid">
        <StatTile
          label="Total PnL"
          value={formatMoney(stats.totalPnl)}
          tone={stats.totalPnl >= 0 ? "good" : "bad"}
        />
        <StatTile label="Wins" value={String(stats.wins)} tone="good" />
        <StatTile label="Losses" value={String(stats.losses)} tone="bad" />
        <StatTile label="Open" value={String(stats.open)} tone="neutral" />
        <StatTile label="Fees" value={formatMoney(stats.totalFees)} tone="neutral" />
        <StatTile
          label="Slippage"
          value={formatMoney(stats.totalSlippage)}
          tone="neutral"
        />
      </div>

      <div className="trade-toolbar">
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search robot, worker, strategy, symbol..."
        />

        <select
          value={sideFilter}
          onChange={(event) =>
            setSideFilter(event.target.value as "all" | "BUY" | "SELL")
          }
        >
          <option value="all">All sides</option>
          <option value="BUY">Buy</option>
          <option value="SELL">Sell</option>
        </select>

        <select
          value={resultFilter}
          onChange={(event) =>
            setResultFilter(event.target.value as "all" | "profit" | "loss" | "open")
          }
        >
          <option value="all">All results</option>
          <option value="profit">Profit</option>
          <option value="loss">Loss</option>
          <option value="open">Open</option>
        </select>

        <select
          value={sortMode}
          onChange={(event) => setSortMode(event.target.value as SortMode)}
        >
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="best_pnl">Best PnL</option>
          <option value="worst_pnl">Worst PnL</option>
          <option value="largest_size">Largest size</option>
        </select>
      </div>

      {filteredRows.length === 0 ? (
        <div className="trade-empty-state">
          <strong>No trades match your filters</strong>
          <span>Try clearing the search or changing filters.</span>
        </div>
      ) : (
        <div className="trade-list">
          {filteredRows.map((row, index) => (
            <TradeRowCard key={row.id} row={row} index={index} mode={mode} />
          ))}
        </div>
      )}
    </section>
  );
}

function TradeRowCard({
  row,
  index,
  mode,
}: {
  row: TradeBlotterRow;
  index: number;
  mode: TradeBlotterMode;
}) {
  const isClosedMode = mode === "closed_trades";

  return (
    <article className={`trade-row-card ${row.resultStatus}`}>
      <div className="trade-row-index">{index + 1}</div>

      <div className="trade-main-cell">
        <div className="trade-symbol-line">
          <strong>{row.symbol}</strong>

          {row.side && (
            <span className={row.side === "BUY" ? "side-badge buy" : "side-badge sell"}>
              {row.side}
            </span>
          )}

          <span className={`result-badge ${row.resultStatus}`}>
            {row.resultLabel}
          </span>
        </div>

        <div className="trade-meta-line">
          {row.robotName && <span>{row.robotName}</span>}
          {row.workerId && <span>{row.workerId}</span>}
          {row.strategy && <span>{row.strategy}</span>}
          {row.reason && <span>{row.reason}</span>}
        </div>
      </div>

      <div className="trade-detail-grid">
        {isClosedMode ? (
          <>
            <InfoCell label="Entry" value={formatMoney(row.entryPrice)} />
            <InfoCell label="Exit" value={formatMoney(row.exitPrice)} />
            <InfoCell label="Qty" value={formatNumber(row.quantity, 6)} />
            <InfoCell label="Return" value={formatNullablePercent(row.returnPct)} />
            <InfoCell label="Entry Time" value={formatDateTime(row.entryTime)} />
            <InfoCell label="Exit Time" value={formatDateTime(row.exitTime)} />
            <InfoCell label="Fees" value={formatMoney(row.fee)} />
            <InfoCell label="Slippage" value={formatMoney(row.slippageCost)} />
          </>
        ) : (
          <>
            <InfoCell label="Fill Price" value={formatMoney(row.fillPrice)} />
            <InfoCell label="Quantity" value={formatNumber(row.quantity, 6)} />
            <InfoCell label="Result" value={row.resultLabel} />
            <InfoCell label="Signal Time" value={formatDateTime(row.signalTime)} />
            <InfoCell label="Fill Time" value={formatDateTime(row.fillTime)} />
            <InfoCell label="Fees" value={formatMoney(row.fee)} />
            <InfoCell label="Slippage" value={formatMoney(row.slippageCost)} />
            <InfoCell label="Worker" value={row.workerId ?? "-"} />
          </>
        )}
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

function calculateStats(rows: TradeBlotterRow[]) {
  return rows.reduce(
    (stats, row) => {
      const pnl = typeof row.netPnl === "number" ? row.netPnl : 0;
      const fee = typeof row.fee === "number" ? row.fee : 0;
      const slippage =
        typeof row.slippageCost === "number" ? row.slippageCost : 0;

      return {
        totalPnl: stats.totalPnl + pnl,
        totalFees: stats.totalFees + fee,
        totalSlippage: stats.totalSlippage + slippage,
        wins: stats.wins + (row.resultStatus === "profit" ? 1 : 0),
        losses: stats.losses + (row.resultStatus === "loss" ? 1 : 0),
        open: stats.open + (row.resultStatus === "open" ? 1 : 0),
      };
    },
    {
      totalPnl: 0,
      totalFees: 0,
      totalSlippage: 0,
      wins: 0,
      losses: 0,
      open: 0,
    },
  );
}

function compareRows(left: TradeBlotterRow, right: TradeBlotterRow, sortMode: SortMode) {
  if (sortMode === "best_pnl") {
    return getPnl(right) - getPnl(left);
  }

  if (sortMode === "worst_pnl") {
    return getPnl(left) - getPnl(right);
  }

  if (sortMode === "largest_size") {
    return right.quantity - left.quantity;
  }

  const leftTime = getRowTime(left);
  const rightTime = getRowTime(right);

  if (sortMode === "oldest") {
    return leftTime - rightTime;
  }

  return rightTime - leftTime;
}

function getPnl(row: TradeBlotterRow): number {
  return typeof row.netPnl === "number" ? row.netPnl : 0;
}

function getRowTime(row: TradeBlotterRow): number {
  const value = row.exitTime ?? row.fillTime ?? row.entryTime ?? row.signalTime ?? "";
  return new Date(value).getTime() || 0;
}

function formatNullablePercent(value: number | null | undefined): string {
  if (typeof value !== "number") return "-";
  return formatPercent(value);
}