import type { TradeRecord } from "../types";
import { formatDateTime, formatMoney, formatNumber, formatPercent } from "../utils/formatters";

interface TradesTableProps {
  trades: TradeRecord[];
}

export function TradesTable({ trades }: TradesTableProps) {
  const sortedTrades = [...trades].sort((left, right) => {
    const leftTime = left.entry_time ? new Date(left.entry_time).getTime() : 0;
    const rightTime = right.entry_time ? new Date(right.entry_time).getTime() : 0;
    return rightTime - leftTime;
  });

  return (
    <section className="glass-card table-card">
      <div className="panel-header">
        <div>
          <div className="section-kicker">Trade Ledger</div>
          <h2>Executed Positions</h2>
          <p>Entry, exit, side, strategy attribution and realized PnL.</p>
        </div>
        <div className="table-count">{trades.length} trades</div>
      </div>

      {sortedTrades.length === 0 ? (
        <div className="empty-state">No trades were generated for this run.</div>
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Side</th>
                <th>Strategy</th>
                <th>Worker</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>Entry Px</th>
                <th>Exit Px</th>
                <th>Qty</th>
                <th>Net PnL</th>
                <th>Return</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {sortedTrades.map((trade, index) => {
                const side = normalizeSide(trade.side);
                const pnl = typeof trade.net_pnl === "number" ? trade.net_pnl : null;
                return (
                  <tr key={trade.trade_id ?? trade.position_id ?? `${trade.entry_time}-${index}`}>
                    <td>
                      <span className={side === "SHORT" ? "side-pill short" : "side-pill long"}>
                        {side}
                      </span>
                    </td>
                    <td>{trade.strategy ?? "-"}</td>
                    <td>{trade.worker_id ?? "-"}</td>
                    <td>{formatDateTime(trade.entry_time)}</td>
                    <td>{formatDateTime(trade.exit_time)}</td>
                    <td>{formatMoney(trade.entry_price)}</td>
                    <td>{formatMoney(trade.exit_price)}</td>
                    <td>{formatNumber(trade.quantity, 4)}</td>
                    <td>
                      <span className={pnl !== null && pnl >= 0 ? "pnl positive" : "pnl negative"}>
                        {formatMoney(trade.net_pnl)}
                      </span>
                    </td>
                    <td>{formatPercent(trade.return_pct)}</td>
                    <td>{trade.exit_reason ?? "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function normalizeSide(value: unknown): "LONG" | "SHORT" {
  const text = String(value ?? "LONG").toUpperCase();
  return text.includes("SHORT") || text.includes("SELL_TO_OPEN") ? "SHORT" : "LONG";
}
