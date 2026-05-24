import type { SeriesMarker, Time } from "lightweight-charts";

import type { ExecutionLogRecord, TradeMarker, TradeRecord } from "../types";
import { toUnixSeconds } from "./formatters";

export function historicalMarkersFromTrades(trades: TradeRecord[]): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];

  for (const trade of trades) {
    const side = normalizeTradeSide(trade.side);
    const quantity = formatTradeQuantity(trade.quantity);
    const worker = trade.worker_id ? `${trade.worker_id} ` : "";
    const strategy = trade.strategy ? `${trade.strategy} ` : "";

    if (trade.entry_time) {
      markers.push({
        time: toUnixSeconds(trade.entry_time) as Time,
        position: side === "SHORT" ? "aboveBar" : "belowBar",
        color: side === "SHORT" ? "#ff6b87" : "#67e8f9",
        shape: side === "SHORT" ? "arrowDown" : "arrowUp",
        text: `${worker}${strategy}${side} OPEN ${quantity}`.trim(),
      });
    }

    if (trade.exit_time) {
      const isWinner = typeof trade.net_pnl === "number" && trade.net_pnl >= 0;
      markers.push({
        time: toUnixSeconds(trade.exit_time) as Time,
        position: side === "SHORT" ? "belowBar" : "aboveBar",
        color: isWinner ? "#8b5cf6" : "#fb4567",
        shape: side === "SHORT" ? "arrowUp" : "arrowDown",
        text: `${worker}${side} CLOSE ${formatPnl(trade.net_pnl)}`.trim(),
      });
    }
  }

  return sortMarkers(markers);
}

export function markersFromApiMarkers(markers: TradeMarker[]): SeriesMarker<Time>[] {
  return sortMarkers(
    markers
      .filter((marker) => Number.isFinite(marker.time))
      .map((marker) => ({
        time: marker.time as Time,
        position: marker.position,
        color: normalizeMarkerColor(marker.color),
        shape: marker.shape,
        text: marker.text,
      })),
  );
}

export function markerFromExecutionFill(fill: ExecutionLogRecord): SeriesMarker<Time> | null {
  if (!fill.filled_bar_time || !fill.side) {
    return null;
  }

  const side = String(fill.side).toUpperCase();
  const isBuy = side.includes("BUY");

  return {
    time: toUnixSeconds(fill.filled_bar_time) as Time,
    position: isBuy ? "belowBar" : "aboveBar",
    color: isBuy ? "#67e8f9" : "#ff6b87",
    shape: isBuy ? "arrowUp" : "arrowDown",
    text: `${fill.side} ${formatTradeQuantity(fill.quantity)}`,
  };
}

function normalizeTradeSide(value: unknown): "LONG" | "SHORT" {
  const side = String(value ?? "LONG").toUpperCase();
  return side.includes("SHORT") || side.includes("SELL_TO_OPEN") ? "SHORT" : "LONG";
}

function normalizeMarkerColor(color: string): string {
  const normalized = color.toLowerCase();
  if (normalized.includes("22c55e") || normalized.includes("green")) return "#67e8f9";
  if (normalized.includes("38bdf8")) return "#8b5cf6";
  return color;
}

function sortMarkers(markers: SeriesMarker<Time>[]): SeriesMarker<Time>[] {
  return [...markers].sort((a, b) => Number(a.time) - Number(b.time));
}

function formatTradeQuantity(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "";
  return value.toFixed(4);
}

function formatPnl(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "";
  const prefix = value >= 0 ? "+" : "";
  return `${prefix}${value.toFixed(2)}`;
}
