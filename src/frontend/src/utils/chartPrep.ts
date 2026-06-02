import type { ChartDataResponse, TradeRecord } from '../types';
import { toUnixSeconds, toNumber } from './formatters';

export interface PreparedCandle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface PreparedLinePoint {
  time: number;
  value: number;
}

export interface PreparedMarker {
  time: number;
  position: 'aboveBar' | 'belowBar' | 'inBar';
  color: string;
  shape: 'arrowUp' | 'arrowDown' | 'circle' | 'square';
  text?: string;
}

export interface PreparedChartData {
  candles: PreparedCandle[];
  equity: PreparedLinePoint[];
  benchmark: PreparedLinePoint[];
  drawdown: PreparedLinePoint[];
  markers: PreparedMarker[];
}

export interface ChartWorkerRequest {
  id: string;
  chartData: ChartDataResponse | null;
  trades: TradeRecord[];
}

export interface ChartWorkerResponse {
  id: string;
  payload: PreparedChartData;
}

export function emptyPreparedChartData(): PreparedChartData {
  return { candles: [], equity: [], benchmark: [], drawdown: [], markers: [] };
}

export function prepareChartPayload(chartData: ChartDataResponse | null, trades: TradeRecord[] = []): PreparedChartData {
  if (!chartData) return emptyPreparedChartData();

  const candles = sanitizeCandles(chartData.candles);
  const equity = sanitizeLine(chartData.equity_curve);
  const benchmark = sanitizeLine(chartData.benchmark_curve);
  const apiMarkers = sanitizeApiMarkers(chartData.markers);
  const tradeMarkers = markersFromTrades(trades);
  const markers = normalizeMarkers([...apiMarkers, ...tradeMarkers], candles.map((candle) => candle.time));

  return {
    candles,
    equity,
    benchmark,
    drawdown: buildDrawdown(equity),
    markers,
  };
}

function sanitizeCandles(points: ChartDataResponse['candles'] | undefined): PreparedCandle[] {
  const byTime = new Map<number, PreparedCandle>();

  for (const point of points ?? []) {
    const time = toUnixSeconds(point?.time);
    const open = toNumber(point?.open);
    const high = toNumber(point?.high);
    const low = toNumber(point?.low);
    const close = toNumber(point?.close);

    if (time === null || open === null || high === null || low === null || close === null) continue;

    const normalizedHigh = Math.max(open, high, low, close);
    const normalizedLow = Math.min(open, high, low, close);
    const existing = byTime.get(time);

    byTime.set(time, existing ? {
      time,
      open: existing.open,
      high: Math.max(existing.high, normalizedHigh),
      low: Math.min(existing.low, normalizedLow),
      close,
    } : { time, open, high: normalizedHigh, low: normalizedLow, close });
  }

  return [...byTime.values()].sort((left, right) => left.time - right.time);
}

function sanitizeLine(points: ChartDataResponse['equity_curve'] | undefined): PreparedLinePoint[] {
  const byTime = new Map<number, PreparedLinePoint>();

  for (const point of points ?? []) {
    const time = toUnixSeconds(point?.time);
    const value = toNumber(point?.value);
    if (time === null || value === null) continue;
    byTime.set(time, { time, value });
  }

  return [...byTime.values()].sort((left, right) => left.time - right.time);
}

function sanitizeApiMarkers(markers: ChartDataResponse['markers'] | undefined): PreparedMarker[] {
  return (markers ?? [])
    .reduce<PreparedMarker[]>((items, marker) => {
      const time = toUnixSeconds(marker.time);
      if (time === null) return items;
      items.push({
        time,
        position: marker.position ?? 'aboveBar',
        color: marker.color ?? '#60a5fa',
        shape: marker.shape ?? 'circle',
        text: marker.text,
      });
      return items;
    }, [])
    .sort((left, right) => left.time - right.time);
}

function normalizeMarkers(markers: PreparedMarker[], candleTimes: number[]): PreparedMarker[] {
  if (markers.length === 0) return [];
  if (candleTimes.length === 0) return dedupeMarkers(markers).sort((left, right) => left.time - right.time);

  const sortedTimes = [...new Set(candleTimes)].sort((left, right) => left - right);
  const toleranceSeconds = inferMarkerSnapTolerance(sortedTimes);

  return dedupeMarkers(
    markers.reduce<PreparedMarker[]>((items, marker) => {
      const snappedTime = snapToNearestCandleTime(marker.time, sortedTimes, toleranceSeconds);
      if (snappedTime === null) return items;
      items.push({ ...marker, time: snappedTime });
      return items;
    }, []),
  ).sort((left, right) => left.time - right.time);
}

function inferMarkerSnapTolerance(sortedTimes: number[]): number {
  if (sortedTimes.length < 2) return Number.POSITIVE_INFINITY;

  const gaps = sortedTimes
    .slice(1)
    .map((time, index) => time - sortedTimes[index])
    .filter((gap) => Number.isFinite(gap) && gap > 0)
    .sort((left, right) => left - right);

  if (gaps.length === 0) return Number.POSITIVE_INFINITY;

  const medianGap = gaps[Math.floor(gaps.length / 2)];
  return Math.max(60, medianGap);
}

function snapToNearestCandleTime(time: number, sortedTimes: number[], toleranceSeconds: number): number | null {
  let low = 0;
  let high = sortedTimes.length - 1;

  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const candidate = sortedTimes[mid];

    if (candidate === time) return candidate;
    if (candidate < time) low = mid + 1;
    else high = mid - 1;
  }

  const right = low < sortedTimes.length ? sortedTimes[low] : null;
  const left = high >= 0 ? sortedTimes[high] : null;
  const candidates = [left, right].filter((value): value is number => value !== null);
  if (candidates.length === 0) return null;

  const nearest = candidates.reduce((best, candidate) => (
    Math.abs(candidate - time) < Math.abs(best - time) ? candidate : best
  ), candidates[0]);

  return Math.abs(nearest - time) <= toleranceSeconds ? nearest : null;
}

function dedupeMarkers(markers: PreparedMarker[]): PreparedMarker[] {
  const seen = new Set<string>();
  const output: PreparedMarker[] = [];

  for (const marker of markers) {
    const key = `${marker.time}|${marker.position}|${marker.shape}|${marker.text ?? ''}`;
    if (seen.has(key)) continue;
    seen.add(key);
    output.push(marker);
  }

  return output;
}

function markersFromTrades(trades: TradeRecord[]): PreparedMarker[] {
  const markers: PreparedMarker[] = [];

  for (const trade of trades) {
    const entryTime = toUnixSeconds(trade.entry_time);
    const exitTime = toUnixSeconds(trade.exit_time);
    const side = String(trade.side ?? '').toUpperCase();
    const isShort = side.includes('SHORT') || side.includes('SELL');
    const pnl = toNumber(trade.net_pnl) ?? 0;

    if (entryTime !== null) {
      markers.push({
        time: entryTime,
        position: isShort ? 'aboveBar' : 'belowBar',
        color: isShort ? '#f87171' : '#34d399',
        shape: isShort ? 'arrowDown' : 'arrowUp',
        text: `${isShort ? 'Short' : 'Long'} ${trade.worker_id ?? trade.strategy ?? ''}`.trim(),
      });
    }

    if (exitTime !== null) {
      markers.push({
        time: exitTime,
        position: pnl >= 0 ? 'aboveBar' : 'belowBar',
        color: pnl >= 0 ? '#22c55e' : '#fb7185',
        shape: 'circle',
        text: `${pnl >= 0 ? 'TP' : 'Exit'} ${trade.exit_reason ?? ''}`.trim(),
      });
    }
  }

  return markers.sort((left, right) => left.time - right.time);
}

function buildDrawdown(equity: PreparedLinePoint[]): PreparedLinePoint[] {
  let peak = Number.NEGATIVE_INFINITY;
  return equity.map((point) => {
    peak = Math.max(peak, point.value);
    const value = peak > 0 ? (point.value - peak) / peak : 0;
    return { time: point.time, value };
  });
}
