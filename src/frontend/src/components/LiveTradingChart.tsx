import { useEffect, useRef, useState } from 'react';
import { createChart, type IChartApi, type ISeriesApi, type SeriesMarker, type Time } from 'lightweight-charts';
import type { ExecutionLogRecord, LiveEvent } from '../types';
import { toNumber, toUnixSeconds } from '../utils/formatters';
import { ChartToolbar, type MarkerMode } from './ChartToolbar';

interface LiveTradingChartProps {
  events: LiveEvent[];
}

type CandlePoint = {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
};

type LinePoint = {
  time: Time;
  value: number;
};

const AUTO_MARKER_BUDGET = 220;
const FOLLOW_VISIBLE_BARS = 140;
const FOLLOW_RIGHT_OFFSET = 10;

export function LiveTradingChart({ events }: LiveTradingChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const priceChartRef = useRef<IChartApi | null>(null);
  const equityChartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  const processedRef = useRef<Set<string>>(new Set());
  const candlesRef = useRef<Map<number, CandlePoint>>(new Map());
  const equityRef = useRef<Map<number, LinePoint>>(new Map());
  const markersRef = useRef<SeriesMarker<Time>[]>([]);

  const [markerMode, setMarkerMode] = useState<MarkerMode>('auto');
  const [equityVisible, setEquityVisible] = useState(true);
  const [counts, setCounts] = useState({ candles: 0, markers: 0 });

  useEffect(() => {
    if (!containerRef.current) return;

    containerRef.current.innerHTML = '';

    const priceBox = document.createElement('div');
    const equityBox = document.createElement('div');

    priceBox.className = 'chart-box price-chart-box live-price';
    equityBox.className = 'chart-box sub-chart-box live-equity';

    containerRef.current.append(priceBox, equityBox);

    const baseOptions = {
      autoSize: true,
      layout: {
        textColor: '#b8c7dd',
        background: { color: '#0b1220' },
      },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
      },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.22)',
        timeVisible: true,
        secondsVisible: true,
        rightOffset: FOLLOW_RIGHT_OFFSET,
        barSpacing: 7,
        minBarSpacing: 0.35,
        lockVisibleTimeRangeOnResize: true,
      },
      rightPriceScale: {
        borderColor: 'rgba(148, 163, 184, 0.22)',
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: 'rgba(96, 165, 250, 0.75)',
          style: 3,
          labelBackgroundColor: '#2563eb',
        },
        horzLine: {
          color: 'rgba(96, 165, 250, 0.75)',
          style: 3,
          labelBackgroundColor: '#2563eb',
        },
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
    } as const;

    const priceChart = createChart(priceBox, { ...baseOptions, height: 430 });

    const candleSeries = priceChart.addCandlestickSeries({
      upColor: '#22ab94',
      downColor: '#f23645',
      borderUpColor: '#22ab94',
      borderDownColor: '#f23645',
      wickUpColor: '#22ab94',
      wickDownColor: '#f23645',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });

    const equityChart = createChart(equityBox, { ...baseOptions, height: 180 });

    const equitySeries = equityChart.addLineSeries({
      color: '#60a5fa',
      lineWidth: 2,
      title: 'Live Equity',
    });

    priceChartRef.current = priceChart;
    equityChartRef.current = equityChart;
    candleSeriesRef.current = candleSeries;
    equitySeriesRef.current = equitySeries;

    const syncRange = () => {
      const range = priceChart.timeScale().getVisibleLogicalRange();
      if (range) equityChart.timeScale().setVisibleLogicalRange(range);
    };

    priceChart.timeScale().subscribeVisibleLogicalRangeChange(syncRange);

    return () => {
      priceChart.timeScale().unsubscribeVisibleLogicalRangeChange(syncRange);

      priceChart.remove();
      equityChart.remove();

      priceChartRef.current = null;
      equityChartRef.current = null;
      candleSeriesRef.current = null;
      equitySeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (events.length === 0) {
      resetChartState();
      return;
    }

    const nextEvents = [...events]
      .reverse()
      .filter((event) => !processedRef.current.has(eventKey(event)));

    if (nextEvents.length === 0) return;

    let changedMarkers = false;
    let changedTimeline = false;

    for (const event of nextEvents) {
      const key = eventKey(event);
      processedRef.current.add(key);

      const update = applyLiveEvent(event, candlesRef.current, equityRef.current, markersRef.current);

      changedMarkers = changedMarkers || update.marker;
      changedTimeline = changedTimeline || Boolean(update.candle || update.equity);

      if (update.candle) {
        candleSeriesRef.current?.update(update.candle);
      }

      if (update.equity) {
        equitySeriesRef.current?.update(update.equity);
      }
    }

    if (changedMarkers) {
      applyMarkers(markerMode);
    }

    setCounts({
      candles: candlesRef.current.size,
      markers: markersRef.current.length,
    });

    if (changedTimeline) {
      followLiveTail();
    }
  }, [events, markerMode]);

  useEffect(() => {
    applyMarkers(markerMode);
  }, [markerMode]);

  useEffect(() => {
    equitySeriesRef.current?.applyOptions({ visible: equityVisible });

    const equityBox = containerRef.current?.querySelector('.live-equity') as HTMLElement | null;
    if (equityBox) equityBox.hidden = !equityVisible;

    if (equityVisible) {
      followLiveTail();
    }
  }, [equityVisible]);

  function resetChartState() {
    processedRef.current.clear();
    candlesRef.current.clear();
    equityRef.current.clear();
    markersRef.current = [];

    candleSeriesRef.current?.setData([]);
    equitySeriesRef.current?.setData([]);
    candleSeriesRef.current?.setMarkers([]);

    setCounts({ candles: 0, markers: 0 });
  }

  function applyMarkers(mode: MarkerMode) {
    const markers = mode === 'off'
      ? []
      : mode === 'auto'
        ? markersRef.current.slice(-AUTO_MARKER_BUDGET)
        : markersRef.current;

    candleSeriesRef.current?.setMarkers(markers);
  }

  function followLiveTail() {
    const priceChart = priceChartRef.current;
    const equityChart = equityChartRef.current;

    if (!priceChart) return;

    const candleCount = candlesRef.current.size;
    if (candleCount <= 0) return;

    const to = Math.max(0, candleCount - 1) + FOLLOW_RIGHT_OFFSET;
    const from = Math.max(0, to - FOLLOW_VISIBLE_BARS);

    priceChart.timeScale().setVisibleLogicalRange({ from, to });
    equityChart?.timeScale().setVisibleLogicalRange({ from, to });
  }

  function fit() {
    priceChartRef.current?.timeScale().fitContent();
    equityChartRef.current?.timeScale().fitContent();
  }

  function zoom(multiplier: number) {
    const range = priceChartRef.current?.timeScale().getVisibleLogicalRange();
    if (!range) return;

    const center = (range.from + range.to) / 2;
    const half = ((range.to - range.from) / 2) * multiplier;

    priceChartRef.current?.timeScale().setVisibleLogicalRange({
      from: center - half,
      to: center + half,
    });
  }

  function goLatest() {
    followLiveTail();
  }

  function screenshot() {
    const canvas = priceChartRef.current?.takeScreenshot();
    if (!canvas) return;

    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'live-chart.png';
    link.click();
  }

  return (
    <section className="panel chart-shell">
      <ChartToolbar
        title="Live Replay Chart"
        subtitle={`${counts.candles.toLocaleString()} bars · ${counts.markers.toLocaleString()} fills`}
        markerMode={markerMode}
        equityVisible={equityVisible}
        onFit={fit}
        onZoomIn={() => zoom(0.74)}
        onZoomOut={() => zoom(1.26)}
        onGoToLatest={goLatest}
        onMarkerModeChange={setMarkerMode}
        onToggleEquity={() => setEquityVisible((value) => !value)}
        onScreenshot={screenshot}
      />

      <section ref={containerRef} className="charts live-charts" />
    </section>
  );
}

function applyLiveEvent(
  event: LiveEvent,
  candles: Map<number, CandlePoint>,
  equity: Map<number, LinePoint>,
  markers: SeriesMarker<Time>[],
) {
  const payload = event.payload;
  const result: { candle?: CandlePoint; equity?: LinePoint; marker: boolean } = { marker: false };

  if (event.event_type === 'MARKET_BAR') {
    const time = toUnixSeconds(payload.timestamp ?? event.timestamp);
    const open = toNumber(payload.open);
    const high = toNumber(payload.high);
    const low = toNumber(payload.low);
    const close = toNumber(payload.close);

    if (time !== null && open !== null && high !== null && low !== null && close !== null) {
      const candle = {
        time: time as Time,
        open,
        high,
        low,
        close,
      };

      candles.set(time, candle);
      result.candle = candle;
    }
  }

  if (event.event_type === 'PORTFOLIO_UPDATED') {
    const time = toUnixSeconds(payload.timestamp ?? event.timestamp);
    const value = toNumber(payload.equity);

    if (time !== null && value !== null) {
      const point = {
        time: time as Time,
        value,
      };

      equity.set(time, point);
      result.equity = point;
    }
  }

  if (event.event_type === 'ORDER_FILLED') {
    const fill = payload as unknown as ExecutionLogRecord;
    const time = toUnixSeconds(payload.fill_time ?? fill.filled_bar_time ?? event.timestamp);

    if (time !== null) {
      const side = String(payload.side ?? fill.side ?? '').toUpperCase();
      const sell = side.includes('SELL') || side.includes('SHORT');

      markers.push({
        time: time as Time,
        position: sell ? 'aboveBar' : 'belowBar',
        color: sell ? '#f23645' : '#22ab94',
        shape: sell ? 'arrowDown' : 'arrowUp',
        text: `${payload.worker_id ?? fill.worker_id ?? 'fill'} ${side}`,
      });

      markers.sort((left, right) => Number(left.time) - Number(right.time));
      result.marker = true;
    }
  }

  return result;
}

function eventKey(event: LiveEvent): string {
  if (event.event_id) return event.event_id;

  return `${event.event_type}:${event.timestamp ?? ''}:${JSON.stringify(event.payload)}`;
}