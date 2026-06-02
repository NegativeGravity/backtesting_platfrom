import { useCallback, useEffect, useRef, useState } from 'react';
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from 'lightweight-charts';
import type { ChartDataResponse, TradeRecord } from '../types';
import { usePreparedChartData } from '../hooks/usePreparedChartData';
import type { PreparedCandle, PreparedLinePoint } from '../utils/chartPrep';
import { ChartToolbar, type MarkerMode } from './ChartToolbar';

const FOLLOW_VISIBLE_BARS = 140;
const FOLLOW_RIGHT_OFFSET = 10;

interface ChartPanelProps {
  chartData: ChartDataResponse | null;
  trades: TradeRecord[];
  title?: string;
}

const INITIAL_BARS = 180;
const AUTO_MARKER_BUDGET = 280;
const RESET_JUMP_THRESHOLD = 1500;

type LineCursorKey = 'equity' | 'benchmark' | 'drawdown';

export function ChartPanel({ chartData, trades, title = 'Research Replay Chart' }: ChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const priceBoxRef = useRef<HTMLDivElement | null>(null);
  const equityBoxRef = useRef<HTMLDivElement | null>(null);
  const drawdownBoxRef = useRef<HTMLDivElement | null>(null);

  const priceChartRef = useRef<IChartApi | null>(null);
  const equityChartRef = useRef<IChartApi | null>(null);
  const drawdownChartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const benchmarkSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const drawdownSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  const prepared = usePreparedChartData(chartData, trades);
  const [markerMode, setMarkerMode] = useState<MarkerMode>('auto');
  const [equityVisible, setEquityVisible] = useState(true);
  const [drawdownVisible, setDrawdownVisible] = useState(false);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(8);
  const [replayIndex, setReplayIndex] = useState(INITIAL_BARS);
  const [chartError, setChartError] = useState<string | null>(null);

  const replayRef = useRef({ index: INITIAL_BARS, lastFrame: 0, remainder: 0 });
  const renderedIndexRef = useRef(0);
  const lineCursorRef = useRef<Record<LineCursorKey, number>>({ equity: 0, benchmark: 0, drawdown: 0 });

  const progress = prepared.candles.length > 0
    ? (Math.min(replayIndex, prepared.candles.length) / prepared.candles.length) * 100
    : 0;

  const markersUntil = useCallback((maxTime: number): SeriesMarker<Time>[] => {
    if (markerMode === 'off' || maxTime <= 0) return [];
    const end = upperBoundMarkers(prepared.markers, maxTime);
    const start = markerMode === 'auto' ? Math.max(0, end - AUTO_MARKER_BUDGET) : 0;
    return prepared.markers.slice(start, end).map((marker) => ({ ...marker, time: marker.time as Time }));
  }, [markerMode, prepared.markers]);

  useEffect(() => {
    const start = Math.min(INITIAL_BARS, Math.max(1, prepared.candles.length));
    setReplayIndex(start);
    replayRef.current = { index: start, lastFrame: 0, remainder: 0 };
    renderedIndexRef.current = 0;
    lineCursorRef.current = { equity: 0, benchmark: 0, drawdown: 0 };
    setIsReplaying(false);
  }, [prepared.candles.length]);

  useEffect(() => {
    if (!containerRef.current) return;

    containerRef.current.innerHTML = '';
    const priceBox = document.createElement('div');
    const equityBox = document.createElement('div');
    const drawdownBox = document.createElement('div');
    priceBox.className = 'chart-box price-chart-box';
    equityBox.className = 'chart-box sub-chart-box';
    drawdownBox.className = 'chart-box sub-chart-box drawdown-chart-box';
    containerRef.current.append(priceBox, equityBox, drawdownBox);
    priceBoxRef.current = priceBox;
    equityBoxRef.current = equityBox;
    drawdownBoxRef.current = drawdownBox;

    const chartOptions = {
      autoSize: true,
      layout: { textColor: '#b8c7dd', background: { color: '#0b1220' } },
      grid: {
        vertLines: { color: 'rgba(148, 163, 184, 0.08)' },
        horzLines: { color: 'rgba(148, 163, 184, 0.08)' },
      },
      rightPriceScale: { borderColor: 'rgba(148, 163, 184, 0.22)' },
      timeScale: {
        borderColor: 'rgba(148, 163, 184, 0.22)',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 12,
        barSpacing: 7,
        minBarSpacing: 0.35,
        lockVisibleTimeRangeOnResize: true,
      },
      crosshair: {
        mode: 1,
        vertLine: { color: 'rgba(96, 165, 250, .75)', style: 3, labelBackgroundColor: '#2563eb' },
        horzLine: { color: 'rgba(96, 165, 250, .75)', style: 3, labelBackgroundColor: '#2563eb' },
      },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true },
    } as const;

    const priceChart = createChart(priceBox, { ...chartOptions, height: 560 });
    const candleSeries = priceChart.addCandlestickSeries({
      upColor: '#22ab94',
      downColor: '#f23645',
      borderUpColor: '#22ab94',
      borderDownColor: '#f23645',
      wickUpColor: '#22ab94',
      wickDownColor: '#f23645',
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });

    const equityChart = createChart(equityBox, { ...chartOptions, height: 210 });
    const equitySeries = equityChart.addLineSeries({ title: 'Strategy Equity', color: '#60a5fa', lineWidth: 2 });
    const benchmarkSeries = equityChart.addLineSeries({ title: 'Benchmark', color: '#f59e0b', lineWidth: 2 });

    const drawdownChart = createChart(drawdownBox, { ...chartOptions, height: 170 });
    const drawdownSeries = drawdownChart.addAreaSeries({
      title: 'Drawdown',
      lineColor: '#fb7185',
      topColor: 'rgba(251, 113, 133, 0.18)',
      bottomColor: 'rgba(251, 113, 133, 0.02)',
      lineWidth: 2,
      priceFormat: { type: 'percent' },
    });

    priceChartRef.current = priceChart;
    equityChartRef.current = equityChart;
    drawdownChartRef.current = drawdownChart;
    candleSeriesRef.current = candleSeries;
    equitySeriesRef.current = equitySeries;
    benchmarkSeriesRef.current = benchmarkSeries;
    drawdownSeriesRef.current = drawdownSeries;

    const syncRange = () => {
      const range = priceChart.timeScale().getVisibleLogicalRange();
      if (range) {
        equityChart.timeScale().setVisibleLogicalRange(range);
        drawdownChart.timeScale().setVisibleLogicalRange(range);
      }
    };
    priceChart.timeScale().subscribeVisibleLogicalRangeChange(syncRange);

    return () => {
      priceChart.timeScale().unsubscribeVisibleLogicalRangeChange(syncRange);
      priceChart.remove();
      equityChart.remove();
      drawdownChart.remove();
      priceChartRef.current = null;
      equityChartRef.current = null;
      drawdownChartRef.current = null;
      candleSeriesRef.current = null;
      equitySeriesRef.current = null;
      benchmarkSeriesRef.current = null;
      drawdownSeriesRef.current = null;
    };
  }, []);

  const applyRange = useCallback((index: number, reset = false, follow = false) => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) return;

    const safeIndex = Math.min(Math.max(index, 0), prepared.candles.length);
    const previousIndex = renderedIndexRef.current;
    const shouldReset = reset || previousIndex <= 0 || safeIndex <= previousIndex || safeIndex - previousIndex > RESET_JUMP_THRESHOLD;

    try {
      if (shouldReset) {
        resetSeriesToIndex(safeIndex);
      } else {
        for (let i = previousIndex; i < safeIndex; i += 1) {
          candleSeries.update(toChartCandle(prepared.candles[i]));
        }
        const lastTime = lastCandleTime(prepared.candles, safeIndex);
        updateLineUntil(equitySeriesRef.current, prepared.equity, 'equity', lastTime);
        updateLineUntil(benchmarkSeriesRef.current, prepared.benchmark, 'benchmark', lastTime);
        updateLineUntil(drawdownSeriesRef.current, prepared.drawdown, 'drawdown', lastTime);
        candleSeries.setMarkers(markersUntil(lastTime));
        renderedIndexRef.current = safeIndex;
      }

      if (reset) priceChartRef.current?.timeScale().fitContent();
      if (follow) followReplayTail(safeIndex);
      setChartError(null);
    } catch (error) {
      console.error(error);
      setChartError(error instanceof Error ? error.message : 'Could not render chart data.');
    }
  }, [markersUntil, prepared.benchmark, prepared.candles, prepared.drawdown, prepared.equity]);

  function resetSeriesToIndex(index: number) {
    const lastTime = lastCandleTime(prepared.candles, index);
    candleSeriesRef.current?.setData(prepared.candles.slice(0, index).map(toChartCandle));

    const equityEnd = upperBoundPoints(prepared.equity, lastTime);
    const benchmarkEnd = upperBoundPoints(prepared.benchmark, lastTime);
    const drawdownEnd = upperBoundPoints(prepared.drawdown, lastTime);
    equitySeriesRef.current?.setData(prepared.equity.slice(0, equityEnd).map(toChartPoint));
    benchmarkSeriesRef.current?.setData(prepared.benchmark.slice(0, benchmarkEnd).map(toChartPoint));
    drawdownSeriesRef.current?.setData(prepared.drawdown.slice(0, drawdownEnd).map(toChartPoint));
    lineCursorRef.current = { equity: equityEnd, benchmark: benchmarkEnd, drawdown: drawdownEnd };
    candleSeriesRef.current?.setMarkers(markersUntil(lastTime));
    renderedIndexRef.current = index;
  }

  function updateLineUntil(series: ISeriesApi<'Line'> | ISeriesApi<'Area'> | null, points: PreparedLinePoint[], key: LineCursorKey, maxTime: number) {
    if (!series) return;
    let cursor = lineCursorRef.current[key];
    while (cursor < points.length && Number(points[cursor].time) <= maxTime) {
      series.update(toChartPoint(points[cursor]));
      cursor += 1;
    }
    lineCursorRef.current[key] = cursor;
  }

  useEffect(() => {
    if (!prepared.candles.length) return;
    applyRange(replayIndex, true);
  }, [applyRange, prepared.candles.length]);

  useEffect(() => {
    const lastTime = lastCandleTime(prepared.candles, replayIndex);
    candleSeriesRef.current?.setMarkers(markersUntil(lastTime));
  }, [markersUntil, replayIndex, prepared.candles]);

  useEffect(() => {
    equitySeriesRef.current?.applyOptions({ visible: equityVisible });
    benchmarkSeriesRef.current?.applyOptions({ visible: equityVisible });
    if (equityBoxRef.current) equityBoxRef.current.hidden = !equityVisible;
  }, [equityVisible]);

  useEffect(() => {
    drawdownSeriesRef.current?.applyOptions({ visible: drawdownVisible });
    if (drawdownBoxRef.current) drawdownBoxRef.current.hidden = !drawdownVisible;
  }, [drawdownVisible]);

  useEffect(() => {
    if (!isReplaying || prepared.candles.length === 0) return;
    let raf = 0;

    const frame = (time: number) => {
      const state = replayRef.current;
      const delta = state.lastFrame === 0 ? 16 : Math.min(120, time - state.lastFrame);
      state.lastFrame = time;
      state.remainder += (delta / 100) * replaySpeed;
      const steps = Math.floor(state.remainder);

      if (steps > 0) {
        state.remainder -= steps;
        const next = Math.min(state.index + steps, prepared.candles.length);
        applyRange(next, false,true);
        state.index = next;
        setReplayIndex(next);

        if (next >= prepared.candles.length) {
          setIsReplaying(false);
          return;
        }
      }

      raf = window.requestAnimationFrame(frame);
    };

    raf = window.requestAnimationFrame(frame);
    return () => window.cancelAnimationFrame(raf);
  }, [applyRange, isReplaying, prepared.candles.length, replaySpeed]);

  function fitCharts() {
    priceChartRef.current?.timeScale().fitContent();
    equityChartRef.current?.timeScale().fitContent();
    drawdownChartRef.current?.timeScale().fitContent();
  }

  function zoom(multiplier: number) {
    const range = priceChartRef.current?.timeScale().getVisibleLogicalRange();
    if (!range) return;
    const center = (range.from + range.to) / 2;
    const half = ((range.to - range.from) / 2) * multiplier;
    priceChartRef.current?.timeScale().setVisibleLogicalRange({ from: center - half, to: center + half });
  }

  function followReplayTail(index: number) {
      if (!priceChartRef.current || index <= 0) return;

      const to = index + FOLLOW_RIGHT_OFFSET;
      const from = Math.max(0, to - FOLLOW_VISIBLE_BARS);

      priceChartRef.current.timeScale().setVisibleLogicalRange({ from, to });
      equityChartRef.current?.timeScale().setVisibleLogicalRange({ from, to });
      drawdownChartRef.current?.timeScale().setVisibleLogicalRange({ from, to });
    }

  function goLatest() {
    const last = prepared.candles.length;
    replayRef.current.index = last;
    setReplayIndex(last);
    applyRange(last, true,true);
    priceChartRef.current?.timeScale().scrollToRealTime();
  }

  function resetReplay() {
    const start = Math.min(INITIAL_BARS, Math.max(1, prepared.candles.length));
    replayRef.current = { index: start, lastFrame: 0, remainder: 0 };
    setReplayIndex(start);
    setIsReplaying(false);
    applyRange(start, true,true);
  }

  function stepReplay() {
    const next = Math.min(replayRef.current.index + 1, prepared.candles.length);
    replayRef.current.index = next;
    setReplayIndex(next);
    applyRange(next, false,true);
  }

  function toggleReplay() {
    replayRef.current.lastFrame = 0;
    setIsReplaying((value) => !value);
  }

  function screenshot() {
    const canvas = priceChartRef.current?.takeScreenshot();
    if (!canvas) return;
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'quant-chart.png';
    link.click();
  }

  if (!chartData) {
    return (
      <section className="panel empty-chart">
        <h2>No chart loaded</h2>
        <p>Select a run to load candles, trades, replay, equity and drawdown.</p>
      </section>
    );
  }

  return (
    <section className="panel chart-shell">
      <ChartToolbar
        title={title}
        subtitle={`${prepared.candles.length.toLocaleString()} candles · ${prepared.markers.length.toLocaleString()} trade markers`}
        markerMode={markerMode}
        equityVisible={equityVisible}
        drawdownVisible={drawdownVisible}
        isReplaying={isReplaying}
        replayProgress={progress}
        replaySpeed={replaySpeed}
        onFit={fitCharts}
        onZoomIn={() => zoom(0.72)}
        onZoomOut={() => zoom(1.28)}
        onGoToLatest={goLatest}
        onMarkerModeChange={setMarkerMode}
        onToggleEquity={() => setEquityVisible((value) => !value)}
        onToggleDrawdown={() => setDrawdownVisible((value) => !value)}
        onScreenshot={screenshot}
        onReplayPlayPause={toggleReplay}
        onReplayReset={resetReplay}
        onReplayStep={stepReplay}
        onReplaySpeedChange={setReplaySpeed}
      />
      <div className="progress-track"><div style={{ width: `${progress}%` }} /></div>
      {chartError && <div className="notice error"><strong>Chart error</strong><p>{chartError}</p></div>}
      <section ref={containerRef} className="charts" />
    </section>
  );
}

function toChartCandle(point: PreparedCandle) {
  return { time: point.time as Time, open: point.open, high: point.high, low: point.low, close: point.close };
}

function toChartPoint(point: PreparedLinePoint) {
  return { time: point.time as Time, value: point.value };
}

function lastCandleTime(candles: PreparedCandle[], index: number): number {
  if (candles.length === 0 || index <= 0) return 0;
  return Number(candles[Math.min(index, candles.length) - 1]?.time ?? 0);
}

function upperBoundPoints(points: PreparedLinePoint[], maxTime: number): number {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (Number(points[mid].time) <= maxTime) low = mid + 1;
    else high = mid;
  }
  return low;
}

function upperBoundMarkers(markers: Array<{ time: number }>, maxTime: number): number {
  let low = 0;
  let high = markers.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (Number(markers[mid].time) <= maxTime) low = mid + 1;
    else high = mid;
  }
  return low;
}
