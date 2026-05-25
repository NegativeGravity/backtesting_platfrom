import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

import type { ChartDataResponse, TradeRecord } from "../types";
import {
  historicalMarkersFromTrades,
  markersFromApiMarkers,
} from "../utils/chartMarkers";
import { ChartToolbar } from "./ChartToolbar";

interface ChartPanelProps {
  chartData: ChartDataResponse | null;
  trades: TradeRecord[];
}

interface SafeCandle {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface SafeLinePoint {
  time: Time;
  value: number;
}

const INITIAL_REPLAY_BARS = 120;

export function ChartPanel({ chartData, trades }: ChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const priceChartRef = useRef<IChartApi | null>(null);
  const equityChartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const benchmarkSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const [markersVisible, setMarkersVisible] = useState(true);
  const [equityVisible, setEquityVisible] = useState(true);
  const [chartError, setChartError] = useState<string | null>(null);
  const [isReplaying, setIsReplaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState(6);
  const [replayIndex, setReplayIndex] = useState(INITIAL_REPLAY_BARS);

  const safeData = useMemo(() => {
    const candles = sanitizeCandles(chartData?.candles);
    const equityCurve = sanitizeLinePoints(chartData?.equity_curve);
    const benchmarkCurve = sanitizeLinePoints(chartData?.benchmark_curve);
    const apiMarkers = markersFromApiMarkers(chartData?.markers ?? []);
    const tradeMarkers = historicalMarkersFromTrades(trades ?? []);
    const markers = sanitizeMarkers(tradeMarkers.length > 0 ? tradeMarkers : apiMarkers);

    return {
      candles,
      equityCurve,
      benchmarkCurve,
      markers,
    };
  }, [chartData, trades]);

  const visibleIndex = Math.min(Math.max(replayIndex, 1), safeData.candles.length);
  const replayProgress =
    safeData.candles.length > 0 ? (visibleIndex / safeData.candles.length) * 100 : 0;

  useEffect(() => {
    setReplayIndex(Math.min(INITIAL_REPLAY_BARS, Math.max(1, safeData.candles.length)));
    setIsReplaying(false);
  }, [safeData.candles.length]);

  useEffect(() => {
    if (!isReplaying || safeData.candles.length === 0) return;

    const interval = window.setInterval(() => {
      setReplayIndex((current) => {
        const next = Math.min(current + replaySpeed, safeData.candles.length);
        if (next >= safeData.candles.length) {
          window.setTimeout(() => setIsReplaying(false), 0);
        }
        return next;
      });
    }, 120);

    return () => window.clearInterval(interval);
  }, [isReplaying, replaySpeed, safeData.candles.length]);

  useEffect(() => {
    if (!containerRef.current || !chartData) {
      return;
    }

    if (safeData.candles.length === 0) {
      setChartError("No valid candle data found for this report.");
      return;
    }

    setChartError(null);
    containerRef.current.innerHTML = "";

    const priceContainer = document.createElement("div");
    priceContainer.className = "chart-box chart-box-large";

    const equityContainer = document.createElement("div");
    equityContainer.className = "chart-box chart-equity-box";

    containerRef.current.appendChild(priceContainer);
    containerRef.current.appendChild(equityContainer);

    const priceChart = createChart(priceContainer, {
      height: 520,
      autoSize: true,
      layout: {
        textColor: "#e6faff",
        background: { color: "#050b16" },
      },
      grid: {
        vertLines: { color: "rgba(34, 211, 238, 0.09)" },
        horzLines: { color: "rgba(34, 211, 238, 0.09)" },
      },
      rightPriceScale: {
        borderColor: "rgba(34, 211, 238, 0.26)",
        scaleMargins: {
          top: 0.08,
          bottom: 0.18,
        },
      },
      timeScale: {
        borderColor: "rgba(34, 211, 238, 0.26)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 14,
        barSpacing: 8,
        minBarSpacing: 0.5,
        lockVisibleTimeRangeOnResize: true,
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: "rgba(165, 243, 252, 0.70)",
          width: 1,
          style: 3,
          labelBackgroundColor: "#6d28d9",
        },
        horzLine: {
          color: "rgba(165, 243, 252, 0.70)",
          width: 1,
          style: 3,
          labelBackgroundColor: "#6d28d9",
        },
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    const candleSeries = priceChart.addCandlestickSeries({
      upColor: "#22d3ee",
      downColor: "#fb4567",
      borderUpColor: "#67e8f9",
      borderDownColor: "#ff6b87",
      wickUpColor: "#a5f3fc",
      wickDownColor: "#fecdd3",
      priceFormat: {
        type: "price",
        precision: 2,
        minMove: 0.01,
      },
    });

    const equityChart = createChart(equityContainer, {
      height: 260,
      autoSize: true,
      layout: {
        textColor: "#e6faff",
        background: { color: "#050b16" },
      },
      grid: {
        vertLines: { color: "rgba(34, 211, 238, 0.09)" },
        horzLines: { color: "rgba(34, 211, 238, 0.09)" },
      },
      rightPriceScale: {
        borderColor: "rgba(34, 211, 238, 0.26)",
      },
      timeScale: {
        borderColor: "rgba(34, 211, 238, 0.26)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 14,
        barSpacing: 8,
        minBarSpacing: 0.5,
        lockVisibleTimeRangeOnResize: true,
      },
    });

    const equitySeries = equityChart.addLineSeries({
      title: "Strategy Equity",
      color: "#67e8f9",
      lineWidth: 2,
    }) as ISeriesApi<"Line">;

    const benchmarkSeries = equityChart.addLineSeries({
      title: "Buy & Hold",
      color: "#fbbf24",
      lineWidth: 2,
    }) as ISeriesApi<"Line">;

    priceChartRef.current = priceChart;
    equityChartRef.current = equityChart;
    candleSeriesRef.current = candleSeries;
    equitySeriesRef.current = equitySeries;
    benchmarkSeriesRef.current = benchmarkSeries;

    return () => {
      priceChart.remove();
      equityChart.remove();

      priceChartRef.current = null;
      equityChartRef.current = null;
      candleSeriesRef.current = null;
      equitySeriesRef.current = null;
      benchmarkSeriesRef.current = null;
    };
  }, [chartData, safeData.candles.length]);

  useEffect(() => {
    if (!candleSeriesRef.current || safeData.candles.length === 0) {
      return;
    }

    const visibleCandles = safeData.candles.slice(0, visibleIndex);
    const lastTime = Number(visibleCandles.at(-1)?.time ?? 0);
    const visibleEquity = safeData.equityCurve.filter((point) => Number(point.time) <= lastTime);
    const visibleBenchmark = safeData.benchmarkCurve.filter((point) => Number(point.time) <= lastTime);
    const visibleMarkers = safeData.markers.filter((marker) => Number(marker.time) <= lastTime);

    try {
      candleSeriesRef.current.setData(visibleCandles);
      equitySeriesRef.current?.setData(visibleEquity);
      benchmarkSeriesRef.current?.setData(visibleBenchmark);
      candleSeriesRef.current.setMarkers(markersVisible ? visibleMarkers : []);
      priceChartRef.current?.timeScale().scrollToPosition(4, false);
      equityChartRef.current?.timeScale().scrollToPosition(4, false);
    } catch (error) {
      console.error("Failed to render chart data", error);
      setChartError(error instanceof Error ? error.message : "Failed to render chart.");
    }
  }, [
    visibleIndex,
    markersVisible,
    safeData.candles,
    safeData.equityCurve,
    safeData.benchmarkCurve,
    safeData.markers,
  ]);

  useEffect(() => {
    equitySeriesRef.current?.applyOptions({ visible: equityVisible });
    benchmarkSeriesRef.current?.applyOptions({ visible: equityVisible });

    if (containerRef.current) {
      const equityBox = containerRef.current.querySelector(".chart-equity-box") as HTMLElement | null;
      if (equityBox) {
        equityBox.style.display = equityVisible ? "block" : "none";
      }
    }
  }, [equityVisible]);

  function fitCharts() {
    priceChartRef.current?.timeScale().fitContent();
    equityChartRef.current?.timeScale().fitContent();
  }

  function zoomChart(multiplier: number) {
    const chart = priceChartRef.current;
    if (!chart) return;

    const range = chart.timeScale().getVisibleLogicalRange();
    if (!range) return;

    const center = (range.from + range.to) / 2;
    const halfSize = ((range.to - range.from) / 2) * multiplier;

    chart.timeScale().setVisibleLogicalRange({
      from: center - halfSize,
      to: center + halfSize,
    });
  }

  function goToLatest() {
    setReplayIndex(safeData.candles.length);
    priceChartRef.current?.timeScale().scrollToRealTime();
    equityChartRef.current?.timeScale().scrollToRealTime();
  }

  function resetReplay() {
    setIsReplaying(false);
    setReplayIndex(Math.min(INITIAL_REPLAY_BARS, Math.max(1, safeData.candles.length)));
    fitCharts();
  }

  function stepReplay() {
    setReplayIndex((current) => Math.min(current + 1, safeData.candles.length));
  }

  function downloadScreenshot() {
    const chart = priceChartRef.current;
    const canvas = chart?.takeScreenshot();

    if (!canvas) {
      return;
    }

    const url = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    link.download = "backtest-chart.png";
    link.click();
  }

  if (!chartData) {
    return (
      <section className="empty-chart">
        <h2>No chart loaded</h2>
        <p>Select a report or run a backtest to visualize candles and trades.</p>
      </section>
    );
  }

  return (
    <section className="chart-shell">
      <ChartToolbar
        title="Backtest Replay Chart"
        markersVisible={markersVisible}
        equityVisible={equityVisible}
        isReplaying={isReplaying}
        replayProgress={replayProgress}
        replaySpeed={replaySpeed}
        onFit={fitCharts}
        onZoomIn={() => zoomChart(0.7)}
        onZoomOut={() => zoomChart(1.3)}
        onGoToLatest={goToLatest}
        onToggleMarkers={() => setMarkersVisible((value) => !value)}
        onToggleEquity={() => setEquityVisible((value) => !value)}
        onScreenshot={downloadScreenshot}
        onReplayPlayPause={() => setIsReplaying((value) => !value)}
        onReplayReset={resetReplay}
        onReplayStep={stepReplay}
        onReplaySpeedChange={setReplaySpeed}
      />

      <div className="replay-progress-track">
        <div className="replay-progress-bar" style={{ width: `${replayProgress}%` }} />
      </div>

      {chartError && (
        <div className="error-box">
          <strong>Chart data was invalid.</strong>
          <p>{chartError}</p>
        </div>
      )}

      <section ref={containerRef} className="charts" />
    </section>
  );
}

function sanitizeCandles(points: ChartDataResponse["candles"] | undefined): SafeCandle[] {
  const byTime = new Map<number, SafeCandle>();

  for (const point of points ?? []) {
    const time = toSafeUnixTime(point?.time);
    const open = toFiniteNumber(point?.open);
    const high = toFiniteNumber(point?.high);
    const low = toFiniteNumber(point?.low);
    const close = toFiniteNumber(point?.close);

    if (time === null || open === null || high === null || low === null || close === null) {
      continue;
    }

    const normalizedHigh = Math.max(high, open, close, low);
    const normalizedLow = Math.min(low, open, close, high);
    const existing = byTime.get(time);

    if (!existing) {
      byTime.set(time, {
        time: time as Time,
        open,
        high: normalizedHigh,
        low: normalizedLow,
        close,
      });
      continue;
    }

    byTime.set(time, {
      time: time as Time,
      open: existing.open,
      high: Math.max(existing.high, normalizedHigh),
      low: Math.min(existing.low, normalizedLow),
      close,
    });
  }

  return [...byTime.values()].sort((left, right) => Number(left.time) - Number(right.time));
}

function sanitizeLinePoints(points: ChartDataResponse["equity_curve"] | undefined): SafeLinePoint[] {
  const byTime = new Map<number, SafeLinePoint>();

  for (const point of points ?? []) {
    const time = toSafeUnixTime(point?.time);
    const value = toFiniteNumber(point?.value);

    if (time === null || value === null) {
      continue;
    }

    byTime.set(time, {
      time: time as Time,
      value,
    });
  }

  return [...byTime.values()].sort((left, right) => Number(left.time) - Number(right.time));
}

// ----------------------------------------------------------------------------
// FIXED: Using reduce prevents TS2322, TS2677, and TS18047 by building a
// perfectly typed array of SeriesMarker<Time> without nulls or map/filter chaining.
// ----------------------------------------------------------------------------
function sanitizeMarkers(inputMarkers: SeriesMarker<Time>[]): SeriesMarker<Time>[] {
  return inputMarkers
    .reduce<SeriesMarker<Time>[]>((validMarkers, marker) => {
      const time = toSafeUnixTime(marker.time as number);

      // Only push valid elements, completely avoiding 'null' in the array
      if (time !== null) {
        validMarkers.push({
          ...marker,
          time: time as Time,
          // Properly type-cast text to 'string | undefined' to satisfy lightweight-charts
          text: marker.text !== undefined ? String(marker.text) : undefined,
        });
      }

      return validMarkers;
    }, [])
    .sort((left, right) => {
      const timeDiff = Number(left.time) - Number(right.time);
      if (timeDiff !== 0) return timeDiff;

      // left and right are guaranteed to be valid markers here
      return String(left.text ?? "").localeCompare(String(right.text ?? ""));
    });
}

function toSafeUnixTime(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Math.floor(value);
  }

  if (typeof value === "string" && value.trim().length > 0) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return Math.floor(numeric);
    }

    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) {
      return Math.floor(parsed / 1000);
    }
  }

  return null;
}

function toFiniteNumber(value: unknown): number | null {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}