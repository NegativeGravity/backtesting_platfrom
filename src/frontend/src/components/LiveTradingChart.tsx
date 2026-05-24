import { useEffect, useRef, useState } from "react";
import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";

import type { LiveEvent } from "../api/liveApi";
import type { CandlePoint, ExecutionLogRecord } from "../types";
import { markerFromExecutionFill } from "../utils/chartMarkers";
import { toUnixSeconds } from "../utils/formatters";
import { ChartToolbar } from "./ChartToolbar";

interface LiveTradingChartProps {
  events: LiveEvent[];
}

export function LiveTradingChart({ events }: LiveTradingChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const screenshotRef = useRef<HTMLDivElement | null>(null);
  const priceChartRef = useRef<IChartApi | null>(null);
  const equityChartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const equitySeriesRef = useRef<ISeriesApi<"Line"> | null>(null);

  const [markersVisible, setMarkersVisible] = useState(true);
  const [equityVisible, setEquityVisible] = useState(true);
  const [markers, setMarkers] = useState<SeriesMarker<Time>[]>([]);

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    containerRef.current.innerHTML = "";

    const priceContainer = document.createElement("div");
    priceContainer.className = "chart-box chart-box-large";

    const equityContainer = document.createElement("div");
    equityContainer.className = "chart-box";

    containerRef.current.appendChild(priceContainer);
    containerRef.current.appendChild(equityContainer);

    const priceChart = createChart(priceContainer, {
      height: 360,
      layout: {
        textColor: "#d4d4d8",
        background: { color: "#0f172a" },
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.12)" },
        horzLines: { color: "rgba(148, 163, 184, 0.12)" },
      },
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
      crosshair: {
        mode: 1,
      },
    });

    const candleSeries = priceChart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    const equityChart = createChart(equityContainer, {
      height: 220,
      layout: {
        textColor: "#d4d4d8",
        background: { color: "#0f172a" },
      },
      grid: {
        vertLines: { color: "rgba(148, 163, 184, 0.12)" },
        horzLines: { color: "rgba(148, 163, 184, 0.12)" },
      },
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
      },
      rightPriceScale: {
        borderColor: "#334155",
      },
    });

    const equitySeries = equityChart.addLineSeries({
      title: "Live Equity",
      color: "#38bdf8",
      lineWidth: 2,
    }) as ISeriesApi<"Line">;

    priceChartRef.current = priceChart;
    equityChartRef.current = equityChart;
    candleSeriesRef.current = candleSeries;
    equitySeriesRef.current = equitySeries;

    return () => {
      priceChart.remove();
      equityChart.remove();

      priceChartRef.current = null;
      equityChartRef.current = null;
      candleSeriesRef.current = null;
      equitySeriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    const candles: CandlePoint[] = [];
    const nextMarkers: SeriesMarker<Time>[] = [];
    const equityByBotAndTime = new Map<string, Map<number, number>>();

    for (const event of [...events].reverse()) {
      if (event.event_type === "MARKET_BAR") {
        const payload = event.payload;
        const timestamp = String(payload.timestamp);

        candles.push({
          time: toUnixSeconds(timestamp),
          open: Number(payload.open),
          high: Number(payload.high),
          low: Number(payload.low),
          close: Number(payload.close),
        });
      }

      if (event.event_type === "ORDER_FILLED") {
        const payload = event.payload;

        const fill: ExecutionLogRecord = {
          side: String(payload.side),
          quantity: Number(payload.quantity),
          filled_bar_time: String(payload.fill_time),
          fill_price: Number(payload.fill_price),
          fee: Number(payload.fee),
          slippage_cost: Number(payload.slippage_cost),
          status: "FILLED",
          reason: String(payload.reason),
        };

        const marker = markerFromExecutionFill(fill);

        if (marker) {
          nextMarkers.push({
            ...marker,
            text: `${String(payload.worker_id ?? "worker")} ${marker.text}`,
          });
        }
      }

      if (event.event_type === "PORTFOLIO_UPDATED") {
          const payload = event.payload;
          const robotId = String(payload.robot_id ?? "unknown_robot");
          const timestamp = toUnixSeconds(String(payload.timestamp));

          if (!equityByBotAndTime.has(robotId)) {
            equityByBotAndTime.set(robotId, new Map<number, number>());
          }

          equityByBotAndTime.get(robotId)?.set(timestamp, Number(payload.equity));
      }
    }

    candleSeriesRef.current?.setData(
      candles.map((point) => ({
        time: point.time as Time,
        open: point.open,
        high: point.high,
        low: point.low,
        close: point.close,
      })),
    );

    setMarkers(nextMarkers);
    candleSeriesRef.current?.setMarkers(markersVisible ? nextMarkers : []);

    const firstEquitySeries = [...equityByBotAndTime.values()][0];

    equitySeriesRef.current?.setData(
      firstEquitySeries
        ? [...firstEquitySeries.entries()]
            .sort(([left], [right]) => left - right)
            .map(([time, value]) => ({
              time: time as Time,
              value,
            }))
        : [],
    );

    if (candles.length > 0) {
      priceChartRef.current?.timeScale().scrollToRealTime();
      equityChartRef.current?.timeScale().scrollToRealTime();
    }
  }, [events, markersVisible]);

  useEffect(() => {
    candleSeriesRef.current?.setMarkers(markersVisible ? markers : []);
  }, [markersVisible, markers]);

  useEffect(() => {
    equitySeriesRef.current?.applyOptions({
      visible: equityVisible,
    });

    if (containerRef.current) {
      const boxes = containerRef.current.querySelectorAll(".chart-box");
      const equityBox = boxes.item(1) as HTMLElement | null;

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
    priceChartRef.current?.timeScale().scrollToRealTime();
    equityChartRef.current?.timeScale().scrollToRealTime();
  }

  function downloadScreenshot() {
    const canvas = screenshotRef.current?.querySelector("canvas");

    if (!canvas) {
      return;
    }

    const url = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    link.href = url;
    link.download = "live-replay-chart.png";
    link.click();
  }

  return (
    <section ref={screenshotRef} className="chart-shell">
      <ChartToolbar
        title="Live Replay Chart"
        markersVisible={markersVisible}
        equityVisible={equityVisible}
        onFit={fitCharts}
        onZoomIn={() => zoomChart(0.7)}
        onZoomOut={() => zoomChart(1.3)}
        onGoToLatest={goToLatest}
        onToggleMarkers={() => setMarkersVisible((value) => !value)}
        onToggleEquity={() => setEquityVisible((value) => !value)}
        onScreenshot={downloadScreenshot}
      />
      <section ref={containerRef} className="charts live-charts" />
    </section>
  );
}