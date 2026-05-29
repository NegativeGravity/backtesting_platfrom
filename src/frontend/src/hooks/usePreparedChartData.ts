import { useEffect, useRef, useState } from 'react';
import type { ChartDataResponse, TradeRecord } from '../types';
import {
  emptyPreparedChartData,
  prepareChartPayload,
  type ChartWorkerResponse,
  type PreparedChartData,
} from '../utils/chartPrep';

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export function usePreparedChartData(chartData: ChartDataResponse | null, trades: TradeRecord[]): PreparedChartData {
  const [prepared, setPrepared] = useState<PreparedChartData>(() => emptyPreparedChartData());
  const workerRef = useRef<Worker | null>(null);

  useEffect(() => {
    if (!chartData) {
      workerRef.current?.terminate();
      workerRef.current = null;
      setPrepared(emptyPreparedChartData());
      return;
    }

    if (typeof Worker === 'undefined') {
      setPrepared(prepareChartPayload(chartData, trades));
      return;
    }

    const id = makeId();
    workerRef.current?.terminate();
    const worker = new Worker(new URL('../workers/chartWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (event: MessageEvent<ChartWorkerResponse>) => {
      if (event.data.id === id) setPrepared(event.data.payload);
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = null;
    };

    worker.onerror = () => {
      setPrepared(prepareChartPayload(chartData, trades));
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = null;
    };

    worker.postMessage({ id, chartData, trades });

    return () => {
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = null;
    };
  }, [chartData, trades]);

  return prepared;
}
