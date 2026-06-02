import { prepareChartPayload, type ChartWorkerRequest, type ChartWorkerResponse } from '../utils/chartPrep';

const ctx = self as DedicatedWorkerGlobalScope;

ctx.onmessage = (event: MessageEvent<ChartWorkerRequest>) => {
  const { id, chartData, trades } = event.data;
  const response: ChartWorkerResponse = {
    id,
    payload: prepareChartPayload(chartData, trades),
  };
  ctx.postMessage(response);
};

export {};
