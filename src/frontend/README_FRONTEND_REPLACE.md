# Quant Frontend Redesign v2

This folder is intended to replace the existing `frontend/` directory completely.

## Local development

```bash
cd frontend
npm install
cp .env .env
npm run dev
```

Default development env talks to `http://127.0.0.1:8000` and `ws://127.0.0.1:8000/ws`.

## Docker / nginx

The included Dockerfile is production-oriented and defaults to:

```env
VITE_API_BASE_URL=/api
VITE_WS_BASE_URL=/ws
```

Nginx proxies `/api/` to `http://backend:8000/` and `/ws/` to `http://backend:8000/ws/`.

## Verified

`npm run build` was executed successfully after installing dependencies.

## Included product upgrades

- React Router direct run URLs: `/runs/:runId`
- TanStack Query for server state, cache, loading, retry, and polling
- Zustand for run tab and compare state
- Deployment-safe API/WS env config
- Request timeout and AbortController support
- WebSocket cleanup on unmount
- TradingView-style chart workspace using lightweight-charts
- Chart worker for sanitizing/preparing heavy chart payloads
- Replay loop based on requestAnimationFrame
- Marker budget modes to avoid clutter on large trade counts
- Virtualized trades table and event stream
- LiveTradingChart, LiveTradesTable, TradeBlotter, WorkerDiagnosticsPanel, RunComparePanel all wired into the app
- CSV, HTML, JSON client exports and server export links for CSV/Parquet/HTML endpoints
