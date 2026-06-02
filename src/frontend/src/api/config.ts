const DEFAULT_DEV_API = 'http://127.0.0.1:8000';

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, '');
}

function readNumber(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function stripWsSuffix(value: string): string {
  return value.replace(/\/ws\/?$/, '');
}

export const API_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.PROD ? '/api' : DEFAULT_DEV_API),
);

const envWs = import.meta.env.VITE_WS_BASE_URL as string | undefined;

export const WS_BASE_URL = stripWsSuffix(trimTrailingSlash(
  envWs || (API_BASE_URL.startsWith('http') ? API_BASE_URL.replace(/^http/, 'ws').replace(/\/api$/, '') : ''),
));

export const REQUEST_TIMEOUT_MS = Math.max(readNumber(import.meta.env.VITE_REQUEST_TIMEOUT_MS, 600000), 600000);
export const JOB_STATUS_TIMEOUT_MS = Math.max(readNumber(import.meta.env.VITE_JOB_STATUS_TIMEOUT_MS, 60000), 60000);

export function buildHttpUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
}

export function buildWsUrl(pathOrUrl: string): string {
  if (pathOrUrl.startsWith('ws://') || pathOrUrl.startsWith('wss://')) {
    return pathOrUrl;
  }

  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) {
    return pathOrUrl.replace(/^http/, 'ws');
  }

  const normalizedPath = pathOrUrl.startsWith('/') ? pathOrUrl : `/${pathOrUrl}`;
  const base = stripWsSuffix(WS_BASE_URL);

  if (base.startsWith('ws://') || base.startsWith('wss://')) {
    return `${base}${normalizedPath}`;
  }

  if (base.startsWith('http://') || base.startsWith('https://')) {
    return `${base.replace(/^http/, 'ws')}${normalizedPath}`;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${normalizedPath}`;
}
