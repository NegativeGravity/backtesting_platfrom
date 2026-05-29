import { buildHttpUrl, REQUEST_TIMEOUT_MS } from './config';

export class ApiError extends Error {
  status?: number;
  detail?: unknown;

  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function makeFriendlyMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    const detail = record.detail ?? record.message ?? record.error;

    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((item) => String(item)).join(', ');
  }

  if (typeof payload === 'string' && payload.trim()) return payload;
  return fallback;
}

async function readPayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? '';

  if (contentType.includes('application/json')) {
    return response.json().catch(() => null);
  }

  return response.text().catch(() => null);
}

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  const externalSignal = options.signal;
  const abortFromExternal = () => controller.abort();
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    externalSignal.addEventListener('abort', abortFromExternal, { once: true });
  }

  try {
    const response = await fetch(buildHttpUrl(path), {
      ...options,
      signal: controller.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    });

    if (!response.ok) {
      const payload = await readPayload(response);
      throw new ApiError(
        makeFriendlyMessage(payload, `Request failed with HTTP ${response.status}`),
        response.status,
        payload,
      );
    }

    if (response.status === 204) return undefined as T;
    return (await readPayload(response)) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError(`Request timed out after ${Math.round(REQUEST_TIMEOUT_MS / 1000)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener('abort', abortFromExternal);
  }
}