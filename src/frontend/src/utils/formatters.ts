export function toNumber(value: unknown): number | null {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function formatNumber(value: unknown, digits = 2): string {
  const parsed = toNumber(value);
  if (parsed === null) return '-';
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
    minimumFractionDigits: Math.min(digits, 2),
  }).format(parsed);
}

export function formatMoney(value: unknown, currency = 'USD'): string {
  const parsed = toNumber(value);
  if (parsed === null) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 2,
  }).format(parsed);
}

export function formatPercent(value: unknown, digits = 2): string {
  const parsed = toNumber(value);
  if (parsed === null) return '-';
  const normalized = Math.abs(parsed) > 1.5 ? parsed / 100 : parsed;
  return `${(normalized * 100).toFixed(digits)}%`;
}

export function formatDateTime(value: unknown): string {
  if (!value) return '-';
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function formatDate(value: unknown): string {
  if (!value) return '-';
  const date = new Date(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: '2-digit' }).format(date);
}

export function toUnixSeconds(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.floor(value > 10_000_000_000 ? value / 1000 : value);
  if (typeof value === 'string' && value.trim()) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return Math.floor(numeric > 10_000_000_000 ? numeric / 1000 : numeric);
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return Math.floor(parsed / 1000);
  }
  return null;
}

export function getRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
