export function formatMoney(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";

  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

export function formatPercent(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value * 100).toFixed(2)}%`;
}

export function formatNumber(value: unknown, digits = 4): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export function formatDateTime(value: unknown): string {
  if (typeof value !== "string") return "-";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

export function toUnixSeconds(value: string): number {
  return Math.floor(new Date(value).getTime() / 1000);
}