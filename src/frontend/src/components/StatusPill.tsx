interface StatusPillProps {
  status: string;
  tone?: 'neutral' | 'good' | 'bad' | 'warn' | 'live';
}

export function StatusPill({ status, tone = 'neutral' }: StatusPillProps) {
  return <span className={`status-pill ${tone}`}>{status}</span>;
}