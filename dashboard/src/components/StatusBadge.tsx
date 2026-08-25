const STATUS_BADGE_CLASS: Record<string, string> = {
  ok: "badge--success",
  completed: "badge--success",
  queued: "badge--warning",
  running: "badge--info",
  cancelled: "badge--neutral",
  cancellation_requested: "badge--warning",
  failed: "badge--danger",
  error: "badge--danger",
  ready: "badge--success",
  not_ready: "badge--danger",
  detected: "badge--warning",
  not_detected: "badge--success",
  active: "badge--success",
  revoked: "badge--danger",
};

export function StatusBadge({ status }: { status: string }) {
  const cls = STATUS_BADGE_CLASS[status] || "badge--neutral";
  return (
    <span data-testid="status-badge" className={`badge ${cls}`}>
      {status}
    </span>
  );
}
