import type { QuotaInfo } from "../types/api";

export function WorkspaceUsagePage({ quota }: { quota: QuotaInfo | null }) {
  const used = quota?.used ?? 0;
  const limit = quota?.limit ?? 0;
  const percent = limit ? Math.min(100, (used / limit) * 100) : 0;
  return (
    <div data-testid="workspace-usage">
      <h2 className="page-header">Usage</h2>
      <p className="page-sub">Your current plan allowance resets at 00:00 UTC each day.</p>
      <div className="workspace-overview-grid">
        <div className="card">
          <span className="plan-badge">{quota?.plan ?? "Free"}</span>
          <h3 className="card__title">Daily analyses</h3>
          <p className="usage-meter__number">{quota ? `${quota.remaining} remaining` : "Loading…"}</p>
          <div className="usage-meter"><span style={{ width: `${percent}%` }} /></div>
          <p className="settings__hint">{used} of {limit} analyses used today</p>
        </div>
        <div className="card">
          <span className="plan-badge">Limit</span>
          <h3 className="card__title">Text size</h3>
          <p className="usage-meter__number">{quota ? quota.max_chars_per_analysis.toLocaleString() : "—"}</p>
          <p className="settings__hint">Maximum characters per analysis on your current plan.</p>
        </div>
      </div>
    </div>
  );
}
