import { useEffect, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type { AnalysisSummary, QuotaInfo } from "../types/api";
import type { ShellPage } from "../components/AppShell";

export function FreeOverviewPage({ quota, onNavigate }: { quota: QuotaInfo | null; onNavigate: (page: ShellPage) => void }) {
  const { client } = useConfig();
  const [recent, setRecent] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!client) { setLoading(false); return; }
    let cancelled = false;
    client.listAnalyses(5, 0).then((response) => {
      if (!cancelled) setRecent(response.analyses);
    }).catch(() => undefined).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, [client]);

  return (
    <div data-testid="free-overview">
      <div className="workspace-page-head">
        <div><p className="site__eyebrow">Free workspace</p><h2 className="page-header">Your provenance workspace</h2></div>
        <button className="btn btn--primary" onClick={() => onNavigate("analyze")}>Analyze text</button>
      </div>
      <div className="workspace-overview-grid">
        <div className="card">
          <span className="plan-badge">Free</span>
          <h3 className="card__title">Today’s allowance</h3>
          <p className="usage-meter__number">{quota ? `${quota.remaining} remaining` : "Loading…"}</p>
          {quota && <><div className="usage-meter"><span style={{ width: `${Math.min(100, (quota.used / Math.max(1, quota.limit)) * 100)}%` }} /></div><p className="settings__hint">{quota.used} of {quota.limit} analyses used today · {quota.max_chars_per_analysis.toLocaleString()} characters per analysis</p></>}
          <button className="link-btn" onClick={() => onNavigate("usage")}>View usage</button>
        </div>
        <div className="card">
          <span className="plan-badge plan-badge--pro">Pro workspace</span>
          <h3 className="card__title">Need advanced workflows?</h3>
          <p className="settings__hint">Pro adds configured reports, robustness, benchmarks, detector tooling, jobs, and API access where enabled.</p>
          <a className="btn" href="#/pricing">Compare plans</a>
        </div>
      </div>
      <div className="card">
        <div className="card__head"><h3 className="card__title">Recent analyses</h3><button className="btn btn--sm" onClick={() => onNavigate("history")}>View history</button></div>
        {loading ? <div className="loading"><span className="spinner spinner-sm" /> Loading history…</div> : recent.length === 0 ? (
          <div className="empty-state"><div className="empty-state__text">No saved analyses yet</div><button className="link-btn" onClick={() => onNavigate("analyze")}>Analyze your first text</button></div>
        ) : <div className="table-wrap"><table className="table"><thead><tr><th>Created</th><th>Characters</th><th>Detectors</th><th>Status</th></tr></thead><tbody>{recent.map((item) => <tr key={item.analysis_id}><td>{new Date(item.timestamp).toLocaleString()}</td><td>{item.character_count.toLocaleString()}</td><td>{item.detector_count}</td><td>{item.status}</td></tr>)}</tbody></table></div>}
      </div>
    </div>
  );
}
