import { useEffect, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type { AnalysisSummary } from "../types/api";
import { ApiError } from "../api/client";

export function HistoryPage() {
  const { client } = useConfig();
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const limit = 20;

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      setError("");
      try {
        const res = await client.listAnalyses(limit, offset);
        if (!cancelled) {
          setAnalyses(res.analyses);
          setTotal(res.total);
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof ApiError ? e.userMessage : "Failed to load");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client, offset]);

  const loadDetail = async (id: string) => {
    if (!client) return;
    try {
      const r = await client.getAnalysis(id);
      setSelected(r);
    } catch (e: unknown) {
      setError(e instanceof ApiError ? e.userMessage : "Failed to load detail");
    }
  };

  return (
    <div>
      <h2 className="page-header">Analysis History</h2>

      {selected && (
        <div data-testid="analysis-detail" className="detail-card">
          <div className="detail-card__header">
            <h3 className="detail-card__title">Analysis Detail</h3>
            <button onClick={() => setSelected(null)} data-testid="btn-close-detail" className="close-btn">×</button>
          </div>

          {/* Render detector results nicely if available */}
          {(selected.results as unknown[])?.length ? (
            <div>
              {Object.entries(selected).filter(([k]) => !["results", "metadata"].includes(k)).map(([k, v]) => (
                <div key={k} style={{ display: "flex", gap: "8px", fontSize: "13px", marginBottom: "4px" }}>
                  <span style={{ color: "#6b7280", minWidth: 120 }}>{k}:</span>
                  <span className="mono" style={{ fontSize: "12px" }}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</span>
                </div>
              ))}
              <h4 style={{ fontSize: "13px", marginTop: "16px", marginBottom: "8px" }}>Detector Results</h4>
              <pre className="code-block" data-testid="analysis-detail-pre">{JSON.stringify(selected.results, null, 2)}</pre>
            </div>
          ) : (
            <pre className="code-block" data-testid="analysis-detail-pre">{JSON.stringify(selected, null, 2)}</pre>
          )}
        </div>
      )}

      {error && (
        <div className="alert alert--error" style={{ marginBottom: "16px" }}>
          {error}
          <button onClick={() => setError("")} className="close-btn" style={{ float: "right" }}>×</button>
        </div>
      )}

      <div className="card">
        {loading ? (
          <div className="loading"><span className="spinner" /> Loading analyses...</div>
        ) : analyses.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state__icon">📄</div>
            <div className="empty-state__text">No analyses found</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>Run an analysis to get started</div>
          </div>
        ) : (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Timestamp</th>
                  <th>Text Hash</th>
                  <th>Chars</th>
                  <th>Detectors</th>
                  <th>Status</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {analyses.map((a) => (
                  <tr key={a.analysis_id}>
                    <td className="mono" style={{ fontSize: "12px" }}>{a.analysis_id.slice(0, 8)}</td>
                    <td style={{ color: "#6b7280" }}>{new Date(a.timestamp).toLocaleString()}</td>
                    <td className="mono truncate" style={{ fontSize: "11px", color: "#9ca3af", maxWidth: 120 }} title={a.text_hash}>
                      {a.text_hash.slice(0, 12)}...
                    </td>
                    <td>{a.character_count.toLocaleString()}</td>
                    <td>{a.detector_count}</td>
                    <td>{a.status}</td>
                    <td>
                      <button onClick={() => loadDetail(a.analysis_id)} data-testid={`btn-detail-${a.analysis_id.slice(0, 8)}`} className="btn btn--sm">View</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="pagination">
              <span className="pagination__info">
                {total === 0 ? "No results" : `${offset + 1}–${Math.min(offset + limit, total)} of ${total}`}
              </span>
              <div className="pagination__controls">
                <button onClick={() => setOffset(Math.max(0, offset - limit))} disabled={offset === 0} data-testid="btn-prev" className="btn btn--sm">Previous</button>
                <button onClick={() => setOffset(offset + limit)} disabled={offset + limit >= total} data-testid="btn-next" className="btn btn--sm">Next</button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
