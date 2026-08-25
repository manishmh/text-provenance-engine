import { useEffect, useState, useRef, useCallback } from "react";
import { useConfig } from "../hooks/useConfig";
import type { JobSummary, JobResponse } from "../types/api";
import { ApiError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

const ACTIVE_STATUSES = new Set(["queued", "running", "cancellation_requested"]);
const POLL_INTERVAL = 3000;

export function JobsPage() {
  const { client } = useConfig();
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [detail, setDetail] = useState<JobResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const limit = 20;

  const refresh = useCallback(async () => {
    if (!client) return;
    try {
      const res = await client.listJobs(limit, offset);
      setJobs(res.jobs);
      setTotal(res.total);
      setError("");
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
    }
  }, [client, offset, limit]);

  // Initial load
  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      setLoading(true);
      await refresh();
      if (!cancelled) setLoading(false);
    })();
    return () => { cancelled = true; };
  }, [client, refresh]);

  // Auto-poll active jobs
  useEffect(() => {
    if (!client) return;
    const hasActive = jobs.some((j) => ACTIVE_STATUSES.has(j.status));
    if (hasActive && !pollingRef.current) {
      pollingRef.current = setInterval(() => {
        refresh();
      }, POLL_INTERVAL);
    } else if (!hasActive && pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [client, jobs, refresh]);

  const viewDetail = async (jobId: string) => {
    if (!client) return;
    try {
      const r = await client.getJob(jobId);
      setDetail(r);
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
    }
  };

  const handleCancel = async (jobId: string) => {
    if (!client) return;
    try {
      const r = await client.cancelJob(jobId);
      setMessage(r.message);
      await refresh();
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
      else setMessage(e instanceof Error ? e.message : "Cancel failed");
    }
  };

  const handleRetry = async (jobId: string) => {
    if (!client) return;
    const text = prompt("Enter text to re-analyze:");
    if (!text) return;
    try {
      const r = await client.retryJob(jobId, text);
      setMessage(r.message);
      await refresh();
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
      else setMessage(e instanceof Error ? e.message : "Retry failed");
    }
  };

  const hasActiveJobs = jobs.some((j) => ACTIVE_STATUSES.has(j.status));

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "20px" }}>
        <h2 className="page-header" style={{ marginBottom: 0 }}>Jobs</h2>
        {hasActiveJobs && (
          <span style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#1e40af" }}>
            <span className="spinner spinner-sm" /> Auto-updating
          </span>
        )}
      </div>

      {message && (
        <div data-testid="jobs-message" className="alert alert--info">
          {message}
          <button onClick={() => setMessage("")} className="close-btn" style={{ float: "right" }}>×</button>
        </div>
      )}

      {error && (
        <div data-testid="jobs-error" className="alert alert--error">
          {error}
          <button onClick={() => setError("")} className="close-btn" style={{ float: "right" }}>×</button>
        </div>
      )}

      {detail && (
        <div data-testid="job-detail" className="detail-card">
          <div className="detail-card__header">
            <h3 className="detail-card__title">
              Job Detail — <StatusBadge status={detail.status} />
            </h3>
            <button onClick={() => setDetail(null)} data-testid="btn-close-job-detail" className="close-btn">×</button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", fontSize: "13px", marginBottom: "16px" }}>
            <div><span style={{ color: "#6b7280" }}>Job ID: </span><span className="mono">{detail.job_id.slice(0, 12)}...</span></div>
            <div><span style={{ color: "#6b7280" }}>Created: </span>{new Date(detail.created_at).toLocaleString()}</div>
            <div><span style={{ color: "#6b7280" }}>Duration: </span>{detail.duration_ms != null ? `${detail.duration_ms.toFixed(0)}ms` : "—"}</div>
            <div><span style={{ color: "#6b7280" }}>Chars: </span>{detail.character_count.toLocaleString()}</div>
            <div><span style={{ color: "#6b7280" }}>Detectors: </span>{detail.detectors.join(", ")}</div>
            <div><span style={{ color: "#6b7280" }}>Retries: </span>{detail.retry_count}</div>
          </div>

          {detail.error_message && (
            <div className="alert alert--error" style={{ marginBottom: "12px" }}>{detail.error_message}</div>
          )}

          {detail.result && (
            <div>
              <h4 style={{ fontSize: "13px", marginBottom: "8px" }}>Result</h4>
              <pre className="code-block" data-testid="job-detail-pre">{JSON.stringify(detail.result, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      <div className="card">
        {loading ? (
          <div className="loading"><span className="spinner" /> Loading jobs...</div>
        ) : jobs.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state__icon">📋</div>
            <div className="empty-state__text">No jobs found</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>Use Analyze → Async to submit background jobs</div>
          </div>
        ) : (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Duration</th>
                  <th>Retries</th>
                  <th>Error</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((j) => (
                  <tr key={j.job_id} data-testid={`job-row-${j.status}`}>
                    <td className="mono" style={{ fontSize: "12px" }}>{j.job_id.slice(0, 8)}</td>
                    <td><StatusBadge status={j.status} /></td>
                    <td style={{ color: "#6b7280" }}>{new Date(j.created_at).toLocaleString()}</td>
                    <td className="mono">{j.duration_ms != null ? `${j.duration_ms.toFixed(0)}ms` : "—"}</td>
                    <td>{j.retry_count}</td>
                    <td className="truncate" style={{ maxWidth: 150, fontSize: "12px", color: "#991b1b" }} title={j.error_message || ""}>
                      {j.error_message?.slice(0, 40) || "—"}
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: "4px" }}>
                        <button onClick={() => viewDetail(j.job_id)} data-testid={`btn-view-${j.job_id.slice(0, 8)}`} className="btn btn--sm">View</button>
                        {ACTIVE_STATUSES.has(j.status) && (
                          <button onClick={() => handleCancel(j.job_id)} data-testid={`btn-cancel-${j.job_id.slice(0, 8)}`} className="btn btn--sm btn--danger">Cancel</button>
                        )}
                        {j.status === "failed" && (
                          <button onClick={() => handleRetry(j.job_id)} data-testid={`btn-retry-${j.job_id.slice(0, 8)}`} className="btn btn--sm btn--info">Retry</button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="pagination">
              <span className="pagination__info">
                {offset + 1}–{Math.min(offset + limit, total)} of {total}
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
