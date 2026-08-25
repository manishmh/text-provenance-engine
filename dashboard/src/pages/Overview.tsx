import { useEffect, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type { MetricsResponse, AnalysisSummary } from "../types/api";
import { ApiError } from "../api/client";
import { StatCard } from "../components/StatCard";
import { BarChart } from "../components/BarChart";

export function OverviewPage() {
  const { client } = useConfig();
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [recent, setRecent] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const [m, a] = await Promise.all([client.metrics(), client.listAnalyses(10, 0)]);
        if (!cancelled) { setMetrics(m); setRecent(a.analyses); }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof ApiError ? e.userMessage : "Failed to load");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client]);

  if (loading) return <div className="loading"><span className="spinner" /> Loading overview...</div>;
  if (error) return <div className="alert alert--error">{error}</div>;
  if (!metrics) return null;

  const detectorData = Object.entries(metrics.by_detector ?? {}).map(([label, value]) => ({ label, value }));
  const endpointData = Object.entries(metrics.by_endpoint ?? {}).map(([label, value]) => ({ label, value }));
  const safeRecent = recent ?? [];

  return (
    <div>
      <h2 className="page-header">Overview</h2>

      <div className="grid-stats">
        <StatCard label="Total Requests" value={metrics.total_requests} />
        <StatCard label="Successful" value={metrics.successful_requests} color="#166534" />
        <StatCard label="Failed" value={metrics.failed_requests} color="#991b1b" />
        <StatCard label="Characters Analyzed" value={metrics.total_characters.toLocaleString()} />
        <StatCard label="Avg Duration" value={metrics.avg_duration_ms != null ? `${metrics.avg_duration_ms.toFixed(1)}ms` : "—"} />
        <StatCard label="Engine" value={metrics.engine_version} />
      </div>

      <div className="grid-charts">
        <BarChart data={detectorData} title="Detector Usage" />
        <BarChart data={endpointData} title="Requests by Endpoint" />
      </div>

      <div className="card">
        <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>Recent Analyses</h3>
        {safeRecent.length === 0 ? (
          <div className="empty-state" style={{ padding: "20px" }}>
            <div className="empty-state__icon">📊</div>
            <div className="empty-state__text">No analyses yet</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>Go to Analyze to get started</div>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Time</th>
                <th>Chars</th>
                <th>Detectors</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {safeRecent.map((a) => (
                <tr key={a.analysis_id}>
                  <td className="mono" style={{ fontSize: "12px" }}>{a.analysis_id.slice(0, 8)}</td>
                  <td style={{ color: "#6b7280" }}>{new Date(a.timestamp).toLocaleString()}</td>
                  <td>{a.character_count.toLocaleString()}</td>
                  <td>{a.detector_count}</td>
                  <td>{a.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
