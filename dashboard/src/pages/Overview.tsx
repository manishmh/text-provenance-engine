import { useEffect, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type {
  MetricsResponse, AnalysisSummary, BenchmarkRunSummary,
  DetectorCapability, RobustnessSummary,
} from "../types/api";
import { ApiError } from "../api/client";
import { StatCard } from "../components/StatCard";
import { BarChart } from "../components/BarChart";
import { StatusBadge } from "../components/StatusBadge";
import { isBenchmarkRunList, isDetectorsResponse, isRobustnessReport } from "../utils/guards";
import type { ShellPage } from "../components/AppShell";

interface Extras {
  runs: BenchmarkRunSummary[] | null;
  runsError: string;
  detectors: DetectorCapability[] | null;
  detectorsError: string;
  robustness: RobustnessSummary | null;
  robustnessTotal: number;
  robustnessError: string;
}

const EMPTY_EXTRAS: Extras = {
  runs: null, runsError: "",
  detectors: null, detectorsError: "",
  robustness: null, robustnessTotal: 0, robustnessError: "",
};

export function OverviewPage({ onNavigate }: { onNavigate: (p: ShellPage) => void }) {
  const { client, usage } = useConfig();
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [recent, setRecent] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [extras, setExtras] = useState<Extras>(EMPTY_EXTRAS);
  const [extrasLoading, setExtrasLoading] = useState(false);

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

  // Second-phase enrichment: runs, detectors, robustness summary. Each
  // section degrades independently; failures never block the core page.
  useEffect(() => {
    if (!client || loading || error) return;
    let cancelled = false;
    setExtrasLoading(true);
    (async () => {
      const next: Extras = { ...EMPTY_EXTRAS };
      try {
        const r: unknown = await client.listBenchmarkRuns();
        if (isBenchmarkRunList(r)) next.runs = r.runs.slice(0, 5);
        else next.runsError = "Unexpected response shape";
      } catch (e: unknown) {
        next.runsError = e instanceof ApiError ? e.userMessage : "Failed to load runs";
      }
      try {
        const d: unknown = await client.listDetectors();
        if (isDetectorsResponse(d)) next.detectors = d.detectors;
        else next.detectorsError = "Unexpected response shape";
      } catch (e: unknown) {
        next.detectorsError = e instanceof ApiError ? e.userMessage : "Failed to load detectors";
      }
      try {
        const r: unknown = await client.getRobustnessResults();
        if (isRobustnessReport(r)) { next.robustness = r.summary; next.robustnessTotal = r.total_results; }
        else next.robustnessError = "Unexpected response shape";
      } catch (e: unknown) {
        next.robustnessError = e instanceof ApiError ? e.userMessage : "Failed to load summary";
      }
      if (!cancelled) { setExtras(next); setExtrasLoading(false); }
    })();
    return () => { cancelled = true; };
  }, [client, loading, error]);

  if (loading) return <div className="loading"><span className="spinner" /> Loading overview...</div>;
  if (error) return <div className="alert alert--error">{error}</div>;
  if (!metrics) return null;

  const detectorData = Object.entries(metrics.by_detector ?? {}).map(([label, value]) => ({ label, value }));
  const endpointData = Object.entries(metrics.by_endpoint ?? {}).map(([label, value]) => ({ label, value }));
  const safeRecent = recent ?? [];
  const successRate = metrics.total_requests > 0
    ? `${((metrics.successful_requests / metrics.total_requests) * 100).toFixed(1)}%`
    : "—";

  return (
    <div>
      <h2 className="page-header">Overview</h2>
      <p className="page-sub">Engine activity, recent work, and system status at a glance.</p>

      <div className="grid-stats">
        <StatCard label="Total Requests" value={metrics.total_requests} />
        <StatCard label="Success Rate" value={successRate} color="#166534" />
        <StatCard label="Failed" value={metrics.failed_requests} color="#991b1b" />
        <StatCard label="Characters Analyzed" value={metrics.total_characters.toLocaleString()} />
        <StatCard label="Avg Duration" value={metrics.avg_duration_ms != null ? `${metrics.avg_duration_ms.toFixed(1)}ms` : "—"} />
        <StatCard label="Engine" value={metrics.engine_version} />
      </div>

      <div className="grid-charts">
        <BarChart data={detectorData} title="Detector Usage" />
        <BarChart data={endpointData} title="Requests by Endpoint" />
      </div>

      <div className="home-grid">
        <div className="card">
          <div className="card__head">
            <h3 className="card__title">Recent Benchmark Runs</h3>
            <button className="btn btn--sm" onClick={() => onNavigate("benchmarks")}>Open Benchmarks</button>
          </div>
          <HomeRuns extras={extras} loading={extrasLoading} onNavigate={onNavigate} />
        </div>

        <div className="card">
          <div className="card__head">
            <h3 className="card__title">Detector Availability</h3>
            <button className="btn btn--sm" onClick={() => onNavigate("detectors")}>Open Detectors</button>
          </div>
          <HomeDetectors extras={extras} loading={extrasLoading} />
        </div>

        <div className="card">
          <div className="card__head">
            <h3 className="card__title">Robustness Summary</h3>
            <button className="btn btn--sm" onClick={() => onNavigate("robustness")}>Open Robustness</button>
          </div>
          <HomeRobustness extras={extras} loading={extrasLoading} />
        </div>

        <div className="card">
          <div className="card__head">
            <h3 className="card__title">Usage</h3>
            <button className="btn btn--sm" onClick={() => onNavigate("usage")}>Open Usage</button>
          </div>
          {usage ? (
            <dl className="kv">
              <div><dt>Requests</dt><dd>{usage.total_requests.toLocaleString()}</dd></div>
              <div><dt>Successful</dt><dd>{usage.successful_requests.toLocaleString()}</dd></div>
              <div><dt>Characters</dt><dd>{usage.total_characters.toLocaleString()}</dd></div>
            </dl>
          ) : (
            <div className="empty-state empty-state--compact">
              <div className="empty-state__text">Usage data unavailable</div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card__head">
          <h3 className="card__title">Recent Analyses</h3>
          <button className="btn btn--sm" onClick={() => onNavigate("history")}>Open History</button>
        </div>
        {safeRecent.length === 0 ? (
          <div className="empty-state" style={{ padding: "20px" }}>
            <div className="empty-state__text">No analyses yet</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>
              <button className="link-btn" onClick={() => onNavigate("analyze")}>Analyze your first text</button>
            </div>
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

function HomeRuns({ extras, loading, onNavigate }: {
  extras: Extras; loading: boolean; onNavigate: (p: ShellPage) => void;
}) {
  if (loading && extras.runs === null && !extras.runsError) {
    return <div className="loading"><span className="spinner spinner-sm" /> Loading runs...</div>;
  }
  if (extras.runsError) return <div className="empty-state empty-state--compact"><div className="empty-state__text">{extras.runsError}</div></div>;
  if (!extras.runs || extras.runs.length === 0) {
    return (
      <div className="empty-state empty-state--compact">
        <div className="empty-state__text">No benchmark runs yet</div>
        <div style={{ fontSize: "12px", marginTop: "4px" }}>
          <button className="link-btn" onClick={() => onNavigate("benchmarks")}>Start your first run</button>
        </div>
      </div>
    );
  }
  return (
    <ul className="home-list">
      {extras.runs.map((r) => (
        <li key={r.run_id}>
          <span className="mono home-list__id">{r.run_id.slice(0, 12)}</span>
          <span className="home-list__meta">{r.config.detector} · {new Date(r.created_at).toLocaleDateString()}</span>
          <StatusBadge status={r.status} />
        </li>
      ))}
    </ul>
  );
}

function HomeDetectors({ extras, loading }: { extras: Extras; loading: boolean }) {
  if (loading && extras.detectors === null && !extras.detectorsError) {
    return <div className="loading"><span className="spinner spinner-sm" /> Loading detectors...</div>;
  }
  if (extras.detectorsError) return <div className="empty-state empty-state--compact"><div className="empty-state__text">{extras.detectorsError}</div></div>;
  if (!extras.detectors || extras.detectors.length === 0) {
    return <div className="empty-state empty-state--compact"><div className="empty-state__text">No detectors reported</div></div>;
  }
  const byKind = new Map<string, number>();
  for (const d of extras.detectors) byKind.set(d.implementation_kind, (byKind.get(d.implementation_kind) ?? 0) + 1);
  return (
    <dl className="kv">
      <div><dt>Available</dt><dd>{extras.detectors.length} detectors</dd></div>
      {[...byKind.entries()].map(([kind, n]) => (
        <div key={kind}><dt>{kind}</dt><dd>{n}</dd></div>
      ))}
    </dl>
  );
}

function HomeRobustness({ extras, loading }: { extras: Extras; loading: boolean }) {
  if (loading && extras.robustness === null && !extras.robustnessError) {
    return <div className="loading"><span className="spinner spinner-sm" /> Loading summary...</div>;
  }
  if (extras.robustnessError) return <div className="empty-state empty-state--compact"><div className="empty-state__text">{extras.robustnessError}</div></div>;
  if (!extras.robustness) {
    return <div className="empty-state empty-state--compact"><div className="empty-state__text">No robustness results yet</div></div>;
  }
  const s = extras.robustness;
  return (
    <dl className="kv">
      <div><dt>Results</dt><dd>{extras.robustnessTotal.toLocaleString()}</dd></div>
      <div><dt>Detectors</dt><dd>{s.detectors.length}</dd></div>
      <div><dt>Transforms</dt><dd>{s.transforms.length}</dd></div>
      <div><dt>Lengths</dt><dd>{s.text_lengths.join(", ")}</dd></div>
    </dl>
  );
}
