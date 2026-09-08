import { useEffect, useMemo, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type {
  AggregatedRobustness,
  ComparisonGroup,
  ComparisonResponse,
  RobustnessReportResponse,
} from "../types/api";
import { ApiError } from "../api/client";
import { isComparisonResponse, isRobustnessReport } from "../utils/guards";
import { RobustnessDisclaimer } from "../components/Disclaimer";
import { StatCard } from "../components/StatCard";

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

function ci(lo: number, hi: number): string {
  return `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
}

function fmtDelta(x: number | null): string {
  return x === null || x === undefined ? "n/a" : `${x >= 0 ? "+" : ""}${x.toFixed(4)}`;
}

function cellKey(detector: string, config: string, transform: string): string {
  return `${detector}||${config}||${transform}`;
}

function groupLabel(g: ComparisonGroup): string {
  for (const k of ["model_config", "transform", "category", "length"]) {
    const v = g[k];
    if (v !== undefined && v !== null) return String(v);
  }
  return "all";
}

function ComparisonTable(props: { title: string; groups: ComparisonGroup[]; testId: string }) {
  const { title, groups, testId } = props;
  if (groups.length === 0) return null;
  return (
    <div className="card" style={{ marginTop: "16px" }} data-testid={testId}>
      <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>{title}</h3>
      <table className="table">
        <thead>
          <tr>
            <th>Group</th>
            <th style={{ textAlign: "right" }}>Baseline rate</th>
            <th style={{ textAlign: "right" }}>Transformed rate</th>
            <th style={{ textAlign: "right" }}>Robustness [95% CI]</th>
            <th style={{ textAlign: "right" }}>Samples</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g, i) => {
            const row = g as unknown as {
              baseline_rate?: number; transformed_rate?: number;
              robustness_rate: number; robustness_ci_low: number;
              robustness_ci_high: number; total_samples: number;
            };
            return (
              <tr key={i}>
                <td className="mono" style={{ fontSize: "12px" }}>{groupLabel(g)}</td>
                <td style={{ textAlign: "right" }}>{row.baseline_rate !== undefined ? pct(row.baseline_rate) : "—"}</td>
                <td style={{ textAlign: "right" }}>{row.transformed_rate !== undefined ? pct(row.transformed_rate) : "—"}</td>
                <td style={{ textAlign: "right" }}>
                  {pct(row.robustness_rate)}{" "}
                  <span style={{ color: "var(--color-text-secondary)", fontSize: "12px" }}>
                    {ci(row.robustness_ci_low, row.robustness_ci_high)}
                  </span>
                </td>
                <td style={{ textAlign: "right" }}>{row.total_samples}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RobustnessPage() {
  const { client } = useConfig();
  const [report, setReport] = useState<RobustnessReportResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [malformed, setMalformed] = useState("");

  const [detector, setDetector] = useState("");
  const [config, setConfig] = useState("");
  const [category, setCategory] = useState("");
  const [length, setLength] = useState("");
  const [selected, setSelected] = useState("");

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const [r, c]: unknown[] = await Promise.all([
          client.getRobustnessResults(),
          client.getRobustnessComparison(),
        ]);
        if (cancelled) return;
        if (!isRobustnessReport(r)) {
          setMalformed("robustness results");
        } else if (!isComparisonResponse(c)) {
          setMalformed("robustness comparison");
        } else {
          setReport(r);
          setComparison(c);
        }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof ApiError ? e.userMessage : "Failed to load robustness data");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client]);

  const categoryOf = useMemo(() => {
    const map = report?.category_map ?? {};
    return (t: string) => map[t] ?? "other";
  }, [report]);

  const filtered: AggregatedRobustness[] = useMemo(() => {
    if (!report) return [];
    const lenNum = length === "" ? null : Number(length);
    return report.aggregated.filter((a) => {
      if (detector && a.detector_name !== detector) return false;
      if (config && a.config_identifier !== config) return false;
      if (category && categoryOf(a.transform_name) !== category) return false;
      if (lenNum !== null && !a.text_lengths.includes(lenNum)) return false;
      return true;
    });
  }, [report, detector, config, category, length, categoryOf]);

  const pairs = useMemo(() => {
    const seen = new Map<string, { detector: string; config: string }>();
    for (const a of filtered) {
      const k = `${a.detector_name}||${a.config_identifier}`;
      if (!seen.has(k)) seen.set(k, { detector: a.detector_name, config: a.config_identifier });
    }
    return [...seen.values()];
  }, [filtered]);

  const transforms = useMemo(() => {
    const set = new Set<string>();
    for (const a of filtered) set.add(a.transform_name);
    return [...set].sort();
  }, [filtered]);

  const byCell = useMemo(() => {
    const m = new Map<string, AggregatedRobustness>();
    for (const a of filtered) m.set(cellKey(a.detector_name, a.config_identifier, a.transform_name), a);
    return m;
  }, [filtered]);

  useEffect(() => {
    if (!byCell.has(selected)) {
      const first = byCell.keys().next();
      setSelected(first.done ? "" : first.value);
    }
  }, [byCell, selected]);

  const selectedCell = selected ? byCell.get(selected) : undefined;
  const totalSamples = filtered.reduce((s, a) => s + a.total_samples, 0);

  if (loading) return <div className="loading"><span className="spinner" /> Loading robustness data...</div>;
  if (error) return <div className="alert alert--error" data-testid="robustness-error">{error}</div>;
  if (malformed || !report || !comparison) {
    return (
      <div className="alert alert--error" data-testid="robustness-malformed">
        The server returned robustness data in an unrecognized format
        {malformed ? ` (${malformed})` : ""}. Please check the server version.
      </div>
    );
  }
  if (report.total_results === 0) {
    return (
      <div>
        <h2 className="page-header">Robustness</h2>
        <RobustnessDisclaimer />
        <div className="empty-state" data-testid="robustness-empty">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No benchmark results stored on the server</div>
          <div style={{ fontSize: "12px", marginTop: "4px" }}>
            Run <span className="mono">provenance benchmark robustness --out-dir</span> and point
            PROVENANCE_ROBUSTNESS_DIR at the output directory.
          </div>
        </div>
      </div>
    );
  }

  const summary = report.summary;
  const configOptions = summary.configs.filter((c) => !detector || filtered.some((a) => a.config_identifier === c));
  const resetFilters = () => { setDetector(""); setConfig(""); setCategory(""); setLength(""); };

  return (
    <div>
      <h2 className="page-header">Robustness</h2>
      <RobustnessDisclaimer />

      {report.warnings.length > 0 && (
        <div className="alert alert--warning" data-testid="robustness-warnings">
          <strong>{report.warnings.length} artifact file(s) could not be loaded:</strong>
          <ul style={{ margin: "8px 0 0", paddingLeft: "18px" }}>
            {report.warnings.map((w, i) => (
              <li key={i}><span className="mono">{w.file}</span> — {w.error_type}: {w.message}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid-stats" data-testid="robustness-summary">
        <StatCard label="Result cells" value={report.total_results} />
        <StatCard label="Detectors" value={summary.detectors.length} />
        <StatCard label="Configs" value={summary.configs.length} />
        <StatCard label="Transforms" value={summary.transforms.length} />
        <StatCard label="Total samples" value={totalSamples.toLocaleString()} />
      </div>

      <div className="card" style={{ marginTop: "16px" }}>
        <div className="filter-bar" data-testid="robustness-filters">
          <label>Detector
            <select data-testid="filter-detector" value={detector} onChange={(e) => { setDetector(e.target.value); setConfig(""); }}>
              <option value="">All detectors</option>
              {summary.detectors.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
          <label>Model / config
            <select data-testid="filter-config" value={config} onChange={(e) => setConfig(e.target.value)}>
              <option value="">All configs</option>
              {configOptions.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label>Transform category
            <select data-testid="filter-category" value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">All categories</option>
              {summary.categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
          <label>Text length
            <select data-testid="filter-length" value={length} onChange={(e) => setLength(e.target.value)}>
              <option value="">All lengths</option>
              {summary.text_lengths.map((l) => <option key={l} value={String(l)}>{l}</option>)}
            </select>
          </label>
          <div style={{ alignSelf: "flex-end" }}>
            <button className="btn btn--sm" data-testid="filter-reset" onClick={resetFilters}>Reset</button>
          </div>
        </div>
        {detector && !summary.detectors.includes(detector) && (
          <div className="alert alert--warning" data-testid="robustness-unavailable">
            Detector <span className="mono">{detector}</span> is not present in the stored results.
          </div>
        )}
      </div>

      {filtered.length === 0 ? (
        <div className="empty-state" data-testid="robustness-no-match">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No results match the current filters</div>
          <button className="btn btn--sm" style={{ marginTop: "8px" }} onClick={resetFilters}>Reset filters</button>
        </div>
      ) : (
        <>
          <div className="card" style={{ marginTop: "16px" }}>
            <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "4px" }}>Robustness matrix</h3>
            <p style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginTop: 0 }}>
              Conditional preservation rate: of the samples detected at baseline, the fraction still
              detected after transformation, with 95% Wilson confidence intervals. Select a cell for detail.
            </p>
            <div className="matrix-wrap">
              <table className="table" data-testid="robustness-matrix">
                <thead>
                  <tr>
                    <th>Detector / config</th>
                    {transforms.map((t) => <th key={t} style={{ textAlign: "right" }}>{t}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {pairs.map((p) => (
                    <tr key={`${p.detector}||${p.config}`}>
                      <td>
                        <strong style={{ fontSize: "12px" }}>{p.detector}</strong>
                        <div className="mono" style={{ fontSize: "11px", color: "var(--color-text-secondary)" }}>{p.config}</div>
                      </td>
                      {transforms.map((t) => {
                        const k = cellKey(p.detector, p.config, t);
                        const cell = byCell.get(k);
                        if (!cell) return <td key={t} style={{ textAlign: "right", color: "var(--color-text-muted)" }}>n/a</td>;
                        const alpha = Math.max(0.02, Math.min(0.2, cell.robustness_rate * 0.2));
                        return (
                          <td
                            key={t}
                            className={`matrix-cell${selected === k ? " matrix-cell--selected" : ""}`}
                            data-testid={`matrix-cell-${p.detector}-${p.config}-${t}`}
                            style={{ background: `rgba(22, 101, 52, ${alpha.toFixed(3)})` }}
                            onClick={() => setSelected(k)}
                          >
                            {pct(cell.robustness_rate)}
                            <span className="matrix-cell__ci">{ci(cell.robustness_ci_low, cell.robustness_ci_high)}</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {selectedCell && (
            <div className="card" style={{ marginTop: "16px" }} data-testid="robustness-detail">
              <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>
                <span className="mono">{selectedCell.detector_name}</span>
                {" / "}
                <span className="mono">{selectedCell.config_identifier}</span>
                {" / "}
                <span className="mono">{selectedCell.transform_name}</span>
              </h3>
              <div className="grid-stats">
                <StatCard label="Baseline detection" value={pct(selectedCell.baseline_rate)} />
                <StatCard label="Baseline 95% CI" value={ci(selectedCell.baseline_ci_low, selectedCell.baseline_ci_high)} />
                <StatCard label="Transformed detection" value={pct(selectedCell.transformed_rate)} />
                <StatCard label="Transformed 95% CI" value={ci(selectedCell.transformed_ci_low, selectedCell.transformed_ci_high)} />
                <StatCard label="Robustness (preserved)" value={pct(selectedCell.robustness_rate)} color="#166534" />
                <StatCard label="Robustness 95% CI" value={ci(selectedCell.robustness_ci_low, selectedCell.robustness_ci_high)} />
                <StatCard label="Samples" value={selectedCell.total_samples} />
                <StatCard label="Mean score delta" value={fmtDelta(selectedCell.mean_score_delta)} />
              </div>
              <p style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginBottom: 0 }}>
                Baseline detection is measured on untransformed text; robustness is the fraction of
                baseline-detected samples still detected after transformation — not a second detection rate.
              </p>
            </div>
          )}
        </>
      )}

      <h3 style={{ fontSize: "15px", fontWeight: 700, margin: "24px 0 4px" }}>Comparison across stored results</h3>
      <p style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginTop: 0 }}>
        Global view over all stored results (filters above apply to the matrix only).
      </p>
      <div data-testid="robustness-comparison">
        <ComparisonTable title="By model / config" groups={comparison.by_model} testId="comparison-by-model" />
        <ComparisonTable title="By transformation" groups={comparison.by_transform} testId="comparison-by-transform" />
        <ComparisonTable title="By category" groups={comparison.by_category} testId="comparison-by-category" />
        <ComparisonTable title="By text length" groups={comparison.by_length} testId="comparison-by-length" />
      </div>

      {report.limitations.length > 0 && (
        <div className="card" style={{ marginTop: "16px" }}>
          <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "8px" }}>Limitations</h3>
          <ul style={{ fontSize: "13px", color: "var(--color-text-secondary)", margin: 0, paddingLeft: "18px" }}>
            {report.limitations.map((l, i) => <li key={i}>{l}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}
