import { useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type { AnalyzeResponse, DetectionResultItem } from "../types/api";
import { ApiError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

const DETECTORS = ["unicode", "kgw", "kgw-reference", "synthid", "synthid-reference"];
const WATERMARK_DETECTORS = new Set(["kgw", "kgw-reference", "synthid", "synthid-reference"]);

function DetectorResult({ r }: { r: DetectionResultItem }) {
  return (
    <div data-testid="detector-result" className="detector-card">
      <div className="detector-card__header">
        <strong style={{ fontSize: "15px" }}>{r.detector}</strong>
        <StatusBadge status={r.detected ? "detected" : "not_detected"} />
        <span className="mono" style={{ color: "#6b7280", fontSize: "12px" }}>v{r.detector_version}</span>
      </div>

      <dl className="detector-card__grid">
        <div><dt>Score</dt><dd>{r.score ?? "—"}</dd></div>
        <div><dt>Threshold</dt><dd>{r.threshold ?? "—"}</dd></div>
        <div><dt>Confidence</dt><dd>{r.confidence}</dd></div>
        <div><dt>Implementation</dt><dd>{r.implementation_kind}</dd></div>
        <div><dt>Compatibility</dt><dd>{r.compatibility}</dd></div>
        <div><dt>Status</dt><dd>{r.status}</dd></div>
      </dl>

      {Object.keys(r.evidence).length > 0 && (
        <details style={{ marginTop: "8px" }}>
          <summary style={{ cursor: "pointer", fontSize: "13px", color: "#6b7280" }}>Evidence</summary>
          <pre className="code-block" style={{ marginTop: "8px" }}>{JSON.stringify(r.evidence, null, 2)}</pre>
        </details>
      )}

      {r.limitations.length > 0 && (
        <div className="warning-box">
          {r.limitations.map((l, i) => <div key={i}>⚠ {l}</div>)}
        </div>
      )}
    </div>
  );
}

export function AnalyzePage() {
  const { client } = useConfig();
  const [text, setText] = useState("");
  const [detectors, setDetectors] = useState<string[]>(["unicode"]);
  const [configPath, setConfigPath] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const needsConfig = detectors.some((d) => WATERMARK_DETECTORS.has(d));
  const canAnalyze = !loading && text.trim().length > 0;

  const handleAnalyze = async () => {
    if (!client || !canAnalyze) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const r = await client.analyze(
        text,
        detectors.length > 0 ? detectors : undefined,
        needsConfig ? configPath || undefined : undefined,
      );
      setResult(r);
    } catch (e: unknown) {
      if (e instanceof ApiError) {
        setError(e.userMessage);
      } else {
        setError(e instanceof Error ? e.message : "Analysis failed");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canAnalyze) {
      handleAnalyze();
    }
  };

  const toggleDetector = (d: string) => {
    setDetectors((prev) => prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]);
  };

  return (
    <div onKeyDown={handleKeyDown}>
      <h2 className="page-header">Analyze</h2>

      <div className="card" style={{ marginBottom: "20px" }}>
        <label className="label">Input Text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          data-testid="input-text"
          placeholder="Enter text to analyze... (Ctrl+Enter to submit)"
          className="textarea"
          disabled={loading}
        />
        <div style={{ fontSize: "12px", color: "#6b7280", marginTop: "4px" }}>
          {text.length.toLocaleString()} characters
        </div>

        <div style={{ marginTop: "16px", marginBottom: "12px" }}>
          <label className="label">Detectors</label>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
            {DETECTORS.map((d) => (
              <button
                key={d}
                onClick={() => toggleDetector(d)}
                data-testid={`detector-${d}`}
                disabled={loading}
                className={`btn btn--sm${detectors.includes(d) ? " btn--primary" : ""}`}
              >
                {d}
              </button>
            ))}
          </div>
        </div>

        {needsConfig && (
          <div style={{ marginTop: "12px" }}>
            <label className="label">Config Path (required for watermark detectors)</label>
            <input
              type="text"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              data-testid="input-config"
              placeholder="/path/to/config.json"
              className="input"
              disabled={loading}
            />
          </div>
        )}

        {error && (
          <div data-testid="analyze-error" className="alert alert--error" style={{ marginTop: "12px" }}>
            {error}
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          data-testid="btn-analyze"
          className="btn btn--primary"
          style={{ marginTop: "16px", padding: "10px 24px", fontSize: "14px" }}
        >
          {loading ? <><span className="spinner spinner-sm" /> Analyzing...</> : "Analyze"}
        </button>
      </div>

      {result && (
        <div data-testid="analyze-result">
          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "16px" }}>
            <h3 style={{ fontSize: "16px", margin: 0 }}>Results</h3>
            <StatusBadge status={result.status} />
            {result.duration_ms != null && (
              <span className="mono" style={{ color: "#6b7280", fontSize: "13px" }}>
                {Number(result.duration_ms).toFixed(1)}ms
              </span>
            )}
            <span className="mono" style={{ color: "#9ca3af", fontSize: "12px" }}>
              ID: {result.analysis_id.slice(0, 8)}
            </span>
          </div>

          {result.results.map((r, i) => <DetectorResult key={i} r={r} />)}

          {result.limitations.length > 0 && (
            <div className="alert alert--warning" style={{ background: "#fef3c7", color: "#92400e" }}>
              <strong>Limitations:</strong>
              {result.limitations.map((l, i) => <div key={i}>⚠ {l}</div>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
