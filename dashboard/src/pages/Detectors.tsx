import { useEffect, useState } from "react";
import type { ReactElement } from "react";
import { useConfig } from "../hooks/useConfig";
import type { DetectorCapability } from "../types/api";
import { ApiError } from "../api/client";
import { isDetectorsResponse } from "../utils/guards";

function boolBadge(value: boolean, trueLabel: string, falseLabel: string): ReactElement {
  return value ? (
    <span className="badge badge--info">{trueLabel}</span>
  ) : (
    <span className="badge badge--neutral">{falseLabel}</span>
  );
}

export function DetectorsPage() {
  const { client } = useConfig();
  const [detectors, setDetectors] = useState<DetectorCapability[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [malformed, setMalformed] = useState(false);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const data: unknown = await client.listDetectors();
        if (cancelled) return;
        if (!isDetectorsResponse(data)) {
          setMalformed(true);
        } else {
          setDetectors(data.detectors);
        }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof ApiError ? e.userMessage : "Failed to load detectors");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client]);

  if (loading) return <div className="loading"><span className="spinner" /> Loading detectors...</div>;
  if (error) return <div className="alert alert--error" data-testid="detectors-error">{error}</div>;
  if (malformed || detectors === null) {
    return (
      <div className="alert alert--error" data-testid="detectors-malformed">
        The server returned detector data in an unrecognized format. Please check the server version.
      </div>
    );
  }
  if (detectors.length === 0) {
    return (
      <div>
        <h2 className="page-header">Detectors</h2>
        <div className="empty-state" data-testid="detectors-empty">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No detectors reported by the server</div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <h2 className="page-header">Detectors</h2>
      <p style={{ fontSize: "13px", color: "var(--color-text-secondary)", marginBottom: "16px" }}>
        Detectors supported by this engine, with capabilities and configuration requirements.
        Only detectors the server reports are shown.
      </p>
      <table className="table" data-testid="detectors-table">
        <thead>
          <tr>
            <th>Detector</th>
            <th>Family</th>
            <th>Compatibility</th>
            <th>Config required</th>
            <th>Generation</th>
            <th>Benchmarking</th>
            <th>Tokenizer</th>
            <th>Limitations</th>
          </tr>
        </thead>
        <tbody>
          {detectors.map((d) => (
            <tr key={d.name} data-testid={`detector-row-${d.name}`}>
              <td>
                <strong>{d.display_name}</strong>
                <div className="mono" style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>{d.name}</div>
                {d.description && (
                  <div style={{ fontSize: "12px", color: "var(--color-text-secondary)", marginTop: "4px" }}>{d.description}</div>
                )}
              </td>
              <td className="mono" style={{ fontSize: "12px" }}>{d.implementation_kind}</td>
              <td style={{ fontSize: "12px" }}>{d.compatibility}</td>
              <td>{boolBadge(d.requires_config, "Required", "Not required")}</td>
              <td>{boolBadge(d.supports_generation, "Supported", "Unsupported")}</td>
              <td>{boolBadge(d.supports_benchmarking, "Supported", "Unsupported")}</td>
              <td style={{ fontSize: "12px" }}>{d.tokenizer_requirements ?? "any"}</td>
              <td style={{ fontSize: "12px" }}>
                {d.known_limitations.length === 0 ? (
                  <span style={{ color: "var(--color-text-muted)" }}>None reported</span>
                ) : (
                  <ul style={{ margin: 0, paddingLeft: "16px" }}>
                    {d.known_limitations.map((lim, i) => <li key={i}>{lim}</li>)}
                  </ul>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
