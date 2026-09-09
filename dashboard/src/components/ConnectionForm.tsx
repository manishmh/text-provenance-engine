import { useState } from "react";
import type { DashboardConfig } from "../types/api";
import { ProvenanceApiClient } from "../api/client";

/** Verify a server is reachable before saving its config. */
export async function connectToApi(baseUrl: string, apiKey: string): Promise<DashboardConfig> {
  const client = new ProvenanceApiClient(baseUrl, apiKey);
  const health = await client.health();
  if (health.status !== "ok") throw new Error("Server returned unexpected status");
  return { baseUrl, apiKey };
}

interface Props {
  initialBaseUrl?: string;
  initialApiKey?: string;
  submitLabel?: string;
  /** Prefix for data-testid attributes (defaults to the setup-screen ids). */
  testPrefix?: string;
  onSave: (c: DashboardConfig) => void;
}

export function ConnectionForm({
  initialBaseUrl = "http://localhost:8000",
  initialApiKey = "",
  submitLabel = "Connect",
  testPrefix = "",
  onSave,
}: Props) {
  const [baseUrl, setBaseUrl] = useState(initialBaseUrl);
  const [apiKey, setApiKey] = useState(initialApiKey);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const tid = (name: string) => `${testPrefix}${name}`;

  const handleSubmit = async () => {
    setError("");
    setLoading(true);
    try {
      onSave(await connectToApi(baseUrl.trim(), apiKey));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Connection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <label className="label" htmlFor={tid("input-base-url")}>API Base URL</label>
      <input
        id={tid("input-base-url")}
        type="text"
        value={baseUrl}
        onChange={(e) => setBaseUrl(e.target.value)}
        data-testid={tid("input-base-url")}
        className="input"
        autoComplete="url"
      />

      <label className="label" htmlFor={tid("input-api-key")} style={{ marginTop: "16px" }}>API Key</label>
      <input
        id={tid("input-api-key")}
        type="password"
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        data-testid={tid("input-api-key")}
        placeholder="Leave empty if auth is disabled"
        className="input"
        style={{ marginBottom: "20px" }}
        autoComplete="current-password"
      />

      {error && (
        <div data-testid={tid("setup-error")} className="alert alert--error">
          {error}
        </div>
      )}

      <button
        onClick={handleSubmit}
        disabled={loading}
        data-testid={tid("btn-connect")}
        className="btn btn--primary btn--block"
      >
        {loading ? <><span className="spinner spinner-sm" /> Connecting...</> : submitLabel}
      </button>
    </div>
  );
}
