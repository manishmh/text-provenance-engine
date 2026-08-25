import { useState } from "react";
import type { DashboardConfig } from "../types/api";
import { ProvenanceApiClient } from "../api/client";

export function SetupScreen({ onSave }: { onSave: (c: DashboardConfig) => void }) {
  const [baseUrl, setBaseUrl] = useState("http://localhost:8000");
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleConnect = async () => {
    setError("");
    setLoading(true);
    try {
      const client = new ProvenanceApiClient(baseUrl, apiKey);
      const health = await client.health();
      if (health.status !== "ok") {
        setError("Server returned unexpected status");
        return;
      }
      onSave({ baseUrl, apiKey });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Connection failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#1a1a2e" }}>
      <div className="card" style={{ padding: "40px", width: "100%", maxWidth: 420, boxShadow: "0 4px 24px rgba(0,0,0,0.3)", background: "#16213e" }}>
        <h1 style={{ color: "#e94560", fontSize: "22px", marginBottom: "8px" }}>Provenance Engine</h1>
        <p style={{ color: "#a0a0b0", fontSize: "14px", marginBottom: "28px" }}>
          Connect to your API server to get started.
        </p>

        <label className="label" style={{ color: "#ccc" }}>API Base URL</label>
        <input
          type="text"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          data-testid="input-base-url"
          className="input input--dark"
        />

        <label className="label" style={{ color: "#ccc", marginTop: "16px" }}>API Key</label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          data-testid="input-api-key"
          placeholder="Leave empty if auth is disabled"
          className="input input--dark"
          style={{ marginBottom: "20px" }}
        />

        {error && (
          <div data-testid="setup-error" className="alert alert--error">
            {error}
          </div>
        )}

        <button
          onClick={handleConnect}
          disabled={loading}
          data-testid="btn-connect"
          className="btn btn--primary"
          style={{ width: "100%", padding: "12px", fontSize: "15px" }}
        >
          {loading ? <><span className="spinner spinner-sm" /> Connecting...</> : "Connect"}
        </button>
      </div>
    </div>
  );
}
