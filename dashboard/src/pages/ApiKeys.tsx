import { useEffect, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type { ApiKeySummary } from "../types/api";
import { ApiError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

export function ApiKeysPage() {
  const { client, config } = useConfig();
  const [keys, setKeys] = useState<ApiKeySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [newKeyName, setNewKeyName] = useState("");
  const [createdKey, setCreatedKey] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const loadKeys = async () => {
    if (!client) return;
    try {
      const res = await client.listApiKeys();
      setKeys(res.keys);
      setError("");
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadKeys();
  }, [client]);

  const handleCreate = async () => {
    if (!client || !newKeyName.trim()) return;
    setCreating(true);
    setError("");
    try {
      const res = await client.createApiKey(newKeyName.trim());
      setCreatedKey(res.key);
      setNewKeyName("");
      await loadKeys();
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId: string) => {
    if (!client) return;
    if (!confirm("Revoke this API key? The key will stop working immediately.")) return;
    try {
      await client.revokeApiKey(keyId);
      await loadKeys();
    } catch (e: unknown) {
      if (e instanceof ApiError) setError(e.userMessage);
    }
  };

  const hasAdminKey = !!config?.apiKey;

  return (
    <div>
      <h2 className="page-header">API Key Management</h2>

      {!hasAdminKey && (
        <div className="alert alert--info" style={{ marginBottom: "16px" }}>
          API key management requires an admin key. Configure an admin key in your API server via <code>PROVENANCE_ADMIN_API_KEY</code> and enter it as your API key on the setup screen.
        </div>
      )}

      {error && (
        <div data-testid="admin-error" className="alert alert--error">
          {error}
          <button onClick={() => setError("")} className="close-btn" style={{ float: "right" }}>×</button>
        </div>
      )}

      {createdKey && (
        <div data-testid="created-key-alert" className="alert alert--success" style={{ marginBottom: "16px" }}>
          <strong>API Key Created</strong>
          <div style={{ marginTop: "8px", padding: "8px", background: "#fff", borderRadius: "4px" }}>
            <div style={{ fontSize: "12px", color: "#6b7280", marginBottom: "4px" }}>Raw secret (shown only once — save it now):</div>
            <code data-testid="created-key-value" className="mono" style={{ fontSize: "13px", wordBreak: "break-all" }}>{createdKey}</code>
          </div>
          <button onClick={() => setCreatedKey(null)} className="btn btn--sm" style={{ marginTop: "8px" }}>Dismiss</button>
        </div>
      )}

      {hasAdminKey && (
        <div className="card" style={{ marginBottom: "20px" }}>
          <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>Create New Key</h3>
          <div style={{ display: "flex", gap: "8px" }}>
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="Key name (e.g. my-app)"
              data-testid="input-key-name"
              className="input"
              style={{ maxWidth: 300 }}
              onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            />
            <button
              onClick={handleCreate}
              disabled={creating || !newKeyName.trim()}
              data-testid="btn-create-key"
              className="btn btn--primary"
            >
              {creating ? <><span className="spinner spinner-sm" /> Creating...</> : "Create Key"}
            </button>
          </div>
        </div>
      )}

      <div className="card">
        <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "12px" }}>Existing Keys</h3>
        {loading ? (
          <div className="loading"><span className="spinner" /> Loading keys...</div>
        ) : keys.length === 0 ? (
          <div className="empty-state" style={{ padding: "20px" }}>
            <div className="empty-state__text">No API keys found</div>
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Key ID</th>
                <th>Status</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k) => (
                <tr key={k.key_id}>
                  <td style={{ fontWeight: 500 }}>{k.name}</td>
                  <td className="mono" style={{ fontSize: "12px" }}>{k.key_id.slice(0, 8)}...</td>
                  <td><StatusBadge status={k.status} /></td>
                  <td style={{ color: "#6b7280" }}>{new Date(k.created_at).toLocaleString()}</td>
                  <td>
                    {k.status === "active" && (
                      <button
                        onClick={() => handleRevoke(k.key_id)}
                        data-testid={`btn-revoke-${k.key_id.slice(0, 8)}`}
                        className="btn btn--sm btn--danger"
                      >
                        Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
