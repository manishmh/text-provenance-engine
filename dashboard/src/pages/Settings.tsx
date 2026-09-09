import { useState } from "react";
import { useConfig } from "../hooks/useConfig";
import { ConnectionForm } from "../components/ConnectionForm";

export function SettingsPage() {
  const { config, setConfig, clearConfig } = useConfig();
  const [saved, setSaved] = useState(false);

  return (
    <div>
      <h2 className="page-header">Settings</h2>
      <p className="page-sub">Server connection and developer setup.</p>

      <div className="card settings__card">
        <h3 className="card__title">Developer Setup</h3>
        {config && (
          <div className="settings__current">
            <span className="badge badge--success">Connected</span>
            <span className="mono settings__url">{config.baseUrl}</span>
            <span className="settings__key">
              {config.apiKey ? "API key configured" : "No API key (auth disabled)"}
            </span>
          </div>
        )}
        <p className="settings__hint">
          Update the API base URL or key below. The connection is verified
          before saving; your key is sent only to this server.
        </p>
        <ConnectionForm
          key={config ? `${config.baseUrl}:${config.apiKey ? "k" : "n"}` : "fresh"}
          initialBaseUrl={config?.baseUrl}
          initialApiKey={config?.apiKey}
          submitLabel="Save connection"
          testPrefix="settings-"
          onSave={(c) => { setConfig(c); setSaved(true); }}
        />
        {saved && (
          <div className="alert alert--success" style={{ marginTop: "16px", marginBottom: 0 }}>
            Connection saved.
          </div>
        )}
        {config && (
          <div style={{ marginTop: "16px" }}>
            <button onClick={() => { clearConfig(); setSaved(false); }} className="btn btn--danger btn--sm">
              Disconnect
            </button>
          </div>
        )}
      </div>

      <div className="card settings__card">
        <h3 className="card__title">Privacy</h3>
        <p className="settings__hint" style={{ marginBottom: 0 }}>
          The dashboard stores the API base URL and key in session storage only.
          Closing the tab clears the session when the browser discards it.
          Keys are never logged, never stored in local storage, and disconnecting
          removes them immediately.
        </p>
      </div>
    </div>
  );
}
