import { useState, useCallback, useEffect } from "react";
import type { DashboardConfig, UsageResponse } from "./types/api";
import { ProvenanceApiClient } from "./api/client";
import { ConfigContext } from "./hooks/useConfig";
import { SetupScreen } from "./components/SetupScreen";
import { OverviewPage } from "./pages/Overview";
import { AnalyzePage } from "./pages/Analyze";
import { HistoryPage } from "./pages/History";
import { JobsPage } from "./pages/Jobs";
import { UsagePage } from "./pages/Usage";
import { ApiKeysPage } from "./pages/ApiKeys";
import { DetectorsPage } from "./pages/Detectors";
import { RobustnessPage } from "./pages/Robustness";

type Page = "overview" | "analyze" | "history" | "jobs" | "usage" | "admin" | "detectors" | "robustness";

const NAV_ITEMS: { key: Page; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "analyze", label: "Analyze" },
  { key: "history", label: "History" },
  { key: "jobs", label: "Jobs" },
  { key: "usage", label: "Usage" },
  { key: "detectors", label: "Detectors" },
  { key: "robustness", label: "Robustness" },
  { key: "admin", label: "API Keys" },
];

export default function App() {
  const [config, setConfigState] = useState<DashboardConfig | null>(() => {
    try {
      const saved = sessionStorage.getItem("provenance_config");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [page, setPage] = useState<Page>("overview");
  const [usage, setUsage] = useState<UsageResponse | null>(null);

  const client = config ? new ProvenanceApiClient(config.baseUrl, config.apiKey) : null;

  const setConfig = useCallback((c: DashboardConfig) => {
    sessionStorage.setItem("provenance_config", JSON.stringify(c));
    setConfigState(c);
  }, []);

  const clearConfig = useCallback(() => {
    sessionStorage.removeItem("provenance_config");
    setConfigState(null);
    setUsage(null);
  }, []);

  const refreshUsage = useCallback(async () => {
    if (!client) return;
    try {
      const u = await client.getUsage();
      setUsage(u);
    } catch {
      /* ignore */
    }
  }, [client]);

  useEffect(() => {
    if (client) refreshUsage();
  }, [client, refreshUsage]);

  if (!config || !client) {
    return <SetupScreen onSave={setConfig} />;
  }

  return (
    <ConfigContext.Provider value={{ config, client, setConfig, clearConfig, usage, refreshUsage }}>
      <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
        <nav className="nav">
          <span className="nav__logo">Provenance Engine</span>
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              onClick={() => setPage(item.key)}
              data-testid={`nav-${item.key}`}
              className={`nav__btn${page === item.key ? " nav__btn--active" : ""}`}
            >
              {item.label}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          <span style={{ color: "#666", fontSize: "12px", marginRight: "12px" }}>
            {config.baseUrl}
          </span>
          <button onClick={clearConfig} data-testid="btn-disconnect" className="btn btn--sm">
            Disconnect
          </button>
        </nav>

        <main style={{ flex: 1, padding: "24px", maxWidth: 1200, margin: "0 auto", width: "100%" }}>
          {page === "overview" && <OverviewPage />}
          {page === "analyze" && <AnalyzePage />}
          {page === "history" && <HistoryPage />}
          {page === "jobs" && <JobsPage />}
          {page === "usage" && <UsagePage />}
          {page === "detectors" && <DetectorsPage />}
          {page === "robustness" && <RobustnessPage />}
          {page === "admin" && <ApiKeysPage />}
        </main>
      </div>
    </ConfigContext.Provider>
  );
}
