import { useState, useCallback, useEffect, useMemo } from "react";
import type { DashboardConfig, RobustnessFocus, UsageResponse } from "./types/api";
import { ProvenanceApiClient } from "./api/client";
import { ConfigContext } from "./hooks/useConfig";
import { SetupScreen } from "./components/SetupScreen";
import { AppShell } from "./components/AppShell";
import type { ShellPage } from "./components/AppShell";
import { OverviewPage } from "./pages/Overview";
import { AnalyzePage } from "./pages/Analyze";
import { HistoryPage } from "./pages/History";
import { JobsPage } from "./pages/Jobs";
import { UsagePage } from "./pages/Usage";
import { ApiKeysPage } from "./pages/ApiKeys";
import { DetectorsPage } from "./pages/Detectors";
import { RobustnessPage } from "./pages/Robustness";
import { BenchmarksPage } from "./pages/Benchmarks";
import { SettingsPage } from "./pages/Settings";
import { PublicSite } from "./pages/Public";
import { supabaseConfigured, apiBaseUrl } from "./lib/supabase";
import { useSaaSSession } from "./hooks/useSession";
import { allowedShellPages } from "./utils/entitlements";

type SaaSRoute = "site" | "workspace" | "developer";

function routeFromHash(): SaaSRoute {
  if (typeof window === "undefined") return "site";
  if (window.location.hash.startsWith("#/app")) return "workspace";
  if (window.location.hash.startsWith("#/dev")) return "developer";
  return "site";
}

export default function App() {
  // Legacy developer flow (API base URL + key in session storage).
  // Unchanged: without Supabase configured this is the entire app.
  const [config, setConfigState] = useState<DashboardConfig | null>(() => {
    try {
      const saved = sessionStorage.getItem("provenance_config");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [page, setPage] = useState<ShellPage>("overview");
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [robustnessFocus, setRobustnessFocus] = useState<RobustnessFocus | undefined>(undefined);
  const [route, setRoute] = useState<SaaSRoute>(() =>
    supabaseConfigured ? routeFromHash() : "developer",
  );

  // SaaS session (anonymous until Supabase is configured and signed in).
  // Safe to run in legacy mode: it resolves to anonymous without fetching.
  const session = useSaaSSession();

  useEffect(() => {
    if (!supabaseConfigured) return;
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const viewRunResults = useCallback((focus: RobustnessFocus) => {
    setRobustnessFocus(focus);
    setPage("robustness");
  }, []);

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

  // SaaS workspace client: Supabase bearer, no API key.
  const saasClient = useMemo(() => {
    const c = new ProvenanceApiClient(apiBaseUrl, "");
    c.setBearer(session.token);
    return c;
  }, [session.token]);

  const goWorkspace = useCallback(() => {
    window.location.hash = "#/app";
    setRoute("workspace");
    setPage("overview");
  }, []);

  const goSite = useCallback(() => {
    window.location.hash = "#/";
    setRoute("site");
  }, []);

  const goDeveloper = useCallback(() => {
    window.location.hash = "#/dev";
    setRoute("developer");
  }, []);

  const saasSignOut = useCallback(async () => {
    await session.signOut();
    setUsage(null);
    goSite();
  }, [session, goSite]);

  // ---- Legacy mode: identical to v1 (welcome screen → full shell) ----
  if (!supabaseConfigured) {
    if (!config || !client) {
      return <SetupScreen onSave={setConfig} />;
    }
    return (
      <ConfigContext.Provider value={{ config, client, setConfig, clearConfig, usage, refreshUsage }}>
        <AppShell page={page} baseUrl={config.baseUrl} onNavigate={setPage} onDisconnect={clearConfig}>
          <ShellPages page={page} onNavigate={setPage} viewRunResults={viewRunResults} robustnessFocus={robustnessFocus} />
        </AppShell>
      </ConfigContext.Provider>
    );
  }

  // ---- SaaS mode ----
  // Developer sign-in preserves the exact legacy workflow (API key shell).
  if (route === "developer" || config) {
    if (!config || !client) {
      return (
        <div>
          <div style={{ padding: "12px 20px" }}>
            <button className="link-btn" onClick={goSite}>← Back to site</button>
          </div>
          <SetupScreen onSave={setConfig} />
        </div>
      );
    }
    return (
      <ConfigContext.Provider value={{ config, client, setConfig, clearConfig, usage, refreshUsage }}>
        <AppShell page={page} baseUrl={config.baseUrl} onNavigate={setPage} onDisconnect={clearConfig}>
          <ShellPages page={page} onNavigate={setPage} viewRunResults={viewRunResults} robustnessFocus={robustnessFocus} />
        </AppShell>
      </ConfigContext.Provider>
    );
  }

  const entitlements = session.me?.entitlements ?? null;
  const allowed = allowedShellPages({ legacyApiKey: false, entitlements });
  const mayEnter = route === "workspace" && session.status === "user" && allowed.length > 0;

  if (!mayEnter) {
    return (
      <PublicSite session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} />
    );
  }

  const effectivePage = allowed.includes(page) ? page : "overview";
  const saasConfig: DashboardConfig = { baseUrl: apiBaseUrl, apiKey: "" };
  const noop = () => {};

  return (
    <ConfigContext.Provider
      value={{
        config: saasConfig, client: saasClient, setConfig: noop,
        clearConfig: () => { saasSignOut(); }, usage, refreshUsage: async () => {},
      }}
    >
      <AppShell
        page={effectivePage} baseUrl={apiBaseUrl} onNavigate={setPage}
        onDisconnect={() => { saasSignOut(); }} pages={allowed}
      >
        {allowed.includes(page) ? (
          <ShellPages page={page} onNavigate={setPage} viewRunResults={viewRunResults} robustnessFocus={robustnessFocus} />
        ) : (
          <UpgradeNotice plan={session.me?.plan ?? "free"} />
        )}
      </AppShell>
    </ConfigContext.Provider>
  );
}

function ShellPages({ page, onNavigate, viewRunResults, robustnessFocus }: {
  page: ShellPage;
  onNavigate: (p: ShellPage) => void;
  viewRunResults: (f: RobustnessFocus) => void;
  robustnessFocus: RobustnessFocus | undefined;
}) {
  return (
    <>
      {page === "overview" && <OverviewPage onNavigate={onNavigate} />}
      {page === "analyze" && <AnalyzePage />}
      {page === "history" && <HistoryPage />}
      {page === "jobs" && <JobsPage />}
      {page === "usage" && <UsagePage />}
      {page === "detectors" && <DetectorsPage />}
      {page === "robustness" && <RobustnessPage focus={robustnessFocus} />}
      {page === "benchmarks" && <BenchmarksPage onViewResults={viewRunResults} />}
      {page === "admin" && <ApiKeysPage />}
      {page === "settings" && <SettingsPage />}
    </>
  );
}

function UpgradeNotice({ plan }: { plan: string }) {
  return (
    <div>
      <h2 className="page-header">Pro workspace</h2>
      <div className="card" style={{ maxWidth: 560 }}>
        <h3 className="card__title">This section requires Pro</h3>
        <p className="settings__hint">
          Your current plan ({plan}) includes analysis, history, and usage.
          Benchmarks, robustness comparisons, background jobs, detector
          management, and API keys are part of the Pro workspace.
        </p>
        <p className="settings__hint" style={{ marginBottom: 0 }}>
          Payments are not available yet — Pro access is currently granted manually.
        </p>
      </div>
    </div>
  );
}
