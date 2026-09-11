import { useState, useCallback, useEffect, useMemo, useRef } from "react";
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
import { PricingPage } from "./pages/Pricing";
import { MethodologyPage } from "./pages/Methodology";
import { FreeOverviewPage } from "./pages/FreeOverview";
import { PrivacyPage, TermsPage } from "./pages/Legal";
import { AccountSettingsPage } from "./pages/AccountSettings";
import { WorkspaceUsagePage } from "./pages/WorkspaceUsage";
import { supabaseConfigured, apiBaseUrl } from "./lib/supabase";
import { consumeAuthReturnTo, isSupabaseAuthCallback } from "./lib/authNavigation";
import { useSaaSSession } from "./hooks/useSession";
import type { SaaSSession } from "./hooks/useSession";
import { allowedShellPages } from "./utils/entitlements";

type SaaSRoute = "site" | "pricing" | "methodology" | "privacy" | "terms" | "workspace" | "developer";

function routeFromHash(): SaaSRoute {
  if (typeof window === "undefined") return "site";
  if (window.location.hash.startsWith("#/pricing")) return "pricing";
  if (window.location.hash.startsWith("#/methodology")) return "methodology";
  if (window.location.hash.startsWith("#/privacy")) return "privacy";
  if (window.location.hash.startsWith("#/terms")) return "terms";
  if (window.location.hash.startsWith("#/app")) return "workspace";
  if (window.location.hash.startsWith("#/dev")) return "developer";
  return "site";
}

function pageFromHash(): ShellPage {
  if (typeof window === "undefined") return "overview";
  const segment = window.location.hash.replace(/^#\/app\/?/, "").split(/[?#]/)[0];
  const pages: Record<string, ShellPage> = {
    "": "overview", analyze: "analyze", history: "history", usage: "usage",
    settings: "settings", robustness: "robustness", benchmarks: "benchmarks",
    detectors: "detectors", jobs: "jobs", api: "admin", admin: "admin",
  };
  return pages[segment] ?? "overview";
}

function hashForPage(page: ShellPage): string {
  const paths: Record<ShellPage, string> = {
    overview: "", analyze: "analyze", history: "history", usage: "usage",
    settings: "settings", robustness: "robustness", benchmarks: "benchmarks",
    detectors: "detectors", jobs: "jobs", admin: "api",
  };
  return `#/app${paths[page] ? `/${paths[page]}` : ""}`;
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
  const [page, setPage] = useState<ShellPage>(() => pageFromHash());
  const [usage, setUsage] = useState<UsageResponse | null>(null);
  const [robustnessFocus, setRobustnessFocus] = useState<RobustnessFocus | undefined>(undefined);
  const [route, setRoute] = useState<SaaSRoute>(() =>
    supabaseConfigured ? routeFromHash() : "developer",
  );

  // SaaS session (anonymous until Supabase is configured and signed in).
  // Safe to run in legacy mode: it resolves to anonymous without fetching.
  const session = useSaaSSession();
  const callbackRoutePending = useRef(isSupabaseAuthCallback());

  useEffect(() => {
    if (!supabaseConfigured) return;
    const onHash = () => {
      setRoute(routeFromHash());
      if (window.location.hash.startsWith("#/app")) setPage(pageFromHash());
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // OAuth/email-confirmation returns to the origin, not the prior hash
  // route. Only restore the saved destination after Supabase JWT validation,
  // backend provisioning, and /v1/me have all completed.
  useEffect(() => {
    if (!supabaseConfigured || session.status !== "user" || !session.me) return;
    const intended = consumeAuthReturnTo() || (callbackRoutePending.current ? "#/app" : null);
    if (!intended) return;
    callbackRoutePending.current = false;
    window.location.hash = intended;
    setRoute(routeFromHash());
    if (intended.startsWith("#/app")) setPage(pageFromHash());
  }, [session.status, session.me]);

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

  const goWorkspace = useCallback((target?: ShellPage) => {
    const next = target ?? pageFromHash();
    window.location.hash = hashForPage(next);
    setRoute("workspace");
    setPage(next);
  }, []);

  const goSite = useCallback(() => {
    window.location.hash = "#/";
    setRoute("site");
  }, []);

  const navigateWorkspace = useCallback((next: ShellPage) => {
    setPage(next);
    if (supabaseConfigured && route === "workspace") window.location.hash = hashForPage(next);
  }, [route]);

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

  if (route === "pricing") {
    return <PublicFrame session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} page="pricing" />;
  }
  if (route === "methodology") {
    return <PublicFrame session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} page="methodology" />;
  }
  if (route === "privacy") {
    return <PublicFrame session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} page="privacy" />;
  }
  if (route === "terms") {
    return <PublicFrame session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} page="terms" />;
  }

  if (!mayEnter) {
    return (
      <PublicSite session={session} onEnterWorkspace={goWorkspace} onDeveloperSignIn={goDeveloper} onSignOut={saasSignOut} />
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
        page={effectivePage} baseUrl={apiBaseUrl} onNavigate={navigateWorkspace}
        onDisconnect={() => { saasSignOut(); }} pages={allowed}
      >
        {allowed.includes(page) ? (
          <ShellPages page={page} onNavigate={navigateWorkspace} viewRunResults={viewRunResults} robustnessFocus={robustnessFocus} freeQuota={!entitlements?.can_access_advanced ? session.quota : null} saasSession={session} onSaaSSignOut={saasSignOut} />
        ) : (
          <UpgradeNotice plan={session.me?.plan ?? "free"} />
        )}
      </AppShell>
    </ConfigContext.Provider>
  );
}

function PublicFrame({ session, onEnterWorkspace, onDeveloperSignIn, page }: {
  session: SaaSSession;
  onEnterWorkspace: () => void;
  onDeveloperSignIn: () => void;
  page: "pricing" | "methodology" | "privacy" | "terms";
}) {
  return (
    <div className="site">
      <header className="site__header">
        <a className="site__logo" href="#/"><span className="site__logo-mark">P</span>Provenance Engine</a>
        <nav className="site__nav" aria-label="Product navigation">
          <a href="#/">Product</a><a href="#/methodology">Methodology</a><a href="#/pricing">Pricing</a>
        </nav>
        <div className="site__actions">
          {session.status === "user" ? <button className="btn btn--primary btn--sm" onClick={onEnterWorkspace}>Open workspace</button> : <a className="btn btn--primary btn--sm" href="#/?auth=signin">Sign in</a>}
        </div>
      </header>
      {page === "pricing" ? <PricingPage session={session} onOpenWorkspace={onEnterWorkspace} /> : page === "methodology" ? <MethodologyPage /> : page === "privacy" ? <PrivacyPage /> : <TermsPage />}
      <footer className="site__footer">
        <div className="site__footer-brand"><a className="site__logo" href="#/"><span className="site__logo-mark">P</span>Provenance Engine</a><p>Evidence-led text provenance analysis for careful inspection.</p><small>© {new Date().getFullYear()} Provenance Engine</small></div>
        <div className="site__footer-links"><div><h2>Product</h2><a href="#/">Analyzer</a><a href="#/methodology">Methodology</a><a href="#/pricing">Pricing</a></div><div><h2>Trust</h2><a href="#/privacy">Privacy</a><a href="#/terms">Terms</a><button className="link-btn" onClick={onDeveloperSignIn}>Developer sign-in</button></div></div>
      </footer>
    </div>
  );
}

function ShellPages({ page, onNavigate, viewRunResults, robustnessFocus, freeQuota, saasSession, onSaaSSignOut }: {
  page: ShellPage;
  onNavigate: (p: ShellPage) => void;
  viewRunResults: (f: RobustnessFocus) => void;
  robustnessFocus: RobustnessFocus | undefined;
  freeQuota?: import("./types/api").QuotaInfo | null;
  saasSession?: SaaSSession;
  onSaaSSignOut?: () => void;
}) {
  return (
    <>
      {page === "overview" && (freeQuota ? <FreeOverviewPage quota={freeQuota} onNavigate={onNavigate} /> : <OverviewPage onNavigate={onNavigate} />)}
      {page === "analyze" && <AnalyzePage />}
      {page === "history" && <HistoryPage />}
      {page === "jobs" && <JobsPage />}
      {page === "usage" && (saasSession ? <WorkspaceUsagePage quota={saasSession.quota} /> : <UsagePage />)}
      {page === "detectors" && <DetectorsPage />}
      {page === "robustness" && <RobustnessPage focus={robustnessFocus} />}
      {page === "benchmarks" && <BenchmarksPage onViewResults={viewRunResults} />}
      {page === "admin" && <ApiKeysPage />}
      {page === "settings" && (saasSession && onSaaSSignOut ? <AccountSettingsPage session={saasSession} onSignOut={onSaaSSignOut} /> : <SettingsPage />)}
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
          Upgrade options are shown on the pricing page when billing is configured for this deployment.
        </p>
      </div>
    </div>
  );
}
