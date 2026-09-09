import type { ReactNode } from "react";

export type ShellPage =
  | "overview" | "analyze" | "history" | "jobs" | "usage"
  | "admin" | "detectors" | "robustness" | "benchmarks" | "settings";

interface NavItem { key: ShellPage; label: string; section: string; }

const NAV_ITEMS: NavItem[] = [
  { key: "overview", label: "Overview", section: "Workspace" },
  { key: "analyze", label: "Analyze", section: "Workspace" },
  { key: "benchmarks", label: "Benchmarks", section: "Workspace" },
  { key: "robustness", label: "Robustness", section: "Workspace" },
  { key: "detectors", label: "Detectors", section: "Workspace" },
  { key: "history", label: "History", section: "Review" },
  { key: "jobs", label: "Jobs", section: "Review" },
  { key: "usage", label: "Usage", section: "Review" },
  { key: "admin", label: "API Keys", section: "Admin" },
  { key: "settings", label: "Settings", section: "Admin" },
];

interface Props {
  page: ShellPage;
  baseUrl: string;
  onNavigate: (p: ShellPage) => void;
  onDisconnect: () => void;
  /** Visible nav pages (entitlement-gated). Defaults to the full workspace. */
  pages?: ShellPage[];
  children: ReactNode;
}

export function AppShell({ page, baseUrl, onNavigate, onDisconnect, pages, children }: Props) {
  const visible = new Set(pages ?? NAV_ITEMS.map((i) => i.key));
  const items = NAV_ITEMS.filter((i) => visible.has(i.key));
  let lastSection = "";
  return (
    <div className="shell">
      <aside className="shell__sidebar">
        <div className="shell__brand">Provenance Engine</div>
        <nav className="shell__nav">
          {items.map((item) => {
            const header = item.section !== lastSection
              ? <div key={item.section} className="shell__section">{item.section}</div>
              : null;
            lastSection = item.section;
            return (
              <div key={item.key}>
                {header}
                <button
                  onClick={() => onNavigate(item.key)}
                  data-testid={`nav-${item.key}`}
                  className={`shell__link${page === item.key ? " shell__link--active" : ""}`}
                  aria-current={page === item.key ? "page" : undefined}
                >
                  {item.label}
                </button>
              </div>
            );
          })}
        </nav>
        <div className="shell__footer">
          <div className="shell__server mono truncate" title={baseUrl}>{baseUrl}</div>
          <button onClick={onDisconnect} data-testid="btn-disconnect" className="btn btn--sm">
            Disconnect
          </button>
        </div>
      </aside>
      <main className="shell__main">{children}</main>
    </div>
  );
}
