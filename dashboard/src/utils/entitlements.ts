import type { Entitlements } from "../types/api";
import type { ShellPage } from "../components/AppShell";

const FULL_ORDER: ShellPage[] = [
  "overview", "analyze", "history", "jobs", "usage",
  "admin", "detectors", "robustness", "benchmarks", "settings",
];
const BASIC_ORDER: ShellPage[] = ["overview", "analyze", "history", "usage", "settings"];

/** Pages visible in the workspace shell for a given access level. */
export function allowedShellPages(opts: {
  legacyApiKey: boolean;
  entitlements: Entitlements | null;
}): ShellPage[] {
  if (opts.legacyApiKey) return FULL_ORDER;
  const ent = opts.entitlements;
  if (!ent || !ent.can_access_dashboard) return [];
  if (ent.can_access_advanced) return FULL_ORDER;
  return BASIC_ORDER;
}

/** Whether the workspace shell may be entered at all. */
export function canEnterWorkspace(opts: {
  legacyApiKey: boolean;
  entitlements: Entitlements | null;
}): boolean {
  return allowedShellPages(opts).length > 0;
}
