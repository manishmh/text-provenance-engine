import { useCallback, useEffect, useMemo, useState } from "react";
import { useConfig } from "../hooks/useConfig";
import type {
  BenchmarkOptions,
  BenchmarkRunCreateInput,
  BenchmarkRunResponse,
  BenchmarkRunSummary,
  RobustnessFocus,
} from "../types/api";
import { ApiError } from "../api/client";
import {
  isBenchmarkOptions,
  isBenchmarkRunList,
  isBenchmarkRunResponse,
} from "../utils/guards";
import { RobustnessDisclaimer } from "../components/Disclaimer";
import { StatCard } from "../components/StatCard";
import { StatusBadge } from "../components/StatusBadge";

/** Polling cadence for active runs. Terminal states stop all polling. */
export const BENCHMARK_POLL_MS = 3000;

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

function isTerminal(status: string): boolean {
  return TERMINAL.has(status);
}

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function errorText(e: unknown, fallback: string): string {
  return e instanceof ApiError ? e.userMessage : fallback;
}

/* ------------------------------------------------------------------ */
/* Runs list                                                           */
/* ------------------------------------------------------------------ */

function RunsList(props: {
  onNew: () => void;
  onOpen: (runId: string) => void;
}) {
  const { client } = useConfig();
  const [runs, setRuns] = useState<BenchmarkRunSummary[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [malformed, setMalformed] = useState(false);
  const [statusFilter, setStatusFilter] = useState("");
  const [actionError, setActionError] = useState("");
  const [acting, setActing] = useState("");

  const refresh = useCallback(async () => {
    if (!client) return;
    try {
      const data: unknown = await client.listBenchmarkRuns(statusFilter || undefined);
      if (!isBenchmarkRunList(data)) {
        setMalformed(true);
        return;
      }
      setRuns(data.runs);
      setError("");
    } catch (e: unknown) {
      setError(errorText(e, "Failed to load benchmark runs"));
    }
  }, [client, statusFilter]);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    setLoading(true);
    refresh().finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [client, refresh]);

  const hasActive = useMemo(() => (runs ?? []).some((r) => !isTerminal(r.status)), [runs]);

  useEffect(() => {
    if (!client || !hasActive) return;
    const timer = setInterval(() => { void refresh(); }, BENCHMARK_POLL_MS);
    return () => clearInterval(timer);
  }, [client, hasActive, refresh]);

  async function act(runId: string, kind: "cancel" | "retry") {
    if (!client) return;
    setActing(runId);
    setActionError("");
    try {
      if (kind === "cancel") await client.cancelBenchmarkRun(runId);
      else await client.retryBenchmarkRun(runId);
      await refresh();
    } catch (e: unknown) {
      setActionError(errorText(e, `Failed to ${kind} run`));
    } finally {
      setActing("");
    }
  }

  if (loading) return <div className="loading"><span className="spinner" /> Loading benchmark runs...</div>;
  if (error && runs === null) return <div className="alert alert--error" data-testid="runs-error">{error}</div>;
  if (malformed || runs === null) {
    return (
      <div className="alert alert--error" data-testid="runs-malformed">
        The server returned benchmark runs in an unrecognized format. Please check the server version.
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "16px" }}>
        <button className="btn btn--primary" data-testid="runs-new" onClick={props.onNew}>
          New benchmark run
        </button>
        <label style={{ fontSize: "12px", color: "var(--color-text-secondary)" }}>
          Status{" "}
          <select
            data-testid="runs-status-filter"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ fontSize: "13px", padding: "6px 8px" }}
          >
            <option value="">All</option>
            <option value="queued">Queued</option>
            <option value="running">Running</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </label>
      </div>

      {actionError && <div className="alert alert--error" data-testid="runs-action-error">{actionError}</div>}

      {runs.length === 0 ? (
        <div className="empty-state" data-testid="runs-empty">
          <div className="empty-state__icon">◌</div>
          <div className="empty-state__text">No benchmark runs yet</div>
          <div style={{ fontSize: "12px", marginTop: "4px" }}>Start a New benchmark run to measure detector robustness.</div>
        </div>
      ) : (
        <table className="table" data-testid="runs-table">
          <thead>
            <tr>
              <th>Run</th>
              <th>Status</th>
              <th>Detector / config</th>
              <th style={{ textAlign: "right" }}>Progress</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const p = r.progress;
              const progressLabel = p && p.experiments_total > 0
                ? `${p.experiments_completed}/${p.experiments_total}`
                : r.status === "queued" ? "queued" : "—";
              return (
                <tr key={r.run_id} data-testid={`run-row-${r.run_id}`}>
                  <td>
                    <button
                      className="btn btn--sm"
                      data-testid={`run-open-${r.run_id}`}
                      onClick={() => props.onOpen(r.run_id)}
                    >
                      <span className="mono" style={{ fontSize: "12px" }}>{r.run_id}</span>
                    </button>
                  </td>
                  <td><StatusBadge status={r.status} /></td>
                  <td style={{ fontSize: "12px" }}>
                    <strong>{r.config.detector}</strong>
                    {r.config.config && <div className="mono" style={{ color: "var(--color-text-secondary)" }}>{r.config.config}</div>}
                  </td>
                  <td style={{ textAlign: "right" }} data-testid={`run-progress-${r.run_id}`}>{progressLabel}</td>
                  <td style={{ fontSize: "12px" }}>{r.created_at}</td>
                  <td>
                    {(r.status === "queued" || r.status === "running") && (
                      <button
                        className="btn btn--sm btn--danger"
                        data-testid={`run-cancel-${r.run_id}`}
                        disabled={acting === r.run_id}
                        onClick={() => void act(r.run_id, "cancel")}
                      >
                        Cancel
                      </button>
                    )}
                    {(r.status === "failed" || r.status === "cancelled") && (
                      <button
                        className="btn btn--sm"
                        data-testid={`run-retry-${r.run_id}`}
                        disabled={acting === r.run_id}
                        onClick={() => void act(r.run_id, "retry")}
                      >
                        Retry
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* New run form                                                        */
/* ------------------------------------------------------------------ */

function parseLengths(raw: string): { lengths: number[]; error: string } {
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  const lengths: number[] = [];
  for (const part of parts) {
    if (!/^\d+$/.test(part)) return { lengths: [], error: `Invalid length: "${part}" (integers only)` };
    lengths.push(parseInt(part, 10));
  }
  if (lengths.length === 0) return { lengths: [], error: "Enter at least one length" };
  return { lengths, error: "" };
}

function NewRunForm(props: { onCreated: (runId: string) => void; onCancel: () => void }) {
  const { client } = useConfig();
  const [options, setOptions] = useState<BenchmarkOptions | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [malformed, setMalformed] = useState(false);

  const [detector, setDetector] = useState("");
  const [configPath, setConfigPath] = useState("");
  const [profile, setProfile] = useState("");
  const [selectedTransforms, setSelectedTransforms] = useState<string[]>([]);
  const [lengthsRaw, setLengthsRaw] = useState("50,100");
  const [samples, setSamples] = useState("5");
  const [seed, setSeed] = useState("42");
  const [formErrors, setFormErrors] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    (async () => {
      try {
        const data: unknown = await client.getBenchmarkOptions();
        if (cancelled) return;
        if (!isBenchmarkOptions(data)) {
          setMalformed(true);
        } else {
          setOptions(data);
          if (data.detectors.length > 0) setDetector(data.detectors[0].name);
          const allSafe = data.profiles.find((p) => p.name === "all_safe");
          setProfile(allSafe ? allSafe.name : "");
        }
      } catch (e: unknown) {
        if (!cancelled) setError(errorText(e, "Failed to load benchmark options"));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [client]);

  if (loading) return <div className="loading"><span className="spinner" /> Loading benchmark options...</div>;
  if (error) return <div className="alert alert--error" data-testid="newrun-error">{error}</div>;
  if (malformed || !options) {
    return (
      <div className="alert alert--error" data-testid="newrun-malformed">
        The server returned benchmark options in an unrecognized format. Please check the server version.
      </div>
    );
  }

  const constraints = options.constraints;
  const selectedDetector = options.detectors.find((d) => d.name === detector);
  const needsConfig = selectedDetector?.requires_config ?? false;

  function validate(): { input: BenchmarkRunCreateInput | null; errors: string[] } {
    const errs: string[] = [];
    if (!detector) errs.push("Choose a detector");
    if (needsConfig && !configPath.trim()) errs.push("Config path is required for this detector");
    const { lengths, error: lenErr } = parseLengths(lengthsRaw);
    if (lenErr) errs.push(lenErr);
    else {
      if (lengths.length > constraints.max_lengths_count) errs.push(`At most ${constraints.max_lengths_count} lengths`);
      for (const l of lengths) {
        if (l < 1 || l > constraints.max_text_length) errs.push(`Lengths must be within [1, ${constraints.max_text_length}]`);
      }
    }
    const nSamples = /^\d+$/.test(samples.trim()) ? parseInt(samples.trim(), 10) : NaN;
    if (!Number.isFinite(nSamples) || nSamples < 1 || nSamples > constraints.max_samples) {
      errs.push(`Samples must be within [1, ${constraints.max_samples}]`);
    }
    if (!/^-?\d+$/.test(seed.trim())) errs.push("Seed must be an integer");
    if (selectedTransforms.length > constraints.max_transforms) {
      errs.push(`At most ${constraints.max_transforms} transforms`);
    }
    if (errs.length > 0) return { input: null, errors: errs };
    return {
      input: {
        detector,
        config: configPath.trim() || null,
        profile: profile || null,
        transforms: selectedTransforms.length > 0 ? selectedTransforms : null,
        lengths,
        samples: nSamples,
        seed: parseInt(seed.trim(), 10),
      },
      errors: [],
    };
  }

  const preview = validate();
  const transformsByCategory = new Map<string, { name: string; description: string }[]>();
  for (const t of options.transforms) {
    const group = transformsByCategory.get(t.category) ?? [];
    group.push(t);
    transformsByCategory.set(t.category, group);
  }

  async function submit() {
    if (!client || !preview.input) return;
    setSubmitting(true);
    setFormErrors([]);
    try {
      const created = await client.createBenchmarkRun(preview.input);
      props.onCreated(created.run_id);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.errors && e.errors.length > 0) {
        setFormErrors(e.errors.map((fe) => `${fe.field}: ${fe.message}`));
      } else {
        setFormErrors([errorText(e, "Failed to create benchmark run")]);
      }
    } finally {
      setSubmitting(false);
    }
  }

  function toggleTransform(name: string) {
    setSelectedTransforms((prev) => prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]);
  }

  return (
    <div data-testid="newrun-form">
      <h3 style={{ fontSize: "15px", fontWeight: 700, marginBottom: "12px" }}>New benchmark run</h3>

      {formErrors.length > 0 && (
        <div className="alert alert--error" data-testid="newrun-errors">
          <ul style={{ margin: 0, paddingLeft: "18px" }}>
            {formErrors.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}

      <div className="card">
        <div className="filter-bar" style={{ marginBottom: 0 }}>
          <label>Detector
            <select data-testid="newrun-detector" value={detector} onChange={(e) => setDetector(e.target.value)}>
              {options.detectors.map((d) => (
                <option key={d.name} value={d.name}>{d.display_name} ({d.name})</option>
              ))}
            </select>
          </label>
          <label>Model config path{needsConfig ? " (required)" : ""}
            <input
              data-testid="newrun-config"
              type="text"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              placeholder={needsConfig ? "configs/kgw.hf.example.json" : "optional"}
              style={{ fontSize: "13px", padding: "6px 8px", minWidth: "260px" }}
            />
          </label>
          <label>Transform profile
            <select data-testid="newrun-profile" value={profile} onChange={(e) => setProfile(e.target.value)}>
              <option value="">All baseline transforms</option>
              {options.profiles.map((p) => (
                <option key={p.name} value={p.name}>{p.name}</option>
              ))}
            </select>
          </label>
          <label>Text lengths (comma-separated)
            <input
              data-testid="newrun-lengths"
              type="text"
              value={lengthsRaw}
              onChange={(e) => setLengthsRaw(e.target.value)}
              style={{ fontSize: "13px", padding: "6px 8px" }}
            />
          </label>
          <label>Samples per length
            <input
              data-testid="newrun-samples"
              type="text"
              value={samples}
              onChange={(e) => setSamples(e.target.value)}
              style={{ fontSize: "13px", padding: "6px 8px", width: "90px" }}
            />
          </label>
          <label>Seed
            <input
              data-testid="newrun-seed"
              type="text"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              style={{ fontSize: "13px", padding: "6px 8px", width: "110px" }}
            />
          </label>
        </div>

        <details style={{ marginTop: "12px" }}>
          <summary style={{ fontSize: "13px", cursor: "pointer" }}>
            Individual transforms (optional — overrides profile when selected)
          </summary>
          <div style={{ fontSize: "12px", color: "var(--color-text-secondary)", margin: "4px 0 8px" }}>
            Selected: {selectedTransforms.length === 0 ? "none (profile applies)" : selectedTransforms.join(", ")}
          </div>
          {[...transformsByCategory.entries()].map(([cat, list]) => (
            <div key={cat} style={{ marginBottom: "8px" }}>
              <div style={{ fontSize: "12px", fontWeight: 600 }}>{cat}</div>
              {list.map((t) => (
                <label key={t.name} style={{ display: "inline-block", fontSize: "12px", marginRight: "12px" }}>
                  <input
                    type="checkbox"
                    data-testid={`newrun-transform-${t.name}`}
                    checked={selectedTransforms.includes(t.name)}
                    onChange={() => toggleTransform(t.name)}
                  />{" "}
                  <span className="mono">{t.name}</span>
                </label>
              ))}
            </div>
          ))}
        </details>
      </div>

      <div className="card" style={{ marginTop: "12px" }} data-testid="newrun-summary">
        <h4 style={{ fontSize: "13px", fontWeight: 600, marginBottom: "8px" }}>Run summary</h4>
        {preview.input ? (
          <div style={{ fontSize: "13px" }}>
            Detector <span className="mono">{preview.input.detector}</span>
            {preview.input.config && <> with config <span className="mono">{preview.input.config}</span></>}
            {", "}
            {preview.input.transforms
              ? `${preview.input.transforms.length} explicit transform(s)`
              : preview.input.profile
                ? <>profile <span className="mono">{preview.input.profile}</span></>
                : "all baseline transforms"}
            {", "}lengths [{preview.input.lengths.join(", ")}]
            {", "}{preview.input.samples} sample(s) each
            {", "}seed {preview.input.seed}.
            {" "}Runs execute asynchronously with server-side validation.
          </div>
        ) : (
          <div style={{ fontSize: "13px", color: "var(--color-warning)" }}>
            Fix the form errors above to preview this run.
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
        <button
          className="btn btn--primary"
          data-testid="newrun-submit"
          disabled={submitting || !preview.input}
          onClick={() => void submit()}
        >
          {submitting ? "Starting..." : "Start benchmark run"}
        </button>
        <button className="btn" data-testid="newrun-cancel" onClick={props.onCancel}>
          Back to runs
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Run detail                                                          */
/* ------------------------------------------------------------------ */

function RunDetail(props: {
  runId: string;
  onBack: () => void;
  onViewResults: (focus: RobustnessFocus) => void;
}) {
  const { client } = useConfig();
  const [run, setRun] = useState<BenchmarkRunResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [malformed, setMalformed] = useState(false);
  const [actionError, setActionError] = useState("");

  const refresh = useCallback(async () => {
    if (!client) return null;
    const data: unknown = await client.getBenchmarkRun(props.runId);
    if (!isBenchmarkRunResponse(data)) {
      setMalformed(true);
      return null;
    }
    setRun(data);
    return data;
  }, [client, props.runId]);

  useEffect(() => {
    if (!client) return;
    let cancelled = false;
    setLoading(true);
    refresh()
      .catch((e: unknown) => { if (!cancelled) setError(errorText(e, "Failed to load run")); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [client, refresh]);

  const active = run !== null && !isTerminal(run.status);
  useEffect(() => {
    if (!client || !active) return;
    const timer = setInterval(() => {
      void refresh().catch((e: unknown) => setError(errorText(e, "Failed to refresh run")));
    }, BENCHMARK_POLL_MS);
    return () => clearInterval(timer);
  }, [client, active, refresh]);

  async function act(kind: "cancel" | "retry") {
    if (!client) return;
    setActionError("");
    try {
      if (kind === "cancel") await client.cancelBenchmarkRun(props.runId);
      else await client.retryBenchmarkRun(props.runId);
      await refresh();
    } catch (e: unknown) {
      setActionError(errorText(e, `Failed to ${kind} run`));
    }
  }

  if (loading) return <div className="loading"><span className="spinner" /> Loading run...</div>;
  if (error) return <div className="alert alert--error" data-testid="rundetail-error">{error}</div>;
  if (malformed || !run) {
    return (
      <div className="alert alert--error" data-testid="rundetail-malformed">
        The server returned run data in an unrecognized format. Please check the server version.
      </div>
    );
  }

  const p = run.progress;
  const pct = p && p.experiments_total > 0
    ? Math.round((100 * p.experiments_completed) / p.experiments_total)
    : 0;
  const canViewResults = run.status === "completed" && run.result?.has_results;

  return (
    <div data-testid="rundetail">
      <button className="btn btn--sm" data-testid="rundetail-back" onClick={props.onBack} style={{ marginBottom: "12px" }}>
        ← Back to runs
      </button>
      <h3 style={{ fontSize: "15px", fontWeight: 700, marginBottom: "4px" }}>
        Run <span className="mono">{run.run_id}</span> <StatusBadge status={run.status} />
      </h3>

      {actionError && <div className="alert alert--error" data-testid="rundetail-action-error">{actionError}</div>}
      {run.error_message && (
        <div className="alert alert--error" data-testid="rundetail-run-error">{run.error_message}</div>
      )}

      <div className="grid-stats">
        <StatCard label="Status" value={run.status} />
        <StatCard
          label="Progress"
          value={p && p.experiments_total > 0
            ? `${p.experiments_completed}/${p.experiments_total} (${pct}%)`
            : run.status}
        />
        <StatCard label="Duration" value={formatDuration(run.duration_ms)} />
        <StatCard label="Retries" value={run.retry_count} />
      </div>

      {p && p.experiments_total > 0 && (
        <div style={{ marginTop: "12px" }} data-testid="rundetail-progressbar">
          <div style={{ height: "8px", background: "var(--color-border-light)", borderRadius: "4px" }}>
            <div style={{ height: "100%", width: `${pct}%`, background: "var(--color-primary)", borderRadius: "4px" }} />
          </div>
          {p.experiments_failed > 0 && (
            <div style={{ fontSize: "12px", color: "var(--color-danger)", marginTop: "4px" }}>
              {p.experiments_failed} experiment(s) failed
            </div>
          )}
        </div>
      )}

      <div className="card" style={{ marginTop: "16px" }}>
        <h4 style={{ fontSize: "13px", fontWeight: 600, marginBottom: "8px" }}>Configuration</h4>
        <table className="table" data-testid="rundetail-config">
          <tbody>
            <tr><td>Detector</td><td className="mono">{run.config.detector}</td></tr>
            <tr><td>Config</td><td className="mono">{run.config.config ?? "—"}</td></tr>
            <tr><td>Profile</td><td className="mono">{run.config.profile ?? "—"}</td></tr>
            <tr><td>Transforms</td><td className="mono">{run.config.transforms ? run.config.transforms.join(", ") : "—"}</td></tr>
            <tr><td>Lengths</td><td className="mono">{run.config.lengths.join(", ")}</td></tr>
            <tr><td>Samples / seed</td><td className="mono">{run.config.samples} / {run.config.seed}</td></tr>
          </tbody>
        </table>
      </div>

      {run.result && (
        <div className="card" style={{ marginTop: "16px" }} data-testid="rundetail-result">
          <h4 style={{ fontSize: "13px", fontWeight: 600, marginBottom: "8px" }}>Result summary</h4>
          <div style={{ fontSize: "13px" }}>
            {run.result.experiments_completed}/{run.result.experiments_total} experiment(s) completed
            {run.result.experiments_failed > 0 && `, ${run.result.experiments_failed} failed`}.
            {run.result.has_results ? " Artifacts available." : " No result artifacts."}
          </div>
        </div>
      )}

      <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
        {(run.status === "queued" || run.status === "running") && (
          <button className="btn btn--danger" data-testid="rundetail-cancel" onClick={() => void act("cancel")}>
            Cancel run
          </button>
        )}
        {(run.status === "failed" || run.status === "cancelled") && (
          <button className="btn" data-testid="rundetail-retry" onClick={() => void act("retry")}>
            Retry run
          </button>
        )}
        {canViewResults && (
          <button
            className="btn btn--primary"
            data-testid="rundetail-view-results"
            onClick={() => props.onViewResults({
              detector: run.config.detector,
              config: run.config.config ?? undefined,
              runId: run.run_id,
              token: Date.now(),
            })}
          >
            View results in Robustness
          </button>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export function BenchmarksPage(props: { onViewResults: (focus: RobustnessFocus) => void }) {
  const [view, setView] = useState<"list" | "new" | "detail">("list");
  const [selectedId, setSelectedId] = useState("");

  return (
    <div>
      <h2 className="page-header">Benchmarks</h2>
      <RobustnessDisclaimer />
      {view === "list" && (
        <RunsList
          onNew={() => setView("new")}
          onOpen={(id) => { setSelectedId(id); setView("detail"); }}
        />
      )}
      {view === "new" && (
        <NewRunForm
          onCreated={(id) => { setSelectedId(id); setView("detail"); }}
          onCancel={() => setView("list")}
        />
      )}
      {view === "detail" && (
        <RunDetail
          runId={selectedId}
          onBack={() => setView("list")}
          onViewResults={props.onViewResults}
        />
      )}
    </div>
  );
}
