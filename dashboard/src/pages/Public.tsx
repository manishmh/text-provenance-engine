import { useEffect, useState } from "react";
import { supabase, supabaseConfigured } from "../lib/supabase";
import type { SaaSSession } from "../hooks/useSession";
import type { PublicAnalyzeResponse, QuotaInfo } from "../types/api";
import { ApiError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

interface Props {
  session: SaaSSession;
  onEnterWorkspace: () => void;
  onDeveloperSignIn: () => void;
}

export function PublicSite({ session, onEnterWorkspace, onDeveloperSignIn }: Props) {
  const { client, quota, refresh } = session;
  const [text, setText] = useState("");
  const [result, setResult] = useState<PublicAnalyzeResponse | null>(null);
  const [localQuota, setLocalQuota] = useState<QuotaInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [authOpen, setAuthOpen] = useState(false);

  const shownQuota = localQuota ?? quota;
  const maxChars = shownQuota?.max_chars_per_analysis
    ?? session.me?.entitlements.max_chars_per_analysis
    ?? 5000;

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const q = await client.publicQuota(session.token ?? undefined);
        if (!cancelled) setLocalQuota(q);
      } catch {
        /* quota indicator stays hidden until first use */
      }
    })();
    return () => { cancelled = true; };
  }, [client, session.token]);

  const analyze = async () => {
    if (!text.trim() || loading) return;
    setError("");
    setLoading(true);
    try {
      const r = await client.publicAnalyze(text, session.token ?? undefined);
      setResult(r);
      setLocalQuota(r.quota);
    } catch (e: unknown) {
      if (e instanceof ApiError && e.status === 429) {
        setError("You have used your 2 free analyses for today. Sign in for higher limits, or try again tomorrow.");
        refresh();
        const q = await client.publicQuota(session.token ?? undefined).catch(() => undefined);
        if (q) setLocalQuota(q);
      } else {
        setError(e instanceof ApiError ? e.userMessage : "Analysis failed; please try again");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="site">
      <header className="site__header">
        <a className="site__logo" href="#top">Provenance Engine</a>
        <nav className="site__nav">
          <a href="#how">How it works</a>
          <a href="#detection">Detection</a>
          <a href="#pricing">Pricing</a>
          <a href="#faq">FAQ</a>
        </nav>
        <div className="site__actions">
          {session.status === "user" ? (
            <button className="btn btn--primary btn--sm" onClick={onEnterWorkspace} data-testid="site-open-workspace">
              Open workspace
            </button>
          ) : (
            <>
              <button className="btn btn--sm" onClick={() => setAuthOpen(true)} data-testid="site-signin">
                Sign in
              </button>
              <button className="btn btn--primary btn--sm" onClick={() => setAuthOpen(true)} data-testid="site-get-started">
                Get started
              </button>
            </>
          )}
        </div>
      </header>

      <main id="top">
        <section className="site__hero">
          <div className="site__hero-inner">
            <h1 className="site__headline">Check text for hidden Unicode provenance signals</h1>
            <p className="site__lede">
              The public analyzer performs a fast, deterministic scan for invisible or
              unusual Unicode characters. KGW and SynthID checks require the matching
              watermark configuration and are not run on arbitrary pasted text. Free for
              your first 2 analyses every day. No account needed.
            </p>

            <div className="card site__tool">
              <label className="label" htmlFor="site-text">Text to analyze</label>
              <textarea
                id="site-text"
                data-testid="site-text"
                className="textarea site__textarea"
                placeholder="Paste text here…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                maxLength={maxChars}
              />
              <div className="site__tool-row">
                <span className="site__count" data-testid="site-char-count">{text.length} / {maxChars} chars</span>
                {shownQuota && (
                  <span className="site__quota" data-testid="site-quota">
                    {shownQuota.remaining} of {shownQuota.limit} free analyses left today
                  </span>
                )}
                <button
                  className="btn btn--primary"
                  data-testid="site-analyze"
                  disabled={!text.trim() || loading}
                  onClick={analyze}
                >
                  {loading ? "Analyzing…" : "Analyze"}
                </button>
              </div>
              {error && <div className="alert alert--error" data-testid="site-error">{error}</div>}
              <p className="site__privacy-note">
                Submitted text is analyzed in memory and never stored. Only aggregate,
                anonymous usage counts are kept for the daily free limit.
              </p>
            </div>

            {result && (
              <div className="card site__result" data-testid="site-result">
                <div className="card__head">
                  <h3 className="card__title">Result</h3>
                  <StatusBadge status={
                    result.overall_result === "signal_detected" ? "detected"
                      : result.overall_result === "no_supported_signal_detected" ? "not_detected"
                        : "inconclusive"
                  } />
                </div>
                <p className="site__verdict">{result.verdict}</p>
                <h4>Signals checked</h4>
                <ul className="home-list">
                  {result.signals_checked.map((s) => (
                    <li key={s.detector}>
                      <span className="home-list__id">{s.display_name}</span>
                      <span className="home-list__meta">
                        {s.status === "detected" ? "hidden Unicode signal detected"
                          : s.status === "not_detected" ? "no hidden Unicode signal detected"
                            : "inconclusive"}
                        {" · "}{s.evidence.join(" ")}
                      </span>
                    </li>
                  ))}
                </ul>
                <h4>Checks not run</h4>
                <ul className="home-list" data-testid="site-unavailable-detectors">
                  {result.unavailable_detectors.map((d) => (
                    <li key={d.detector}>
                      <span className="home-list__id">{d.display_name}</span>
                      <span className="home-list__meta">
                        {d.status === "not_applicable" ? "benchmark-only" : "configuration required"}
                        {" · "}{d.reason}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="site__disclaimer">{result.disclaimer}</p>
                <div className="site__cta-row">
                  <span>{result.upgrade_hint}</span>
                  {session.status === "user" ? (
                    <button className="btn btn--sm" onClick={onEnterWorkspace}>Open full report</button>
                  ) : (
                    <button className="btn btn--sm" onClick={() => setAuthOpen(true)} data-testid="site-result-signin">
                      Sign in
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="site__section" id="detection">
          <h2>What the public analyzer checks</h2>
          <p>
            This page checks for deterministic, machine-readable Unicode artifacts such as
            invisible controls and unusual spacing. It does not run statistical KGW or
            SynthID verification without the key and generation configuration those methods
            require, and it does not classify text as AI-generated or human-written.
          </p>
        </section>

        <section className="site__section" id="how">
          <h2>How it works</h2>
          <ol className="site__steps">
            <li><strong>Paste text.</strong> Analysis runs on our API; nothing is stored.</li>
            <li><strong>The public check runs.</strong> The cheap Unicode detector reports detected, not detected, or inconclusive.</li>
            <li><strong>Boundaries stay visible.</strong> Config-specific and benchmark-only detectors are listed as not run.</li>
          </ol>
        </section>

        <section className="site__section">
          <h2>Supported detector families</h2>
          <div className="site__grid3">
            <div className="card">
              <h3 className="card__title">Unicode artifacts</h3>
              <p>Deterministic scan for invisible or unusual characters introduced by tooling. Works fully offline, no configuration needed.</p>
            </div>
            <div className="card">
              <h3 className="card__title">KGW watermarks</h3>
              <p>Known-configuration statistical verification only. It needs the matching key, parameters, variant, and tokenizer, so the public analyzer does not run it.</p>
            </div>
            <div className="card">
              <h3 className="card__title">SynthID watermarks</h3>
              <p>Known-configuration statistical verification only. It needs matching watermark and tokenizer settings, so the public analyzer does not run it.</p>
            </div>
          </div>
        </section>

        <section className="site__section">
          <h2>Why provenance matters</h2>
          <p>
            Knowing whether a passage carries a known machine signal helps editors, researchers,
            and platform teams triage content responsibly. Signals are evidence — not proof of
            authorship — and every result ships with its limitations attached.
          </p>
        </section>

        <section className="site__section">
          <h2>Privacy</h2>
          <ul className="site__list">
            <li>Anonymous use needs no account; identity is a random cookie, not a fingerprint.</li>
            <li>Submitted text is analyzed in memory and never persisted.</li>
            <li>Only aggregate counts (how many analyses, how many characters) are stored.</li>
            <li>No watermark keys, secrets, or raw text ever reach the browser bundle.</li>
          </ul>
        </section>

        <section className="site__section" id="pricing">
          <h2>Free vs Pro</h2>
          <div className="site__grid2">
            <div className="card">
              <h3 className="card__title">Free</h3>
              <p className="site__price">2 analyses / day</p>
              <ul className="site__list">
                <li>Public Unicode provenance scan with explicit unavailable checks</li>
                <li>Signed-in free accounts get higher daily limits plus history</li>
              </ul>
            </div>
            <div className="card">
              <h3 className="card__title">Pro</h3>
              <p className="site__price">High-volume limits + workspace</p>
              <ul className="site__list">
                <li>Full detailed reports, configured detector workflows, and analysis history</li>
                <li>Robustness benchmarks, detector management, and API access</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="site__section" id="faq">
          <h2>FAQ</h2>
          <details><summary>Can this prove who wrote a text?</summary><p>No. It reports detector-specific signals. A clean public Unicode result does not imply human authorship or prove that every watermark is absent.</p></details>
          <details><summary>Does it remove watermarks?</summary><p>No. This product performs detection and provenance analysis only. It cannot modify text or remove watermarks.</p></details>
          <details><summary>Do I need an account?</summary><p>No. Anonymous visitors get 2 free analyses per day. Signing in raises your limits and unlocks history.</p></details>
          <details><summary>Is my text stored?</summary><p>No. Text is analyzed in memory and discarded. Only anonymous aggregate counts are kept for quota enforcement.</p></details>
          <details><summary>Does the public check test KGW or SynthID?</summary><p>No. Those statistical detectors need the matching secret key and generation configuration. Authenticated workflows can verify a known configuration where access and deployment setup allow it.</p></details>
          <details><summary>Which models are attributed?</summary><p>None automatically. Configured reference detectors test compatibility with a specific setup; results never claim universal or official vendor attribution.</p></details>
        </section>
      </main>

      <footer className="site__footer">
        <span>Provenance Engine — research-oriented text provenance analysis.</span>
        <span>
          <button className="link-btn" onClick={onDeveloperSignIn}>Developer sign-in</button>
        </span>
      </footer>

      {authOpen && (
        <AuthModal
          onClose={() => setAuthOpen(false)}
          onSignedIn={() => { setAuthOpen(false); refresh(); }}
        />
      )}
    </div>
  );
}

function AuthModal({ onClose, onSignedIn }: { onClose: () => void; onSignedIn: () => void }) {
  const [mode, setMode] = useState<"password" | "magic">("password");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (!supabaseConfigured || !supabase) {
    return (
      <div className="modal" data-testid="site-auth">
        <div className="card modal__card">
          <h3 className="card__title">Sign in is not configured</h3>
          <p className="settings__hint">This deployment has no authentication provider set up.</p>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  const run = async (fn: () => Promise<{ error: { message: string } | null }>, okMsg: string) => {
    setError("");
    setMessage("");
    setBusy(true);
    try {
      const { error: err } = await fn();
      if (err) setError(err.message);
      else {
        if (okMsg) setMessage(okMsg);
        else onSignedIn();
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal" data-testid="site-auth">
      <div className="card modal__card">
        <div className="card__head">
          <h3 className="card__title">Sign in</h3>
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="site__tabs">
          <button className={`btn btn--sm${mode === "password" ? " btn--primary" : ""}`} onClick={() => setMode("password")}>Password</button>
          <button className={`btn btn--sm${mode === "magic" ? " btn--primary" : ""}`} onClick={() => setMode("magic")}>Email link</button>
        </div>
        <label className="label" htmlFor="auth-email">Email</label>
        <input
          id="auth-email" data-testid="site-auth-email" className="input" type="email"
          value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email"
        />
        {mode === "password" && (
          <>
            <label className="label" htmlFor="auth-password" style={{ marginTop: "12px" }}>Password</label>
            <input
              id="auth-password" data-testid="site-auth-password" className="input" type="password"
              value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password"
            />
          </>
        )}
        {error && <div className="alert alert--error" data-testid="site-auth-error">{error}</div>}
        {message && <div className="alert alert--success">{message}</div>}
        <div className="site__cta-row" style={{ marginTop: "16px" }}>
          {mode === "password" ? (
            <>
              <button
                className="btn btn--primary" data-testid="site-auth-submit" disabled={busy || !email || !password}
                onClick={() => run(
                  () => supabase!.auth.signInWithPassword({ email, password }), "")}
              >
                Sign in
              </button>
              <button
                className="btn" disabled={busy || !email || !password}
                onClick={() => run(
                  () => supabase!.auth.signUp({ email, password }),
                  "Account created. You can now sign in.")}
              >
                Create account
              </button>
            </>
          ) : (
            <button
              className="btn btn--primary" data-testid="site-auth-submit" disabled={busy || !email}
              onClick={() => run(
                () => supabase!.auth.signInWithOtp({ email }),
                "Check your email for a sign-in link.")}
            >
              Send magic link
            </button>
          )}
        </div>
        <div style={{ marginTop: "12px" }}>
          <button
            className="btn" data-testid="site-auth-google" disabled={busy}
            onClick={() => run(
              () => supabase!.auth.signInWithOAuth({ provider: "google" }), "")}
          >
            Continue with Google
          </button>
        </div>
      </div>
    </div>
  );
}
