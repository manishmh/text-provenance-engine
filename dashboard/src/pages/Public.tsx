import { useEffect, useRef, useState } from "react";
import { supabase, supabaseConfigured } from "../lib/supabase";
import {
  authRedirectUrl,
  oauthCallbackError,
  rememberAuthReturnTo,
} from "../lib/authNavigation";
import type { SaaSSession } from "../hooks/useSession";
import type { BillingStatusResponse, BillingSubscriptionResponse, PublicAnalyzeResponse, QuotaInfo } from "../types/api";
import { ApiError } from "../api/client";
import { StatusBadge } from "../components/StatusBadge";

interface Props {
  session: SaaSSession;
  onEnterWorkspace: () => void;
  onDeveloperSignIn: () => void;
  onSignOut?: () => void;
}

type RazorpayInstance = { open: () => void; on: (event: string, callback: () => void) => void };
type RazorpayConstructor = new (options: Record<string, unknown>) => RazorpayInstance;

function loadRazorpayCheckout(): Promise<RazorpayConstructor> {
  const loaded = (window as Window & { Razorpay?: RazorpayConstructor }).Razorpay;
  if (loaded) return Promise.resolve(loaded);
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>('script[data-razorpay-checkout="true"]');
    if (existing) {
      existing.addEventListener("load", () => resolve((window as Window & { Razorpay?: RazorpayConstructor }).Razorpay!), { once: true });
      existing.addEventListener("error", () => reject(new Error("Razorpay Checkout could not be loaded")), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.async = true;
    script.dataset.razorpayCheckout = "true";
    script.onload = () => {
      const Razorpay = (window as Window & { Razorpay?: RazorpayConstructor }).Razorpay;
      Razorpay ? resolve(Razorpay) : reject(new Error("Razorpay Checkout could not be loaded"));
    };
    script.onerror = () => reject(new Error("Razorpay Checkout could not be loaded"));
    document.head.appendChild(script);
  });
}

export function PublicSite({ session, onEnterWorkspace, onDeveloperSignIn, onSignOut }: Props) {
  const { client, quota, refresh } = session;
  const [text, setText] = useState("");
  const [result, setResult] = useState<PublicAnalyzeResponse | null>(null);
  const [localQuota, setLocalQuota] = useState<QuotaInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [authOpen, setAuthOpen] = useState(() => window.location.hash.includes("auth=signin"));
  const [authCallbackMessage, setAuthCallbackMessage] = useState("");
  const [billing, setBilling] = useState<BillingSubscriptionResponse | null>(null);
  const [billingStatus, setBillingStatus] = useState<BillingStatusResponse | null>(null);
  const [billingBusy, setBillingBusy] = useState(false);
  const [billingMessage, setBillingMessage] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const checkoutIntentConsumed = useRef(false);

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

  useEffect(() => {
    if (session.status !== "user" || !session.token) return;
    client.getBillingSubscription(session.token).then(setBilling).catch(() => undefined);
  }, [client, session.status, session.token]);

  useEffect(() => {
    client.billingStatus().then(setBillingStatus).catch(() => setBillingStatus({ provider: "razorpay", configured: false, checkout_mode: null }));
  }, [client]);

  useEffect(() => {
    const checkout = new URLSearchParams(window.location.search).get("checkout");
    if (checkout === "cancelled") setBillingMessage("Checkout was cancelled. Your plan has not changed.");
    if (checkout === "success") {
      setBillingMessage("Payment received. Pro access will appear after the signed webhook is processed.");
      refresh();
    }
  }, [refresh]);

  useEffect(() => {
    const openRequestedAuth = () => setAuthOpen(window.location.hash.includes("auth=signin"));
    window.addEventListener("hashchange", openRequestedAuth);
    return () => window.removeEventListener("hashchange", openRequestedAuth);
  }, []);

  useEffect(() => {
    const callbackError = oauthCallbackError();
    if (!callbackError) return;
    setAuthCallbackMessage(callbackError);
    setAuthOpen(true);
    // Do not leave a provider error in the address bar where it could reopen
    // the dialog on a later render. Preserve the hash route, if any.
    window.history.replaceState({}, document.title, `${window.location.pathname}${window.location.hash}`);
  }, []);

  useEffect(() => {
    if (!session.authIssue) return;
    setAuthCallbackMessage(session.authIssue);
    setAuthOpen(true);
  }, [session.authIssue]);

  const upgrade = async () => {
    if (session.status !== "user" || !session.token) { setAuthOpen(true); return; }
    setBillingBusy(true); setBillingMessage("");
    try {
      const checkout = await client.createBillingCheckout(session.token);
      if (checkout.provider === "razorpay" && checkout.checkout_data) {
        const key = checkout.checkout_data.key_id;
        const subscriptionId = checkout.checkout_data.subscription_id;
        if (typeof key !== "string" || typeof subscriptionId !== "string") throw new Error("Invalid Razorpay checkout response");
        const Razorpay = await loadRazorpayCheckout();
        const razorpay = new Razorpay({
          key, subscription_id: subscriptionId, name: checkout.checkout_data.name,
          description: checkout.checkout_data.description,
          handler: () => {
            // A browser callback only starts the wait state. The webhook-backed
            // subscription endpoint remains the sole source of Pro access.
            setBillingMessage("Payment received. Verifying your subscription…");
            client.getBillingSubscription(session.token!).then(setBilling).catch(() => undefined);
            refresh();
          },
          modal: { ondismiss: () => setBillingMessage("Checkout was cancelled. Your plan has not changed.") },
        });
        razorpay.on("payment.failed", () => setBillingMessage("Payment failed or was cancelled. Your plan has not changed."));
        razorpay.open();
        return;
      }
      throw new Error("Unsupported billing checkout response");
    } catch (e) {
      setBillingMessage(e instanceof ApiError ? e.userMessage : "Could not start checkout.");
    } finally { setBillingBusy(false); }
  };

  // Pricing uses this hash intent so it can share the one provider-neutral
  // checkout implementation without duplicating a second checkout page.
  useEffect(() => {
    if (!checkoutIntentConsumed.current && session.status === "user" && session.token && window.location.hash.includes("upgrade=pro")) {
      checkoutIntentConsumed.current = true;
      void upgrade();
    }
    // The intent is only meaningful when the authenticated session changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.status, session.token]);

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
        <a className="site__logo" href="#/"><span className="site__logo-mark">P</span>Provenance Engine</a>
        <button className="site__menu-button" aria-expanded={mobileMenuOpen} aria-controls="site-navigation" onClick={() => setMobileMenuOpen((open) => !open)}>Menu</button>
        <nav id="site-navigation" className={`site__nav${mobileMenuOpen ? " site__nav--open" : ""}`} aria-label="Product navigation">
          <a href="#detection" onClick={() => setMobileMenuOpen(false)}>Product</a>
          <a href="#/methodology" onClick={() => setMobileMenuOpen(false)}>Methodology</a>
          <a href="#/pricing" onClick={() => setMobileMenuOpen(false)}>Pricing</a>
          <a href="#faq" onClick={() => setMobileMenuOpen(false)}>FAQ</a>
        </nav>
        <div className="site__actions">
          {session.status === "user" ? (
            <>
              <button className="btn btn--sm" onClick={onSignOut}>Sign out</button>
              <button className="btn btn--primary btn--sm" onClick={onEnterWorkspace} data-testid="site-open-workspace">Open workspace</button>
            </>
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
          <div className="site__hero-grid">
            <div className="site__hero-copy">
              <p className="site__eyebrow">Text provenance inspection</p>
              <h1 className="site__headline">Inspect text for hidden provenance signals.</h1>
              <p className="site__lede">Find invisible Unicode artifacts and review evidence-backed signals—without relying on generic AI-authorship guesses.</p>
              <div className="site__hero-actions">
                <a className="btn btn--primary btn--hero" href="#analyzer">Analyze text</a>
                <a className="site__text-link" href="#/methodology">How it works <span aria-hidden="true">→</span></a>
              </div>
              <div className="site__trust-row">
                <span><b aria-hidden="true">✓</b> No account needed</span>
                <span><b aria-hidden="true">✓</b> Text analyzed in memory</span>
                <span><b aria-hidden="true">✓</b> Clear limitations</span>
              </div>
              <div className="site__hero-signal" aria-hidden="true">
                <span>Visible text</span><i>+</i><span className="site__hidden-glyph">invisible signal</span><i>→</i><strong>evidence</strong>
              </div>
            </div>

            <div className="card site__tool" id="analyzer" aria-busy={loading}>
              <div className="site__tool-head"><div><p className="site__eyebrow">Public analyzer</p><h2>Paste text to inspect</h2></div><span className="site__tool-live"><i />Ready</span></div>
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
              <p className="site__privacy-note">Submitted text is analyzed in memory and never stored. Usage counts enforce the daily free limit.</p>
            </div>
          </div>
            {result && (
              <div className="card site__result" data-testid="site-result">
                <div className="card__head">
                  <div><p className="site__eyebrow">Result</p><h2 className="site__result-title">{result.overall_result === "signal_detected" ? "Possible hidden provenance signal" : "No supported signal detected"}</h2></div>
                  <StatusBadge status={
                    result.overall_result === "signal_detected" ? "detected"
                      : result.overall_result === "no_supported_signal_detected" ? "not_detected"
                        : "inconclusive"
                  } />
                </div>
                <p className="site__verdict">{result.verdict}</p>
                <div className={`result-strength result-strength--${result.overall_result === "signal_detected" ? "strong" : "none"}`}>
                  <span>Evidence level</span><strong>{result.overall_result === "signal_detected" ? "Moderate · artifact present" : "None · supported scan clean"}</strong>
                </div>
                <div className="site__result-grid">
                  <section><p className="site__result-label">Detected</p><strong>{result.signals_detected.length ? `${result.signals_detected.length} finding${result.signals_detected.length === 1 ? "" : "s"}` : "No findings"}</strong><span>{result.signals_detected.length ? "Review the evidence below." : "No supported Unicode artifact found."}</span></section>
                  <section><p className="site__result-label">Checked</p><strong>{result.signals_checked.length} detector</strong><span>Unicode artifact inspection</span></section>
                  <section><p className="site__result-label">Not run</p><strong>{result.unavailable_detectors.length} workflows</strong><span>Key/configuration required</span></section>
                </div>
                <details className="site__technical-details">
                  <summary>View technical details</summary>
                  <div className="site__detail-columns">
                  <section><h3>Signals checked</h3><ul className="home-list">
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
                  </ul></section>
                  <section><h3>Not run on pasted text</h3><ul className="home-list" data-testid="site-unavailable-detectors">
                  {result.unavailable_detectors.map((d) => (
                    <li key={d.detector}>
                      <span className="home-list__id">{d.display_name}</span>
                      <span className="home-list__meta">
                        {d.status === "not_applicable" ? "benchmark-only" : "configuration required"}
                        {" · "}{d.reason}
                      </span>
                    </li>
                  ))}
                  </ul></section></div>
                  <p className="site__disclaimer">{result.disclaimer}</p>
                </details>
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
        </section>

        <section className="site__section site__problem" id="detection">
          <div className="site__section-intro"><p className="site__eyebrow">Why inspect provenance?</p><h2>Text can carry signals your eyes never see.</h2><p>Invisible formatting can survive copying, generation, and review. Provenance Engine turns those machine-readable traces into a clear, bounded report.</p></div>
          <div className="site__problem-grid">
            <article className="site__problem-card"><span className="site__icon-badge">Aa</span><h3>Looks ordinary</h3><p>A sentence can read normally while containing a hidden format mark.</p><div className="site__text-sample">hello<span className="site__sample-hidden">·</span>world</div></article>
            <article className="site__problem-card"><span className="site__icon-badge">⌁</span><h3>Signals stay hidden</h3><p>Zero-width characters, bidi controls, and unusual whitespace can be difficult to spot manually.</p><div className="site__signal-pills"><span>zero-width</span><span>bidi</span><span>spacing</span></div></article>
            <article className="site__problem-card site__problem-card--accent"><span className="site__icon-badge">✓</span><h3>Review evidence</h3><p>Inspect what was found, what was checked, and what was outside the scan.</p><div className="site__evidence-lines"><i /><i /><i /></div></article>
          </div>
        </section>

        <section className="site__section site__section--flow" id="how">
          <div className="site__section-intro"><p className="site__eyebrow">A simple inspection flow</p><h2>Three steps. No authorship guessing.</h2></div>
          <ol className="site__steps">
            <li><span>01</span><div><h3>Paste text</h3><p>Drop in text directly from the source you are reviewing.</p></div></li>
            <li><span>02</span><div><h3>Analyze supported signals</h3><p>Run the deterministic Unicode check available for arbitrary input.</p></div></li>
            <li><span>03</span><div><h3>Review evidence</h3><p>See findings, clean checks, and workflows that require known configuration.</p></div></li>
          </ol>
        </section>

        <section className="site__section site__capabilities">
          <div className="site__section-intro"><p className="site__eyebrow">Detector coverage</p><h2>Know exactly what is being tested.</h2><p>Public inspection stays deliberately narrow. Advanced workflows remain explicit about their matching configuration requirements.</p><p className="site__capability-note">The public analyzer does not run statistical KGW or SynthID verification without matching configuration.</p></div>
          <div className="site__capability-grid">
            <article className="site__capability-card site__capability-card--public"><div className="site__capability-top"><span className="site__icon-badge">⌘</span><span className="site__availability">Public analyzer</span></div><h3>Unicode signals</h3><p>Invisible, machine-readable formatting artifacts in arbitrary pasted text.</p><ul><li>Zero-width characters</li><li>Bidirectional controls</li><li>Unusual whitespace</li><li>Hidden format controls</li></ul></article>
            <article className="site__capability-card"><div className="site__capability-top"><span className="site__icon-badge">K</span><span className="site__availability site__availability--muted">Advanced workflow</span></div><h3>KGW</h3><p>A statistical watermark family for a matching known configuration.</p><div className="site__card-note">Requires matching key, parameters, variant, and tokenizer.</div></article>
            <article className="site__capability-card"><div className="site__capability-top"><span className="site__icon-badge">S</span><span className="site__availability site__availability--muted">Research workflow</span></div><h3>SynthID</h3><p>A statistical watermark family for controlled reference verification.</p><div className="site__card-note">Requires matching generation-time configuration; never presented as universal detection.</div></article>
          </div>
        </section>

        <section className="site__section site__pro-showcase">
          <div className="site__pro-copy"><p className="site__eyebrow">Pro workspace</p><h2>Move from a quick scan to controlled research.</h2><p>Pro brings together configured detector workflows, reproducible experiments, and technical reporting without changing what a result can truthfully claim.</p><a className="btn btn--primary" href="#/pricing">Explore Pro capabilities</a></div>
          <div className="site__mock-dashboard" aria-label="Illustration of advanced workspace capabilities"><div className="site__mock-top"><span>Research workspace</span><b>Active</b></div><div className="site__mock-stats"><span><b>12</b>runs</span><span><b>94%</b>coverage</span><span><b>±2.1</b>CI</span></div><div className="site__mock-chart"><i /><i /><i /><i /><i /><i /></div><div className="site__mock-tags"><span>Robustness</span><span>Benchmarks</span><span>API</span></div></div>
        </section>

        <section className="site__section site__uses">
          <div className="site__section-intro"><p className="site__eyebrow">Made for careful review</p><h2>Useful wherever text needs inspection.</h2></div>
          <div className="site__use-grid">
            <article><span>✦</span><h3>Editorial review</h3><p>Spot invisible artifacts before publication.</p></article><article><span>⌁</span><h3>AI research</h3><p>Inspect controlled provenance experiments.</p></article><article><span>◫</span><h3>Model evaluation</h3><p>Compare detector behavior on known setups.</p></article><article><span>↗</span><h3>Platform operations</h3><p>Triage evidence without authorship claims.</p></article><article><span>▤</span><h3>Document inspection</h3><p>Review unusual text formatting signals.</p></article><article><span>⌘</span><h3>Developer testing</h3><p>Use configured workflows and API access.</p></article>
          </div>
        </section>

        <section className="site__section site__privacy-band">
          <span className="site__icon-badge">⌑</span><div><p className="site__eyebrow">Privacy by design</p><h2>Your pasted text is not saved as public analysis content.</h2><p>Anonymous use is tied to a random cookie for quota behavior. The product stores only the account and aggregate usage metadata needed to operate the service.</p></div><a href="#/privacy" className="site__text-link">Privacy details →</a>
        </section>

        <section className="site__section site__pricing" id="pricing">
          <div className="site__section-intro site__section-intro--center"><p className="site__eyebrow">Plans</p><h2>Start with a clear signal. Scale when the work demands it.</h2></div>
          <div className="site__pricing-grid">
            <div className="site__pricing-card">
              <span className="plan-badge">Free</span><h3>Everyday inspection</h3><p className="site__price">2 anonymous analyses / day</p>
              <ul className="site__list">
                <li>Public Unicode provenance scan with explicit unavailable checks</li>
                <li>Signed-in free accounts get higher daily limits plus history</li>
                <li>Basic workspace and usage overview</li>
              </ul>
              <button className="btn" onClick={() => session.status === "user" ? onEnterWorkspace() : setAuthOpen(true)}>{session.status === "user" ? "Open workspace" : "Create free account"}</button>
            </div>
            <div className="site__pricing-card site__pricing-card--pro">
              <span className="plan-badge plan-badge--pro">Pro</span><h3>Advanced provenance work</h3><p className="site__price">Higher limits + research workspace</p>
              <ul className="site__list">
                <li>Full detailed reports, configured detector workflows, and analysis history</li>
                <li>Robustness benchmarks, detector management, and API access</li>
              </ul>
              {session.me?.plan === "pro" || billing?.plan === "pro" ? (
                <>
                  <p data-testid="site-pro-state">Pro is active{billing?.subscription?.cancel_at_period_end ? " until the end of the current paid period." : "."}</p>
                  <button className="btn" data-testid="site-cancel-subscription" disabled={billingBusy} onClick={async () => {
                      if (!session.token) return;
                      setBillingBusy(true); setBillingMessage("");
                      try { await client.cancelBillingSubscription(session.token); setBillingMessage("Cancellation is scheduled for the end of the current billing cycle."); refresh(); }
                      catch (e) { setBillingMessage(e instanceof ApiError ? e.userMessage : "Could not schedule cancellation."); }
                      finally { setBillingBusy(false); }
                    }}>Cancel subscription</button>
                </>
              ) : (
                <button className="btn btn--primary" data-testid="site-upgrade-pro" disabled={billingBusy || (session.status === "user" && billingStatus?.configured === false)} onClick={upgrade}>
                  {billingBusy ? "Opening checkout…" : "Upgrade to Pro"}
                </button>
              )}
              {billingStatus?.configured === false && <p className="pricing-card__notice" data-testid="site-billing-unavailable">Pro coming soon — payments are temporarily unavailable for this deployment.</p>}
              {billing?.webhook_processing && <p data-testid="site-payment-pending">Payment received; activating Pro access…</p>}
              {billingMessage && <div className="alert alert--error">{billingMessage}</div>}
            </div>
          </div>
        </section>

        <section className="site__section site__faq" id="faq">
          <div className="site__section-intro site__section-intro--center"><p className="site__eyebrow">FAQ</p><h2>Clear answers, clear boundaries.</h2></div>
          <details><summary>Can this prove who wrote a text?<span>+</span></summary><p>No. It reports detector-specific signals. A clean public Unicode result does not imply human authorship or prove that every watermark is absent.</p></details>
          <details><summary>Does it remove watermarks?</summary><p>No. This product performs detection and provenance analysis only. It cannot modify text or remove watermarks.</p></details>
          <details><summary>Do I need an account?</summary><p>No. Anonymous visitors get 2 free analyses per day. Signing in raises your limits and unlocks history.</p></details>
          <details><summary>Is my text stored?</summary><p>No. Text is analyzed in memory and discarded. Only anonymous aggregate counts are kept for quota enforcement.</p></details>
          <details><summary>Does the public check test KGW or SynthID?</summary><p>No. Those statistical detectors need the matching secret key and generation configuration. Authenticated workflows can verify a known configuration where access and deployment setup allow it.</p></details>
          <details><summary>Which models are attributed?</summary><p>None automatically. Configured reference detectors test compatibility with a specific setup; results never claim universal or official vendor attribution.</p></details>
        </section>
      </main>

      <footer className="site__footer">
        <div className="site__footer-brand"><a className="site__logo" href="#/"><span className="site__logo-mark">P</span>Provenance Engine</a><p>Evidence-led text provenance analysis for careful inspection.</p><small>© {new Date().getFullYear()} Provenance Engine</small></div>
        <div className="site__footer-links">
          <div><h2>Product</h2><a href="#detection">What we check</a><a href="#/methodology">Methodology</a><a href="#/pricing">Pricing</a></div>
          <div><h2>Trust</h2><a href="#/privacy">Privacy</a><a href="#/terms">Terms</a><button className="link-btn" onClick={onDeveloperSignIn}>Developer sign-in</button></div>
        </div>
      </footer>

      {authOpen && (
        <AuthModal
          initialError={authCallbackMessage}
          onClose={() => { setAuthOpen(false); setAuthCallbackMessage(""); }}
          onSignedIn={async () => {
            const synced = await refresh();
            if (!synced) return false;
            setAuthOpen(false);
            setAuthCallbackMessage("");
            if (!window.location.hash.includes("upgrade=pro")) onEnterWorkspace();
            return true;
          }}
        />
      )}
    </div>
  );
}

export function AuthModal({ onClose, onSignedIn, initialError = "" }: {
  onClose: () => void;
  onSignedIn: () => Promise<boolean>;
  initialError?: string;
}) {
  const [mode, setMode] = useState<"password" | "magic">("password");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState(initialError);
  const [busy, setBusy] = useState(false);
  const emailRef = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    emailRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>("button:not([disabled]), input:not([disabled]), [href]");
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  useEffect(() => {
    if (initialError) setError(initialError);
  }, [initialError]);

  if (!supabaseConfigured || !supabase) {
    return (
      <div className="modal" data-testid="site-auth" role="dialog" aria-modal="true" aria-labelledby="auth-unavailable-title">
        <div className="card modal__card" ref={dialogRef}>
          <h3 id="auth-unavailable-title" className="card__title">Sign in is not configured</h3>
          <p className="settings__hint">This deployment has no authentication provider set up.</p>
          {initialError && <div className="alert alert--error" data-testid="site-auth-error">{initialError}</div>}
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  const completeAuthenticatedSession = async () => {
    const completed = await onSignedIn();
    if (!completed) {
      setError("Your session was created, but your workspace could not be set up. Please try again.");
    }
  };

  const runPassword = async (action: "sign_in" | "sign_up") => {
    setError("");
    setMessage("");
    setBusy(true);
    try {
      rememberAuthReturnTo();
      if (action === "sign_in") {
        const { error: err } = await supabase!.auth.signInWithPassword({ email, password });
        if (err) setError("Could not sign in with that email and password.");
        else await completeAuthenticatedSession();
      } else {
        const { data, error: err } = await supabase!.auth.signUp({ email, password });
        if (err) {
          setError("Could not create the account. Check the email and password requirements, then try again.");
        } else if (data.session) {
          await completeAuthenticatedSession();
        } else {
          setMessage("Account created. Verify your email to continue; this page will open your workspace after you sign in.");
        }
      }
    } finally {
      setBusy(false);
    }
  };

  const sendMagicLink = async () => {
    setError(""); setMessage(""); setBusy(true);
    try {
      rememberAuthReturnTo();
      const { error: err } = await supabase!.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: authRedirectUrl() },
      });
      if (err) setError("Could not send a sign-in link. Please try again.");
      else setMessage("Check your email for a sign-in link.");
    } finally { setBusy(false); }
  };

  const signInWithGoogle = async () => {
    setError(""); setMessage(""); setBusy(true);
    try {
      rememberAuthReturnTo();
      const { data, error: err } = await supabase!.auth.signInWithOAuth({
        provider: "google",
        options: { redirectTo: authRedirectUrl() },
      });
      if (err || !data.url) {
        setError("Google sign-in is unavailable for this deployment. Please use email sign-in or try again later.");
      } else {
        setMessage("Redirecting to Google…");
      }
    } finally { setBusy(false); }
  };

  return (
    <div className="modal" data-testid="site-auth" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <div className="card modal__card" ref={dialogRef}>
        <div className="card__head">
          <h3 id="auth-title" className="card__title">Sign in</h3>
          <button className="close-btn" onClick={onClose} aria-label="Close">×</button>
        </div>
        <div className="site__tabs">
          <button className={`btn btn--sm${mode === "password" ? " btn--primary" : ""}`} onClick={() => setMode("password")}>Password</button>
          <button className={`btn btn--sm${mode === "magic" ? " btn--primary" : ""}`} onClick={() => setMode("magic")}>Email link</button>
        </div>
        <label className="label" htmlFor="auth-email">Email</label>
        <input
          id="auth-email" data-testid="site-auth-email" className="input" type="email"
          value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" ref={emailRef}
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
                onClick={() => void runPassword("sign_in")}
              >
                Sign in
              </button>
              <button
                className="btn" disabled={busy || !email || !password}
                onClick={() => void runPassword("sign_up")}
              >
                Create account
              </button>
            </>
          ) : (
            <button
              className="btn btn--primary" data-testid="site-auth-submit" disabled={busy || !email}
              onClick={() => void sendMagicLink()}
            >
              Send magic link
            </button>
          )}
        </div>
        <div style={{ marginTop: "12px" }}>
          <button
            className="btn" data-testid="site-auth-google" disabled={busy}
            onClick={() => void signInWithGoogle()}
          >
            Continue with Google
          </button>
        </div>
      </div>
    </div>
  );
}
