import { useEffect, useState } from "react";
import type { SaaSSession } from "../hooks/useSession";
import type { BillingStatusResponse } from "../types/api";

interface Props {
  session: SaaSSession;
  onOpenWorkspace: () => void;
}

/** A deliberately small product page: checkout itself remains in the shared
 * landing flow, where the provider-neutral billing response is handled. */
export function PricingPage({ session, onOpenWorkspace }: Props) {
  const [billing, setBilling] = useState<BillingStatusResponse | null>(null);

  useEffect(() => {
    session.client.billingStatus().then(setBilling).catch(() =>
      setBilling({ provider: "razorpay", configured: false, checkout_mode: null }),
    );
  }, [session.client]);

  const proActive = session.me?.plan === "pro";
  const cta = proActive ? "Open workspace" : billing?.configured ? "Choose Pro" : "Payments temporarily unavailable";
  const href = proActive ? "#/app" : billing?.configured ? "#/?upgrade=pro" : "#/";

  return (
    <main className="site-page" data-testid="pricing-page">
      <section className="site-page__intro">
        <p className="site__eyebrow">Plans</p>
        <h1>Start with a clear signal. Scale into a research workspace.</h1>
        <p>Every plan keeps the public analyzer honest about what it can and cannot test.</p>
      </section>
      <section className="pricing-grid" aria-label="Plan comparison">
        <article className="card pricing-card">
          <span className="plan-badge">Free</span>
          <h2>Everyday inspection</h2>
          <p className="pricing-card__lead">A practical starting point for hidden Unicode inspection.</p>
          <ul className="site__list">
            <li>2 anonymous analyses per day</li>
            <li>Authenticated free quota at the configured limit</li>
            <li>Basic workspace, history, and usage</li>
            <li>Unicode artifact results with clear limitations</li>
          </ul>
          <a className="btn" href={session.status === "user" ? "#/app" : "#/?auth=signin"}>
            {session.status === "user" ? "Open workspace" : "Create free account"}
          </a>
        </article>
        <article className="card pricing-card pricing-card--featured">
          <span className="plan-badge plan-badge--pro">Pro</span>
          <h2>Advanced provenance work</h2>
          <p className="pricing-card__lead">Higher limits and the full technical workspace for configured, reproducible workflows.</p>
          <ul className="site__list">
            <li>Higher configured daily and character limits</li>
            <li>Full reports and advanced dashboard access</li>
            <li>Robustness experiments and benchmark runs</li>
            <li>Configured detector workflows and API access where enabled</li>
          </ul>
          <a className={`btn btn--primary${!billing?.configured && !proActive ? " btn--disabled" : ""}`} href={href} aria-disabled={!billing?.configured && !proActive} onClick={(event) => {
            if (!billing?.configured && !proActive) event.preventDefault();
            if (proActive) onOpenWorkspace();
          }}>
            {cta}
          </a>
          {!billing?.configured && !proActive && <p className="pricing-card__notice" data-testid="pricing-billing-unavailable">Payments are not configured for this deployment. Free analysis remains available.</p>}
        </article>
      </section>
      <section className="site-page__note">
        <h2>No hidden pricing claims</h2>
        <p>Pro unlocks access according to your configured entitlement. It does not turn an arbitrary pasted passage into a universal AI-authorship or vendor-watermark determination.</p>
      </section>
    </main>
  );
}
