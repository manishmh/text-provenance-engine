import { useEffect, useState } from "react";
import type { SaaSSession } from "../hooks/useSession";
import type { BillingStatusResponse } from "../types/api";

export function AccountSettingsPage({ session, onSignOut }: { session: SaaSSession; onSignOut: () => void }) {
  const [billing, setBilling] = useState<BillingStatusResponse | null>(null);

  useEffect(() => {
    session.client.billingStatus().then(setBilling).catch(() =>
      setBilling({ provider: "razorpay", configured: false, checkout_mode: null }),
    );
  }, [session.client]);

  return (
    <div data-testid="account-settings">
      <h2 className="page-header">Settings</h2>
      <p className="page-sub">Account, plan, and product access.</p>
      <div className="card settings__card">
        <div className="card__head"><h3 className="card__title">Account</h3><span className={`plan-badge${session.me?.plan === "pro" ? " plan-badge--pro" : ""}`}>{session.me?.plan ?? "Free"}</span></div>
        <dl className="kv">
          <div><dt>Email</dt><dd>{session.me?.email ?? "Verified Supabase account"}</dd></div>
          <div><dt>Workspace access</dt><dd>{session.me?.entitlements.can_access_advanced ? "Advanced" : "Basic"}</dd></div>
          <div><dt>Billing</dt><dd>{billing?.configured ? "Razorpay available" : "Payments temporarily unavailable"}</dd></div>
        </dl>
        <button className="btn btn--danger btn--sm" style={{ marginTop: 20 }} onClick={onSignOut}>Sign out</button>
      </div>
      <div className="card settings__card">
        <h3 className="card__title">Privacy</h3>
        <p className="settings__hint" style={{ marginBottom: 0 }}>Public submitted text is analyzed in memory. Your signed-in workspace retains the records and artifacts created by the workflows you run; see the Privacy page for the current retention details.</p>
      </div>
    </div>
  );
}
