export function PrivacyPage() {
  return (
    <main className="site-page legal-page" data-testid="privacy-page">
      <section className="site-page__intro">
        <p className="site__eyebrow">Privacy</p>
        <h1>What the product stores.</h1>
        <p>This page describes the current implementation. It is not legal advice or a substitute for a jurisdiction-specific privacy notice.</p>
      </section>
      <section className="site-page__note"><h2>Public analysis</h2><p>Text submitted to the public analyzer is processed in memory and is not stored as public analysis content. The service records quota-related counts and opaque visitor/account identifiers needed to operate the limit.</p></section>
      <section className="site-page__note"><h2>Accounts and billing</h2><p>Supabase provides authentication. Razorpay provides optional subscription billing when configured. The application stores a verified account identifier, email when supplied by the verified session, plan status, usage metadata, and subscription/provider identifiers. It does not store card details, CVVs, or raw webhook payloads.</p></section>
      <section className="site-page__note"><h2>Retention</h2><p>There is currently no automatic deletion schedule for account, subscription, usage, or benchmark records. Advanced benchmark and robustness artifacts may persist in the configured artifact directory. Completed job records follow the configured job retention setting. See the deployment documentation for the complete implementation inventory.</p></section>
    </main>
  );
}

export function TermsPage() {
  return (
    <main className="site-page legal-page" data-testid="terms-page">
      <section className="site-page__intro">
        <p className="site__eyebrow">Terms</p>
        <h1>Use the evidence responsibly.</h1>
        <p>This lightweight product notice describes the boundaries of the current service; obtain legal review before a public commercial launch.</p>
      </section>
      <section className="site-page__note"><h2>What the service does</h2><p>The public tool checks arbitrary text for supported Unicode artifacts. Configured watermark and benchmark workflows are not universal AI detection, vendor attribution, or proof of authorship.</p></section>
      <section className="site-page__note"><h2>Acceptable use</h2><p>Do not use the service to make unsupported claims about people, authorship, model providers, or the absence of every watermark. Do not submit content you lack permission to process or attempt to bypass quota, authentication, entitlement, or payment controls.</p></section>
      <section className="site-page__note"><h2>Service availability</h2><p>Availability, limits, and optional billing are configuration-dependent. A checkout redirect or browser callback is not proof of a subscription; verified provider webhook state controls Pro access.</p></section>
    </main>
  );
}
