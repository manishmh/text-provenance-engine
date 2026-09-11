export function MethodologyPage() {
  return (
    <main className="site-page methodology" data-testid="methodology-page">
      <section className="site-page__intro">
        <p className="site__eyebrow">Methodology</p>
        <h1>Evidence first, with the limits visible.</h1>
        <p>Text Provenance Engine reports detector-specific evidence. It is not a general-purpose AI authorship classifier.</p>
      </section>
      <section className="methodology__grid">
        <article className="card">
          <h2>Public arbitrary-text check</h2>
          <p>The public analyzer scans pasted text for invisible Unicode controls, zero-width characters, bidirectional controls, unusual whitespace, and related formatting artifacts. This deterministic check is inexpensive and needs no hidden watermark key.</p>
        </article>
        <article className="card">
          <h2>Configured statistical workflows</h2>
          <p>KGW and SynthID-style reference verification depends on the matching watermark configuration, tokenizer, and—in some cases—secret material. They are therefore not presented as universal checks on public input.</p>
        </article>
        <article className="card">
          <h2>Benchmarks and robustness</h2>
          <p>Controlled benchmark and transformation workflows are advanced research tools. They help compare detector behaviour under known conditions; they do not establish attribution for arbitrary third-party text.</p>
        </article>
      </section>
      <section className="site-page__note">
        <h2>Reading a result</h2>
        <ul className="site__list">
          <li><strong>Signal detected</strong> means the supported check found an artifact worth reviewing.</li>
          <li><strong>No supported signal detected</strong> means the checks actually run found none; it is not a guarantee that no watermark exists.</li>
          <li><strong>Unavailable</strong> means a detector requires matching configuration or belongs to a controlled benchmark path.</li>
        </ul>
      </section>
      <section className="site-page__note">
        <h2>Privacy and data handling</h2>
        <p>Public text is analyzed in memory and is not retained as public analysis content. The service stores the minimal usage and account metadata needed for quota, workspace, and subscription behaviour. No detector keys or raw debug values are displayed in the public product.</p>
      </section>
    </main>
  );
}
