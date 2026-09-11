# Privacy and retention

This is an implementation description, not legal advice or a privacy-policy
substitute for a particular jurisdiction.

| Data | Stored behavior |
|---|---|
| Public analyzer text | analyzed in memory; not stored by public routes |
| Full API analysis | no raw input text; stores text hash, counts, requested detector names, and result metadata/results |
| Anonymous identity | signed opaque `pv_visitor` cookie; database stores opaque visitor ID and HMAC-hashed coarse abuse signal, never raw IP |
| Usage | timestamps, plan/account/visitor linkage where applicable, character count, success, and endpoint-level aggregate usage |
| Supabase account | Supabase owns authentication identity; application stores auth user ID, email if supplied in verified JWT, plan label, and timestamps |
| Billing | Razorpay provider/customer/subscription/plan identifiers, status, period dates, cancellation flag, and webhook event IDs; never card data, CVV, or raw webhook payloads |
| Benchmark artifacts | server-side manifests, configured experiment metadata, aggregate results, and generated research artifacts under `PROVENANCE_ROBUSTNESS_DIR` |

There is no automatic retention deletion for account, subscription, usage, or
benchmark records in the current product. Completed API jobs are cleaned after
`PROVENANCE_JOB_RETENTION_HOURS` (24 by default); their other stored analysis
records follow normal database retention. Set an operational retention policy
before public launch, then implement deletion/export workflows only after legal
requirements are decided. Cookies expire after one year unless cleared or the
cookie signing secret changes.

Third-party services used by the product are Supabase (authentication and the
hosted PostgreSQL deployment) and Razorpay only when billing is configured.
