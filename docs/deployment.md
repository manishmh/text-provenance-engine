# Production deployment guide

## Topology

Deploy the Vite dashboard as a static site and the FastAPI application as a
separate HTTPS service. The backend connects to Supabase PostgreSQL through
the session-mode pooler and verifies Supabase Auth JWTs locally. Browser code
talks only to the API and Supabase Auth; it never accesses application tables
through the Supabase Data API. Razorpay is an optional backend integration;
the product remains fully usable when its credentials are absent.

Persist only the PostgreSQL database and, when advanced workspaces are enabled,
the `PROVENANCE_ROBUSTNESS_DIR` tree. That directory holds benchmark manifests,
aggregate reports, and generated research artifacts. It must be mounted on
durable storage or object-backed storage before enabling benchmark execution.
Model caches are optional, disposable performance caches; do not treat them as
durable data.

For a single-domain Vercel deployment, use [vercel.md](vercel.md). Its FastAPI
Function is deliberately limited to synchronous, persistent-PostgreSQL work;
the local `ThreadPoolExecutor` and filesystem-backed benchmark artifacts are
not a durable serverless worker.

## Environment matrix

| Setting | Development | Staging | Production | Classification |
|---|---|---|---|---|
| `PROVENANCE_ENVIRONMENT` | `development` | `staging` | `production` | public configuration |
| `DATABASE_URL` | SQLite or local Postgres | Supabase/staging pooler | Supabase session pooler | backend secret |
| `CORS_ORIGINS` | `http://localhost:5173` | staging static origin | explicit HTTPS static origin | public configuration |
| `PROVENANCE_TRUST_PROXY` | `0` | `1` only behind trusted proxy | `1` only behind trusted proxy | optional configuration |
| `PROVENANCE_ANON_COOKIE_SECRET` | generated/ephemeral permitted | generated | generated, required | generated secret |
| cookie secure/domain/samesite | `0` / host-only / `lax` | `1` / domain as needed / `lax` or `none` | `1` / domain as needed / `lax` or `none` | configuration |
| public/free/pro quota variables | local defaults | intended limits | intended limits | configuration |
| `PROVENANCE_MAX_REQUEST_BODY_BYTES` | `1048576` | intended limit | intended limit | configuration |
| `PROVENANCE_ROBUSTNESS_DIR` | local directory | durable mount | durable mount | backend path |
| `PROVENANCE_MODEL_CACHE_SIZE` | `0`–`2` | capacity-based | capacity-based | optional configuration |
| `SUPABASE_URL` | project URL | staging project URL | production project URL | public identifier (backend copy) |
| `SUPABASE_JWT_SECRET` | legacy project secret only | legacy project secret only | legacy project secret only | backend secret, optional with JWKS |
| `VITE_API_BASE_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY` | local values | staging URLs/anon key | production URLs/anon key | browser-public |
| Razorpay secrets and webhook secret | empty or test values | provider test values | live values after activation | provider credentials |

`VITE_*` is the complete frontend configuration surface. Never place
`DATABASE_URL`, `SUPABASE_JWT_SECRET`, API keys, Razorpay
secrets, or webhook secrets in a Vite environment file.

The remaining backend-only values are intentionally grouped here so the matrix
does not hide a second configuration channel: `PROVENANCE_API_KEY` and
`PROVENANCE_ADMIN_API_KEY` are optional generated secrets for legacy/developer
API access; `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`,
`MAX_LIST_LIMIT`, `PROVENANCE_DAILY_REQUEST_LIMIT`, and
`PROVENANCE_DAILY_CHARACTER_LIMIT` are optional service limits;
`PROVENANCE_MAX_BACKGROUND_JOBS`, `PROVENANCE_JOB_RETENTION_HOURS`, and
`PROVENANCE_MAX_BENCHMARK_RUNS` are runtime limits. Billing is optional and
Razorpay reads
`RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`,
`RAZORPAY_PRO_PLAN_ID`, and `RAZORPAY_PRO_TOTAL_COUNT`. All provider secrets
and plan identifiers are backend-only. `SUPABASE_ANON_KEY` was removed from
the backend template because the backend does not read it; the browser-safe
value belongs only in `dashboard/.env` as `VITE_SUPABASE_ANON_KEY`.

## Backend deployment

```bash
cp .env.example .env
# Populate production values in your secret manager or deployment platform.
docker build -t text-provenance-api:2.0.0 .
docker run --rm -p 8000:8000 --env-file .env text-provenance-api:2.0.0
curl -fsS https://api.example.com/health
curl -fsS https://api.example.com/ready
```

Use `PROVENANCE_ENVIRONMENT=production`. Startup rejects wildcard CORS,
non-HTTPS production origins, insecure/short anonymous-cookie configuration,
and a non-PostgreSQL production `DATABASE_URL`. Set `PROVENANCE_TRUST_PROXY=1`
only when the API is behind a proxy that replaces, rather than forwards,
`X-Forwarded-For` from clients.

For a separate frontend `https://app.example.com` and API
`https://api.example.com`, set:

```dotenv
CORS_ORIGINS=https://app.example.com
PROVENANCE_ANON_COOKIE_SECURE=1
PROVENANCE_ANON_COOKIE_DOMAIN=api.example.com
PROVENANCE_COOKIE_SAMESITE=none
```

Use `SameSite=lax` and host-only cookies when frontend and API share an origin.
`SameSite=none` requires Secure cookies. Add the frontend URL and its auth
callback (`https://app.example.com/`) to Supabase Auth's Site URL and Redirect
URLs. Add Google OAuth redirect URLs in Supabase, not in this application.

## Database bootstrap, backup, and restore

Repository startup is the migration command. It applies the shared SQLite/PG
DDL plus additive `ALTER ... IF NOT EXISTS` migrations in one PostgreSQL
transaction. It never drops tables or data.

```bash
dotenv -f .env run -- python -c \
  "from provenance.api.db import create_repository; r=create_repository(); r.close()"
dotenv -f .env run -- sh -c 'psql "$DATABASE_URL" -c "\\dt"'
dotenv -f .env run -- sh -c 'psql "$DATABASE_URL" -c "select count(*) from app_users"'
```

Use the Supabase **session pooler** URI. The current deployment has RLS enabled
on all ten application tables and zero permissive `public`, `anon`, or
`authenticated` policies. FastAPI connects as database owner `postgres`; table
owners bypass RLS unless `FORCE ROW LEVEL SECURITY` is enabled. The frontend has
no database credentials and no direct table access. Do not add permissive RLS
policies; if changing to a non-owner backend role, define restrictive backend
policies and grants first.

Before any non-additive future migration, create a Supabase backup or:

```bash
dotenv -f .env run -- sh -c 'pg_dump --format=custom --file=provenance.backup "$DATABASE_URL"'
dotenv -f .env run -- sh -c 'pg_restore --clean --if-exists --dbname="$DATABASE_URL" provenance.backup'
```

Test restores in a non-production project. Application rollback does not undo
additive schema changes.

## Frontend deployment and smoke test

```bash
cd dashboard
cp .env.example .env
npm ci
npm run build
# Deploy dashboard/dist through a static HTTPS host.
```

After deployment, verify signed-out public analysis, quota exhaustion, Supabase
sign-in, Free workspace, a Pro-gated route, and `/v1/billing/status`. Billing
being `configured:false` is healthy: checkout must be unavailable while public
analysis, auth, Free workspace, `/health`, and `/ready` remain available.

When Razorpay is configured later, set its backend credentials and restart the
API. Configure its webhook at
`https://api.example.com/v1/billing/webhook/razorpay`; no frontend code change
is required.

## Operational semantics

- `/health`: process is alive; it does not test dependencies.
- `/ready`: persistence and executor/runtime are usable. Optional billing does
  not affect readiness; inspect `/v1/billing/status` separately.
- Request logs include method, route, status, duration, and request ID. They do
  not include text, authorization values, API keys, database URLs, or provider
  secrets. Send `X-Request-ID` to correlate a support request.
- The application rate limiter is process-local. Run a single API replica or
  enforce an equivalent shared limit at the trusted edge before horizontally
  scaling the API.
