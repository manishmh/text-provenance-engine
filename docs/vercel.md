# Single-project Vercel deployment

This deployment uses one Vercel project and one HTTPS domain:

```text
browser ── / ───────────────> dashboard/dist (Vite static site)
browser ── /api/v1/* ───────> api/index.py (FastAPI ASGI Function)
                              └── Supabase Auth + PostgreSQL
Razorpay ─ /api/v1/billing/webhook/razorpay
```

`api/index.py` is only an ASGI prefix adapter. It calls the existing
`provenance.api.app.create_app()` factory, so the normal routes remain
`/v1/*`, `/health`, and `/ready` inside FastAPI and are publicly reached as
`/api/v1/*`, `/api/health`, and `/api/ready`.

## Vercel project settings

Import this repository with the repository root as the Vercel project root.
`vercel.json` supplies the required settings:

| Setting | Value |
| --- | --- |
| Framework preset | Vite |
| Install command | `npm --prefix dashboard ci` |
| Build command | `npm --prefix dashboard run build` |
| Output directory | `dashboard/dist` |
| Python function | `api/index.py` |

Vercel detects the ASGI object named `app` in `api/index.py` and installs the
lightweight production Python dependencies from root `pyproject.toml`.
Torch/Transformers remain in the optional `hf` extra and are not part of this
Function bundle.

## Browser variables

Set these as Vercel **Production**, **Preview**, and/or **Development** values
as appropriate. They are intentionally browser-public.

```dotenv
VITE_SUPABASE_URL=https://<project-ref>.supabase.co
VITE_SUPABASE_ANON_KEY=<Supabase publishable/anon key>
# Leave empty in production: the frontend defaults to same-origin /api.
VITE_API_BASE_URL=
```

For `vercel dev`, either leave `VITE_API_BASE_URL` empty to exercise `/api`, or
set it to `http://localhost:8000` when running Uvicorn separately.

## Backend and operational variables

Enter these in Vercel only; never prefix them with `VITE_`.

| Variable | Classification | Vercel production value |
| --- | --- | --- |
| `PROVENANCE_ENVIRONMENT` | operational | `production` |
| `PROVENANCE_RUNTIME` | optional operational override | leave unset; Vercel supplies `VERCEL=1` automatically |
| `DATABASE_URL` | secret | Supabase **session pooler** PostgreSQL URI |
| `SUPABASE_URL` | backend configuration | `https://<project-ref>.supabase.co` |
| `SUPABASE_JWT_SECRET` | optional backend secret | only for legacy HS256 projects; omit for modern ES256/RS256 JWKS projects |
| `PROVENANCE_ANON_COOKIE_SECRET` | generated secret | unique random value, at least 32 characters |
| `PROVENANCE_ANON_COOKIE_SECURE` | operational | `1` |
| `PROVENANCE_ANON_COOKIE_DOMAIN` | operational | empty (host-only cookie) |
| `PROVENANCE_COOKIE_SAMESITE` | operational | `lax` |
| `CORS_ORIGINS` | operational | `https://<production-domain>` |
| `PROVENANCE_TRUST_PROXY` | operational | `1` |
| `PROVENANCE_PUBLIC_DAILY_LIMIT`, `PROVENANCE_PUBLIC_MAX_CHARS`, `PROVENANCE_FREE_DAILY_LIMIT`, `PROVENANCE_FREE_MAX_CHARS`, `PROVENANCE_PRO_DAILY_LIMIT`, `PROVENANCE_PRO_MAX_CHARS` | operational | intended product limits |
| `PROVENANCE_MAX_REQUEST_BODY_BYTES`, `RATE_LIMIT_MAX_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `MAX_LIST_LIMIT`, `PROVENANCE_DAILY_REQUEST_LIMIT`, `PROVENANCE_DAILY_CHARACTER_LIMIT` | operational | intended edge/request limits; the in-process rate limiter is best-effort only |
| `PROVENANCE_MAX_BACKGROUND_JOBS`, `PROVENANCE_JOB_RETENTION_HOURS`, `PROVENANCE_MODEL_CACHE_SIZE`, `PROVENANCE_MAX_BENCHMARK_RUNS`, `PROVENANCE_ROBUSTNESS_DIR` | operational | leave defaults; worker/cache/artifact settings do not make durable Pro execution available on Vercel |
| `PROVENANCE_API_KEY`, `PROVENANCE_ADMIN_API_KEY` | optional backend secrets | only if legacy developer/admin API access is enabled |
| `RAZORPAY_KEY_ID`, `RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`, `RAZORPAY_PRO_PLAN_ID`, `RAZORPAY_PRO_TOTAL_COUNT` | optional provider configuration | leave unset until KYC/test credentials exist |

The application will start and serve the free product when Razorpay variables
are absent. `GET /api/v1/billing/status` then returns `configured:false`.

## Supabase and Razorpay URLs

Set Supabase **Authentication → URL Configuration** Site URL to
`https://<production-domain>/` and add it to Redirect URLs. For Google, use
the Supabase-generated provider callback URL in Google Cloud, then add the
production app URL to Supabase's redirect allow-list. No Google secret belongs
in Vercel browser variables.

After Razorpay KYC, configure the signed webhook endpoint as:

```text
https://<production-domain>/api/v1/billing/webhook/razorpay
```

## Serverless capability boundary

Vercel supports the synchronous, lightweight product surface: public Unicode
analysis, quota cookie issuance/consumption, Supabase JWT verification and
account sync, Free workspace/history stored in PostgreSQL, Razorpay checkout
initiation, signed webhook processing, and `/api/health` / `/api/ready`.

It does **not** make the current local worker durable. On Vercel the API
returns HTTP 503 for async analysis jobs, benchmark execution/listing/retry,
and filesystem-backed robustness reports. It also rejects configured/model
detector paths; only synchronous Unicode analysis remains available through
the authenticated `/v1/analyze` route. `ready.executor` is
`not_applicable`, while overall readiness remains healthy when PostgreSQL is
available.

The reasons are intentional: `ThreadPoolExecutor`, process-local rate-limit
memory/model caches, and `PROVENANCE_ROBUSTNESS_DIR` do not survive Function
cold starts or scale-out. Existing local read-only fixtures can be packaged,
but generated benchmark manifests and artifacts must move to durable object
storage/Supabase Storage plus a separate worker before those Pro workflows can
be enabled on Vercel. Do not treat the Function filesystem as persistent.

## Verification

```bash
npx vercel build
npx vercel dev

curl -i http://localhost:3000/api/health
curl -i http://localhost:3000/api/ready
curl -c cookies.txt -H 'content-type: application/json' \
  --data '{"text":"normal text"}' \
  http://localhost:3000/api/v1/public/analyze
python -c 'import json; print(json.dumps({"text": "hello\u200bworld"}))' \
  | curl -i -b cookies.txt -c cookies.txt -H 'content-type: application/json' \
      --data-binary @- http://localhost:3000/api/v1/public/analyze
```

The second request must detect the actual U+200B character. Check
`/api/v1/billing/status` with no Razorpay credentials and confirm it is safe
and unconfigured rather than an API failure.
