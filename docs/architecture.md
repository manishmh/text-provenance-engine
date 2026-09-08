# Architecture

## What This Project Is

Text Provenance Engine is a local, text-only provenance-signal analysis package.
Phase 1 focuses on deterministic Unicode artifacts and known-configuration
watermark evidence.

## Phase 1 Scope

Phase 1 implements:

- text validation and statistics;
- Unicode artifact reporting;
- a common detector interface;
- controlled KGW and SynthID simulations;
- a model-token KGW reference research path;
- a local KGW evaluation/benchmark subsystem (TPR/FPR across lengths and
  thresholds);
- structured result schemas;
- a local CLI and tests.

It does not implement an HTTP API, database, authentication, a hosted
benchmarking *service*, watermark removal, attack experiments, or generic AI
detection. The benchmark is a local, offline evaluation harness, not a service.

## Watermark Detection vs AI Detection

A watermark detector tests whether text contains statistical evidence matching a
specific configured signal. This is not the same as detecting whether arbitrary
text was written by AI. The engine deliberately avoids fields such as
`ai_probability`, `human_generated`, or model-provider attribution.

## KGW Assumptions

KGW detection is meaningful only when the detector configuration corresponds to
the generation configuration. The current code has two KGW paths:

- `detectors.simulated.kgw`: controlled-local-only simulation using a regex
  tokenizer over synthetic token generation. This is not real KGW compatibility.
- `detectors.reference.kgw`: `kgw-python-left-v1`, a MarkLLM-style left-hash KGW
  detector over exact token IDs.

The reference detector's classification depends on the tokenizer backend it is
given, because the KGW math is identical but the meaning of the token IDs is not:

- simple/controlled tokenizer -> `implementation_kind="reference-adapted"`,
  `compatibility="controlled-local-only"`. The algorithm is exercised over toy
  token IDs; this must not be advertised as a real model-backed run.
- Hugging Face tokenizer (real model) -> `implementation_kind="reference"`,
  `compatibility="kgw-reference"`. Only this is a genuine model/tokenizer-backed
  KGW experiment.

The reference detector reports green-token counts, z-scores, and p-values. A
negative KGW result does not imply human authorship. A successful KGW detection
demonstrates evidence consistent with the tested KGW configuration; it does not
identify the model provider or prove that text was AI-generated.

### Upstream compatibility (verified)

`kgw-python-left-v1` is a structural port of MarkLLM's KGW left-hash/additive
scheme: identical gamma, delta, greenlist size, f-scheme formula, seed formula
`(hash_key * f) % vocab`, z-score, and p-value. The one difference is the
permutation source — we use Python `random.Random(seed).shuffle` where MarkLLM
uses `torch.randperm`.

Compatibility classification for `kgw-python-left-v1`: **reference-aligned but RNG-incompatible**. The
detector is a valid, self-consistent research detector for text generated with
*this* variant, but it is not interoperable with MarkLLM, lm-watermarking, or any
external KGW deployment, and must not be called production- or byte-compatible.
Full evidence: `docs/kgw-upstream-crosscheck.md`.

To address interoperability, we provide `kgw-markllm-v1`. This variant uses `torch.randperm`
and exactly matches genuine MarkLLM 0.1.5 behavior. It requires PyTorch and is explicitly intended
for interoperability testing.
It has been verified against `markllm` via `scripts/markllm_kgw_compatibility.py` to ensure
identical greenlists, identical detection statistics (z-score, p-value), and full generation
interoperability.

## Scheme-Agnostic Evaluation and Benchmark Subsystem

`provenance.benchmark` is a scheme-agnostic evaluation framework that currently
supports KGW and SynthID-Text. It has four layers, kept deliberately separate
so the statistics are unit-testable without a model:

- `records`: a flat, JSON-serializable `EvaluationRecord` capturing one
  generated-and-scored sample -- scheme/variant, model and tokenizer identifiers
  and revisions, watermark parameters (with `hash_key_id`, never the raw key),
  generation parameters, seed, prompt id, ground-truth watermark flag, token
  counts, generic `score` (z_score for KGW, weighted-mean for SynthID),
  threshold, and `detected`. Scheme-specific fields (e.g. green_fraction for
  KGW, ngram_len for SynthID) are optional.
- `statistics`: pure aggregation -- per-condition means/stdev/min/max,
  detection rate, and `true_positive_rate`/`false_positive_rate`/
  `threshold_analysis`/`length_effect`. TPR/FPR are computed over the generic
  `score` field, making them work for any scheme.
- `calibration`: pure statistical calibration -- Wilson score confidence
  intervals and ROC/AUC. AUC is scheme-agnostic (compares watermarked vs
  unwatermarked scores).
- `report`: builds a JSON report and renders a scheme-aware text summary
  (title, config keys, limitations adapt to the scheme).
- `runner`: model-backed loops for each scheme. KGW uses
  `generate_records`/`run_benchmark`; SynthID uses
  `generate_synthid_records`/`run_synthid_benchmark`. Both share the same
  seed derivation (`seed + length*100003 + sample_index`).

CLI:
- `python -m provenance benchmark kgw --config <cfg> ...` for KGW
- `python -m provenance benchmark synthid --config <cfg> ...` for SynthID

Both write `records.jsonl`, `report.json`, and `report.txt`.

**TPR** (true-positive / detection rate) is how often a *watermarked* sample
clears a score threshold; **FPR** (false-positive rate) is how often an
*unwatermarked* sample does. Both are reported with 95% Wilson confidence
intervals. **AUC** measures watermarked-vs-unwatermarked score separability
across all thresholds. No threshold is universally correct; the threshold table
and confidence intervals exist to choose a calibrated operating point.

## SynthID Assumptions

SynthID-Text detection requires a known watermark configuration. The project
provides two SynthID paths:

- `detectors.simulated.synthid`: controlled-local-only simulation using a regex
  tokenizer and blake2b-based hashing. Labeled `implementation_kind="simulation"`,
  `compatibility="controlled-local-only"`.
- `detectors.reference.synthid`: reference-backed SynthID-Text weighted-mean
  detector. Hashing, g-value derivation, context repetition masking, and
  weighted-mean scoring follow the Google DeepMind public reference
  implementation (https://github.com/google-deepmind/synthid-text). Pure-Python
  reimplementation matches PyTorch int64 arithmetic.

The reference detector's classification depends on the tokenizer backend:

- simple/controlled tokenizer -> `implementation_kind="reference"`,
  `compatibility="synthid-reference"`. Algorithm exercised over toy token IDs.
- Hugging Face tokenizer (real model) -> `implementation_kind="reference"`,
  `compatibility="synthid-reference"`. Genuine model/tokenizer-backed SynthID
  experiment using distilgpt2.

The reference detector supports direct token-ID scoring via `score_token_ids()`
for model-backed experiments where exact token IDs from generation are available.

This detector scores *known* SynthID configurations. It is NOT general AI-vs-human
detection, does NOT identify Gemini unless the configuration is independently known
to correspond to Gemini, and does NOT detect production Gemini watermarks (which
require unavailable keys/configuration).

## Tokenizers And Models

Detectors use a tokenizer abstraction with `encode`, `decode`, vocabulary size,
and tokenizer/model identifiers. The simulation tokenizer is a local regex
vocabulary tokenizer. Hugging Face tokenizers and causal LMs are optional
adapters and are not loaded during normal installation.

Model loading is separate from detector logic. Controlled reference KGW
generation follows:

```text
prompt -> model logits -> KGW logits processor -> sampling -> token IDs -> text
```

The offline validation fixture uses `provenance-toy-causal-lm-v1` and
`kgw-reference-toy-tokenizer-v1` to avoid network and heavyweight dependency
requirements.

The genuine model-backed experiment (`configs/kgw.hf.example.json`,
`scripts/generate_kgw_hf_samples.py`) uses a real Hugging Face causal LM
(`distilgpt2`) and its tokenizer. It follows the same
`prompt -> tokenizer -> causal LM -> logits -> KGW logits processor -> sampling
-> token IDs -> text` pipeline, writing samples to
`data/generated/kgw/reference/`. The model/tokenizer are configurable and are
loaded only when the `hf` extra is installed; nothing is downloaded at
installation time. CPU execution is practical for small models (about two
minutes for two 200-token samples on `distilgpt2`).

## Unicode Analyzer

The Unicode analyzer reports zero-width characters, bidi controls, tag
characters, unusual whitespace, private-use characters, variation selectors,
format controls, and compatibility-width confusables. It does not modify input
text.

## Future Detectors

Add future detectors by subclassing `WatermarkDetector` and returning a
`DetectionResult`. Keep tokenizer/model-specific logic inside the detector
adapter and expose only structured evidence through the shared schema.

## Persistence Architecture (Phase 2B)

The API layer depends on an `AnalysisRepository` protocol defined in
`src/provenance/api/db.py`. Two backends are provided:

- **SqliteRepository** — default for local development. Uses Python's built-in
  `sqlite3` module. Database stored at `data/provenance.db`.
- **PostgresRepository** — for production deployments. Uses `psycopg2`. Activated
  via `DATABASE_URL` with a `postgresql://` scheme.

Backend selection is handled by `create_repository()` and controlled by the
`DATABASE_URL` environment variable:

| Value | Backend |
|-------|----------|
| unset (default) | SQLite at `data/provenance.db` |
| `sqlite:///path` | SQLite at path |
| `postgresql://…` | PostgreSQL |

Route handlers depend only on the protocol, so no changes are needed when
switching backends. The schema (`analyses` table) is identical across backends
and is initialized automatically on startup.

**Privacy guarantees**: raw input text is never stored (SHA-256 hash only).
Raw API keys and watermark secret keys are never persisted or logged.
Database passwords are never printed or included in connection strings.

## Production Deployment

### Docker

A multi-stage `Dockerfile` builds a minimal `python:3.10-slim` image:

- **Builder stage**: installs `build-essential`, `libpq-dev`, and the package
  with `[api,api-pg]` extras (non-editable install).
- **Runtime stage**: copies only `site-packages` and `uvicorn` binary.
  Installs `libpq5` for psycopg2 runtime. Runs as non-root user `provenance`.

```bash
docker build -t provenance-api .
docker run -p 8000:8000 \
  -e DATABASE_URL='postgresql://user:pass@db:5432/provenance' \
  -e PROVENANCE_API_KEY='production-key' \
  provenance-api
```### Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `DATABASE_URL` | No | SQLite at `data/provenance.db` |
| `PROVENANCE_API_KEY` | Production | None (auth disabled) |
| `PROVENANCE_ADMIN_API_KEY` | No | None (key management disabled) |
| `RATE_LIMIT_MAX_REQUESTS` | No | 60 |
| `RATE_LIMIT_WINDOW_SECONDS` | No | 60 |
| `MAX_LIST_LIMIT` | No | 100 |
| `CORS_ORIGINS` | No | unset (restrictive) |
| `PROVENANCE_DAILY_REQUEST_LIMIT` | No | None (unlimited) |
| `PROVENANCE_DAILY_CHARACTER_LIMIT` | No | None (unlimited) |

See `.env.example` for a template.  Never commit real credentials.

### Health Check

`GET /health` is always unauthenticated and returns:

```json
{"status": "ok", "engine_version": "0.1.0"}
```

`GET /ready` verifies the persistence backend is usable:

```json
{"status": "ready", "engine_version": "0.1.0"}
```

Suitable for Kubernetes `livenessProbe` / `readinessProbe`, load balancer
health checks, etc.  Neither endpoint exposes database credentials.

## API Hardening (Phase 2D)

### Rate Limiting

In-memory per-IP sliding-window rate limiter.  Configurable via:

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_MAX_REQUESTS` | 60 | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 60 | Window duration |

Applies to `/v1/*` endpoints only.  Returns HTTP 429 with `Retry-After`.
`/health` and `/ready` are not rate-limited.

### Request IDs

Every request receives an `X-Request-ID` header (UUID v4).  If the client
provides one, it is echoed back.  The ID appears in structured logs and
error responses.

### Structured Logging

All requests are logged with: method, path, HTTP status, duration (ms),
and request ID.  **Never logged**: raw input text, API keys, watermark
keys, DATABASE_URL credentials, database passwords.

### Error Handling

Unexpected server errors return a clean JSON response:

```json
{"detail": "Internal server error", "request_id": "..."}
```

No stack traces, credentials, or internal details are exposed.
Existing 401/403/404/422/429 behavior is preserved.

### CORS

Disabled by default (restrictive).  Set `CORS_ORIGINS` to a comma-separated
list of allowed origins.  Never use `*` in production with authentication.

### Pagination Hardening

`GET /v1/analyses` accepts a `limit` query parameter capped by
`MAX_LIST_LIMIT` (default 100).  Exceeding the cap returns 422.

## Productization Foundation (Phase 3A)

### API Key Management

API keys are stored as SHA-256 hashes in the `api_keys` table:

| Column | Type | Description |
|--------|------|-------------|
| `key_id` | TEXT PK | Unique identifier |
| `key_hash` | TEXT UNIQUE | SHA-256 of the raw secret |
| `name` | TEXT | Human-readable label |
| `status` | TEXT | `active` or `revoked` |
| `created_at` | TEXT | ISO 8601 timestamp |

The raw API key is **never** stored or returned. The legacy `PROVENANCE_API_KEY`
environment variable is supported for backwards compatibility — stored keys
are checked first, then the legacy env var as fallback.

### Usage Tracking

Every `/v1/*` request is recorded in the `usage_records` table:

| Column | Type | Description |
|--------|------|-------------|
| `key_id` | TEXT | API key identifier (or `_none`/`_legacy`) |
| `endpoint` | TEXT | Route path |
| `timestamp` | TEXT | ISO 8601 timestamp |
| `status_code` | INTEGER | HTTP status |
| `duration_ms` | REAL | Request duration |
| `character_count` | INTEGER | Input text length |
| `success` | INTEGER | 1 if status < 400, else 0 |
| `detector` | TEXT | Primary detector name (nullable) |

`GET /v1/usage` returns aggregated usage for the calling key: total requests,
successful/failed counts, characters analyzed, breakdown by endpoint and
detector.

### API Key Provisioning (Phase 3B)

Admin-protected endpoints for managing API keys:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/api-keys` | Admin | Create a key (raw secret returned once) |
| `GET` | `/v1/api-keys` | Admin | List all keys (no secrets) |
| `DELETE` | `/v1/api-keys/{id}` | Admin | Revoke a key |

The admin key is configured via `PROVENANCE_ADMIN_API_KEY`. When set, these
endpoints require it. The admin key also works as a normal API key for `/v1/*`
endpoints.

Key lifecycle:
1. Admin creates key → raw secret returned once in response
2. User stores secret securely (e.g., env var)
3. User authenticates with `X-API-Key: <secret>`
4. Admin revokes key → deactivated but record remains for audit

### Daily Limits

Optional per-key daily limits via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCE_DAILY_REQUEST_LIMIT` | None | Max requests per key per day |
| `PROVENANCE_DAILY_CHARACTER_LIMIT` | None | Max characters per key per day |

Limits are derived from the database (`usage_records` table), making them safe
for multiple API instances sharing the same database. When a limit is exceeded,
the request returns HTTP 429. These limits are separate from the per-IP rate
limiter.

### Usage Headers

Authenticated responses include usage-limit headers when limits are configured:

| Header | Description |
|--------|-------------|
| `X-Usage-Request-Limit` | Daily request limit for this key |
| `X-Usage-Requests-Remaining` | Requests remaining today |
| `X-Usage-Character-Limit` | Daily character limit for this key |
| `X-Usage-Characters-Remaining` | Characters remaining today |

### Response Metadata

`POST /v1/analyze` includes a `duration_ms` field with the server-side
processing time. This does not change the existing response schema —
it adds a new optional field.

### Database Schema (Phase 3A additions)

Both SQLite and PostgreSQL backends create three tables on initialization:

- `analyses` — analysis results (existing)
- `api_keys` — API key records (new)
- `usage_records` — per-request usage tracking (new)

Schema initialization is deterministic (`CREATE TABLE IF NOT EXISTS`) and
safe for existing databases.

### Privacy Guarantees (Phase 3A)

- Raw input text is never stored (SHA-256 hash only).
- Raw API keys are never stored or logged (SHA-256 hash only).
- Raw watermark keys are never exposed in responses or stored.
- Database passwords are never printed or included in connection strings.
- Usage records never contain raw text or API secrets.
- Structured logs never contain raw text, API keys, or credentials.

## Production Inference + Async Jobs (Phase 3C)

### Analysis Service

Analysis business logic is extracted into `src/provenance/api/service.py`,
independent of FastAPI. Both sync and async endpoints share the same
`run_analysis()` function, avoiding code duplication.

### Async Job System

A lightweight `ThreadPoolExecutor`-based background worker processes
expensive analysis jobs. This is single-process only — NOT a distributed
job queue.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /v1/analyze/async` | Yes | Submit analysis job (returns 202) |
| `GET /v1/jobs/{job_id}` | Yes | Get job status and result |
| `GET /v1/jobs` | Yes | List jobs for the calling key |

### Job Lifecycle

```
queued → running → completed (with result)
                   → failed (with error message)
```

- **queued**: Job created, waiting for a worker thread
- **running**: Worker thread executing analysis
- **completed**: Analysis finished, result stored in `result_json`
- **failed**: Analysis failed, sanitized error stored in `error_message`

### Job Ownership

Each job is associated with a `key_id`. Only the key that created the job
(or an admin key) can access it via `GET /v1/jobs/{job_id}`.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVENANCE_MAX_BACKGROUND_JOBS` | 2 | Max concurrent worker threads |
| `PROVENANCE_JOB_RETENTION_HOURS` | 24 | Hours before completed/failed jobs are cleaned up |

### Database Schema (Phase 3C)

A `jobs` table is added to both backends:

| Column | Type | Description |
|--------|------|-------------|
| `job_id` | TEXT PK | UUID identifier |
| `key_id` | TEXT | API key that created the job |
| `status` | TEXT | queued/running/completed/failed/cancelled/cancellation_requested |
| `created_at` | TEXT | ISO 8601 timestamp |
| `started_at` | TEXT | When worker started processing |
| `completed_at` | TEXT | When processing finished |
| `input_hash` | TEXT | SHA-256 of input text (raw text NOT stored) |
| `character_count` | INTEGER | Input text length |
| `detectors` | TEXT | JSON array of detector names |
| `config_path` | TEXT | Detector config path (nullable) |
| `result_json` | TEXT | Full analysis result (completed jobs only) |
| `error_message` | TEXT | Sanitized error (failed jobs only) |
| `duration_ms` | REAL | Processing time in milliseconds |
| `retry_count` | INTEGER | Number of retries (0 for original jobs) |
| `parent_job_id` | TEXT | Parent job ID for retries (nullable) |

### Privacy

- Raw input text is never stored (SHA-256 hash only)
- Error messages are sanitized (no connection strings or credentials)
- Job results follow the same privacy rules as sync analysis

## Job Reliability & Recovery (Phase 3D)

### Cancellation

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `DELETE` | `/v1/jobs/{job_id}` | Owner/Admin | Cancel a queued or running job |

Cancellation transitions:
- `queued` → `cancelled` (atomic, job will not execute)
- `running` → `cancellation_requested` (worker checks after analysis)
- `completed`/`failed` → HTTP 409 (cannot cancel)
- `cancelled` → HTTP 200 (idempotent)

The `cancel_job_if_status()` repository method provides atomic
conditional status transitions to handle race conditions between
the worker thread and the cancel request.

### Retry

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/v1/jobs/{job_id}/retry` | Owner/Admin | Retry a failed job |

Retry creates a **new** job (never mutates the original):
- `parent_job_id` links to the original job
- `retry_count` increments with each retry
- The user must re-submit the text (raw text is not stored)
- Normal rate limits, daily limits, and concurrency limits apply
- Only `failed` jobs can be retried (HTTP 409 otherwise)

### Restart Recovery

On application startup, `recover_stale_jobs()` is called:
- Jobs in `running` status → marked `failed` with recovery message
- Jobs in `queued` status → marked `failed` with recovery message
- `completed`, `failed`, and `cancelled` jobs are untouched

This prevents jobs from being permanently stuck if the process crashes.

### Valid State Transitions

```
queued → running → completed
queued → cancelled (via cancel)
running → failed (via error)
running → cancellation_requested → cancelled
failed → retry creates NEW queued job
```

Invalid transitions (rejected with HTTP 409):
- Cannot cancel `completed` or `failed` jobs
- Cannot retry non-`failed` jobs

### Updated Job Fields

`JobSummary` and `JobResponse` now include:
- `retry_count` (int, default 0)
- `parent_job_id` (str | None)

These fields are backward compatible — existing jobs return
`retry_count: 0` and `parent_job_id: null`.

## Production Observability & Operational Controls (Phase 3E)

### Metrics Endpoint

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/metrics` | No | Machine-readable application metrics |

Returns JSON with:
- Analysis request totals (total, successful, failed)
- Total characters analyzed
- Average analysis duration (ms)
- Breakdowns by endpoint, detector, and status code
- Job queue counts (queued, running, completed, failed, cancelled)
- Configured worker count

No external dependencies (no Prometheus). Raw API keys, text hashes,
job IDs, and other secrets are never exposed.

### Enhanced Readiness Probe

`GET /ready` now reports:
- `persistence`: database connectivity (`ok` or `error`)
- `executor`: background worker state (`ok`, `shutting_down`, or `error`)
- `status`: overall readiness (`ready` only when all subsystems are `ok`)

Returns HTTP 200 when ready, but the status field should be checked
programmatically.

### Configuration Validation

Operational environment variables are validated at startup via
`validate_config()` in `provenance.api.config`:

| Variable | Constraint | Default |
|----------|-----------|---------|
| `PROVENANCE_MAX_BACKGROUND_JOBS` | >= 1 | 2 |
| `PROVENANCE_JOB_RETENTION_HOURS` | >= 1 | 24 |
| `RATE_LIMIT_MAX_REQUESTS` | >= 1 | 60 |
| `RATE_LIMIT_WINDOW_SECONDS` | >= 1 | 60 |
| `MAX_LIST_LIMIT` | >= 1 | 100 |
| `PROVENANCE_DAILY_REQUEST_LIMIT` | >= 1 if set | None |
| `PROVENANCE_DAILY_CHARACTER_LIMIT` | >= 1 if set | None |

Invalid values log a warning at startup via `ConfigError` but do not
crash the application. Validation runs before the repository is created.

### Graceful Shutdown

Shutdown sequence:
1. `_shutting_down` flag is set (new jobs rejected by executor)
2. `ThreadPoolExecutor.shutdown(wait=True)` blocks until running jobs finish
3. Final job state is persisted by the worker threads
4. Executor reference is cleared
5. Repository is closed

This ensures no job state is lost when the process terminates cleanly.

### Repository Extensions (Phase 3E)

New methods on `AnalysisRepository`:

| Method | Description |
|--------|-------------|
| `get_job_counts()` | Returns counts by status (queued, running, completed, etc.) |
| `get_avg_duration()` | Returns average duration across all usage records |
| `get_usage_by_status()` | Returns request counts grouped by HTTP status code |

### Module Structure

| Module | Purpose |
|--------|--------|
| `config.py` | Environment variable validation |
| `state.py` | Shared application state (repo reference) |
| `keys.py` | API key generation, hashing, validation |
| `service.py` | Analysis business logic (sync + async) |
| `jobs.py` | Background job executor and lifecycle |
| `routes.py` | FastAPI route handlers |
| `auth.py` | Authentication dependencies |
| `middleware.py` | Rate limiting, logging, CORS, exception handling |
| `models.py` | Pydantic request/response models |
| `db.py` | Persistence layer (SQLite + PostgreSQL) |
| `app.py` | Application factory |

### Updated Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| `PROVENANCE_MAX_BACKGROUND_JOBS` | Max concurrent worker threads | 2 |
| `PROVENANCE_JOB_RETENTION_HOURS` | Hours to keep completed/failed jobs | 24 |

These variables are validated at startup (Phase 3E).

### API Reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Lightweight health check |
| `GET` | `/ready` | No | Readiness probe (DB + executor) |
| `GET` | `/metrics` | No | Application metrics (JSON) |
| `POST` | `/v1/analyze` | Yes | Analyze text |
| `POST` | `/v1/analyze/async` | Yes | Submit background analysis |
| `GET` | `/v1/analyses` | Yes | List analyses |
| `GET` | `/v1/analyses/{id}` | Yes | Get analysis by ID |
| `GET` | `/v1/usage` | Yes | Usage statistics |
| `POST` | `/v1/api-keys` | Admin | Create API key |
| `GET` | `/v1/api-keys` | Admin | List API keys |
| `DELETE` | `/v1/api-keys/{id}` | Admin | Revoke API key |
| `GET` | `/v1/jobs/{job_id}` | Yes | Get job status |
| `GET` | `/v1/jobs` | Yes | List jobs |
| `DELETE` | `/v1/jobs/{job_id}` | Yes | Cancel job |
| `POST` | `/v1/jobs/{job_id}/retry` | Yes | Retry failed job |

## Python SDK (Phase 4A)

A typed Python client (`provenance_client`) is included for integrating with the API.
It uses only the standard library (`urllib.request`) — no external dependencies.

### Package Structure

```
src/provenance_client/
├── __init__.py       # Public API: ProvenanceClient + exceptions
├── _client.py        # HTTP client with typed methods
├── _exceptions.py    # Exception hierarchy (maps HTTP status codes)
└── _models.py        # Typed response dataclasses
```

### Exception Hierarchy

| Exception | HTTP Status | Description |
|-----------|-------------|-------------|
| `AuthenticationError` | 401 | Missing or invalid API key |
| `ForbiddenError` | 403 | Invalid key or admin required |
| `NotFoundError` | 404 | Resource not found |
| `ConflictError` | 409 | Conflict (e.g., cancel completed job) |
| `RateLimitError` | 429 | Rate/daily limit exceeded |
| `ValidationError` | 422 | Invalid request body |
| `TimeoutError` | — | Transport timeout |
| `ProvenanceAPIError` | other | Base class for all errors |

### SDK Methods

| Method | API Endpoint | Description |
|--------|-------------|-------------|
| `analyze()` | `POST /v1/analyze` | Synchronous analysis |
| `analyze_async()` | `POST /v1/analyze/async` | Submit background analysis |
| `get_job()` | `GET /v1/jobs/{id}` | Get job status and result |
| `list_jobs()` | `GET /v1/jobs` | List jobs for calling key |
| `cancel_job()` | `DELETE /v1/jobs/{id}` | Cancel queued/running job |
| `retry_job()` | `POST /v1/jobs/{id}/retry` | Retry failed job |
| `wait_for_job()` | Polls `GET /v1/jobs/{id}` | Poll until terminal state |
| `get_usage()` | `GET /v1/usage` | Usage statistics |

### Privacy Guarantees

- API keys are never logged or included in exception messages.
- Raw input text is never stored (SHA-256 hash only).
- SDK uses standard library HTTP only (no telemetry dependencies).

## Web Dashboard (Phase 4B)

### Architecture

```
Browser
   |
   v
Dashboard (React + TypeScript + Vite)
   |
   v
Existing FastAPI API (/v1/*)
   |
   v
AnalysisService / JobSystem
   |
   v
Detectors
   |
   v
SQLite/PostgreSQL
```

### File Structure

```
dashboard/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── src/
    ├── main.tsx              # React entry point
    ├── App.tsx                # Config provider, nav, page routing
    ├── api/
    │   └── client.ts          # TypeScript API client
    ├── components/
    │   ├── SetupScreen.tsx     # API URL + key setup
    │   ├── StatCard.tsx        # Reusable stat card
    │   ├── BarChart.tsx        # Simple horizontal bar chart
    │   └── StatusBadge.tsx     # Color-coded status indicator
    ├── pages/
    │   ├── Overview.tsx        # Metrics, recent analyses, charts
    │   ├── Analyze.tsx         # Text analysis with detector selection
    │   ├── History.tsx         # Paginated analysis history
    │   ├── Jobs.tsx            # Job management (view/cancel/retry)
    │   └── Usage.tsx           # Usage statistics and breakdowns
    ├── hooks/
    │   └── useConfig.ts       # React context for API config
    └── types/
        └── api.ts             # TypeScript types matching API contracts
```

### Dashboard Pages

| Page | Description | Key Endpoints Used |
|------|-------------|--------------------|
| Overview | Metrics, detector usage, recent analyses | `GET /metrics`, `GET /v1/analyses` |
| Analyze | Text analysis with detector selection | `POST /v1/analyze` |
| History | Paginated analysis list with detail | `GET /v1/analyses`, `GET /v1/analyses/{id}` |
| Jobs | View, cancel, retry background jobs | `GET /v1/jobs`, `DELETE /v1/jobs/{id}`, `POST /v1/jobs/{id}/retry` |
| Usage | Usage statistics and breakdowns | `GET /v1/usage` |

### Authentication

- API key stored in `sessionStorage` (cleared on tab close)
- Sent via `X-API-Key` header only
- Never logged or stored in `localStorage`
- Setup screen tests connection before saving

### Backend Additions

No new backend endpoints were added for Phase 4B.
The dashboard uses only existing API endpoints:
`/metrics`, `/ready`, `/health`, `/v1/analyze`, `/v1/analyses`, `/v1/jobs`, `/v1/usage`.

## Detection Engine Expansion (Phase 5A)

### Detector Registry

A centralized registry (`src/provenance/detectors/registry.py`) provides:
- Single source of truth for supported detector names
- Machine-readable capability metadata per detector
- Factory pattern for detector instantiation
- Eliminates scattered detector selection logic across CLI/API

Registered detectors: `unicode`, `kgw`, `kgw-reference`, `synthid`, `synthid-reference`.

### Detector Capability Metadata

Each detector exposes:
- `name`, `display_name`
- `implementation_kind` (unicode, watermark-kgw, watermark-synthid)
- `compatibility` (any, gpt-2, gemini)
- `requires_config` (bool)
- `supports_generation`, `supports_benchmarking`
- `tokenizer_requirements`
- `known_limitations`
- `description`

### GET /v1/detectors

Authenticated endpoint returning the detector registry capabilities.
No secrets, watermark keys, or filesystem paths are exposed.
Deterministic response — useful for SDK/dashboard clients.

### Robustness Evaluation Framework

Located in `src/provenance/robustness/`:

- **transforms.py** — Named, deterministic text transformations:
  identity, whitespace normalization, Unicode NFC/NFD, lowercase,
  uppercase, punctuation normalization, blank line stripping,
  whitespace injection, double spaces.
- **evaluator.py** — Runs detector against transformed text.
  Records contain only SHA-256 hashes, never raw text.
  Computes detection rate and score changes per transformation.

### CLI Robustness Benchmark

```bash
python -m provenance benchmark robustness \
  --text input.txt --detector unicode
```

Produces per-transformation detection rates and score deltas.
Reuses the existing benchmark output conventions.

### Watermark Robustness Pipeline (Phase 5B)

The watermark robustness experiment follows this pipeline:

1. Load detector from config
2. Generate watermarked samples using the existing generation pipeline
   (KGW or SynthID logits processor)
3. For each sample:
   a. Detect original text
   b. For each transformation: transform -> detect -> compare
4. Compute robustness metrics:
   - Baseline detection rate
   - Transformed detection rate
   - Robustness rate (fraction of baseline-detected samples still detected after transformation)
   - Mean score delta per transformation
5. Render report with explicit limitations

The pipeline loads the model once and generates each sample once,
then transforms and detects multiple times per sample.

### Robustness Metrics

- **baseline_detection_rate**: fraction of identity-transform samples detected
- **transformed_detection_rate**: fraction of non-identity samples detected
- **robustness_rate**: fraction of baseline-detected samples still detected after transformation (paired per-sample comparison; always in [0, 1])
- **mean_score_delta**: average score change across all transformations

These metrics describe behavior under the tested transformations only.
They do not establish resistance to adversarial attacks, watermark removal,
paraphrasing, or generic AI-text detection.

### Benchmark Reporting Layer (Phase 5C)

Located in `src/provenance/robustness/benchmark.py`:

- **RobustnessBenchmarkResult** — Versioned, machine-readable benchmark result.
  Each result identifies: schema version, detector/scheme, config identifier,
  transform, text length, sample count, seed, baseline/transformed detection
  rates with Wilson CIs, robustness rate, mean scores, detection change count.
  Never stores raw generated text.

- **AggregatedResult** — Cross-experiment aggregation by
  (detector, config, transform). Supports comparison across detectors,
  models, text lengths, and transformations.

- **Wilson confidence intervals** — 95% Wilson score intervals for
  detection/robustness rates. Reuses the same statistical method as
  the existing benchmark calibration module.

- **RobustnessMatrix** — Machine-readable matrix (detectors × transforms)
  for future dashboard visualization.

- **Persistence** — JSONL format for benchmark results. Supports
  write/read roundtrip and directory-based loading.

#### CLI

```bash
# Save benchmark results
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw --out-dir data/robustness/kgw

# Aggregate saved results
python -m provenance benchmark robustness-report \
  --input data/robustness/ --json
```

#### JSON Output Contract

All reports use stable field names and include `schema_version`.
Field names are lowercase_snake_case. Reports are JSON-serializable
and avoid dumping implementation-specific Python objects.

#### API Preparation

The aggregation/reporting code is structured as pure functions over
result dataclasses, so it can be exposed through the FastAPI API
and dashboard without rewriting core logic.

### Advanced Robustness Evaluation (Phase 5D)

Located in `src/provenance/robustness/advanced_transforms.py` and
`src/provenance/robustness/profiles.py`:

#### Transformation Taxonomy

Transformations are organized into categories:

- **formatting** (`TransformCategory.FORMATTING`): paragraph_reflow, line_wrap_normalize, blank_line_normalize
- **whitespace** (`TransformCategory.WHITESPACE`): whitespace_collapse, whitespace_expand, tab_space_normalize, double_spaces, leading_trailing_whitespace
- **unicode** (`TransformCategory.UNICODE`): unicode_nfc, unicode_nfd, unicode_nfkd, unicode_punctuation_normalize
- **casing** (`TransformCategory.CASING`): lowercase, uppercase, title_case
- **punctuation** (`TransformCategory.PUNCTUATION`): punctuation_normalize, repeated_punctuation_normalize
- **lexical** (`TransformCategory.LEXICAL`): conservative_synonym_substitution, contraction_expansion
- **tokenization-sensitive** (`TransformCategory.TOKENIZATION_SENSITIVE`): insert_formatting_boundaries, remove_formatting_boundaries

#### Metadata

Each `AdvancedTransform` exposes:
- `name`: unique identifier
- `category`: TransformCategory enum value
- `description`: human-readable description
- `deterministic`: always True
- `severity`: low, medium, or high (relative text modification magnitude)

#### Transform Configuration

`TransformConfig` provides structured configuration:
- transform name
- category
- enabled/disabled
- optional severity
- seed for reproducibility

JSON-serializable for persistence.

#### Profiles

Predefined profiles group transforms by category or purpose:
- `formatting`, `unicode`, `whitespace`, `casing`, `punctuation`, `lexical`, `tokenization-sensitive`, `all_safe`

A profile expands into a deterministic list of transform names.
Profiles must only SELECT transformations — they must not perform
adaptive optimization.

#### CLI Integration

```bash
# Use a specific profile
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw --profile unicode

# Run all safe transformations
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw --profile all_safe
```

The `--profile` argument selects a predefined set of transforms.
The `--transforms` argument (Phase 5A) selects individual transforms.
Default behavior (no --profile or --transforms) uses all Phase 5A transforms.

#### Category Aggregation in Reports

Benchmark reports can include category-level aggregation:
- Per-category robustness rates with Wilson CIs
- Per-category mean score deltas
- JSON output for programmatic consumption

#### Safety Boundary

These transforms measure robustness under controlled perturbations.
They do NOT implement:
- Adaptive attacks or optimization against detector scores
- Gradient-based evasion
- Automated paraphrasing designed to evade detection
- Watermark removal
- External LLM queries for adversarial transformations

The purpose is measurement, not evasion.

### Benchmark Orchestration (Phase 5E)

Located in `src/provenance/robustness/orchestration.py`:

#### Benchmark Plan Schema

`BenchmarkPlan` contains:
- `schema_version`: versioned for forward compatibility
- `name`: human-readable plan identifier
- `specs`: tuple of `BenchmarkSpec` objects
- `output_dir`: default output directory
- `metadata`: arbitrary metadata

Each `BenchmarkSpec` contains:
- `detector`, `config`, `lengths`, `samples`, `seed`
- `profile` or `transforms` (optional)
- `experiment_id`: deterministic SHA-256 of normalized spec

#### Deterministic Experiment IDs

Experiment IDs are computed as SHA-256 of the canonical JSON of:
detector/config/lengths/samples/seed/profile/transforms.

No timestamps are included — the same spec always produces the same ID.
This enables resume logic to identify identical experiments.

#### Plan Validation

`validate_plan()` checks:
- Non-empty plan name
- At least one experiment spec
- Known detector names
- Config required for watermark detectors
- Positive lengths and sample counts
- Known profiles
- Duplicate experiment detection

Returns useful validation errors with field and spec index.

#### Run Manifests

`RunManifest` tracks:
- `run_id`: unique identifier for the run
- `benchmark_version`: schema version
- `plan_name`: reference to the plan
- `started_at` / `completed_at`: timestamps
- `experiments`: list of `ExperimentManifest` entries
- `status`: running/completed/partial

Each `ExperimentManifest` tracks:
- `experiment_id`: deterministic ID
- `spec`: the experiment specification
- `status`: pending/running/completed/failed/skipped
- `result_files`: references to output files
- `error_type` / `error_message`: sanitized error info

Manifests are saved after each experiment for crash recovery.
No raw text, API keys, or watermark keys are stored.

#### Resume / Partial Runs

With `--resume`:
- Loads existing manifest from output directory
- Skips completed experiments (unless `--force`)
- Continues pending/failed experiments
- Preserves existing results

With `--force`:
- Re-runs all experiments regardless of status
- Overwrites existing results

#### Failure Isolation

Each experiment runs independently. One failure does not stop others.
Failed experiments record error type and sanitized message.
The manifest status is "partial" if any experiment fails.

#### Cross-Model Comparison

`ComparisonReport` provides:
- Per-experiment rows with Wilson CIs
- By-model aggregation
- By-transform aggregation
- By-category aggregation (with category_map)
- By-length aggregation

The report explicitly does NOT produce a single "best detector" score
across incompatible watermark schemes.

#### CLI

```bash
# Run a plan
python -m provenance benchmark plan --plan benchmark_plan.json

# Dry run
python -m provenance benchmark plan --plan benchmark_plan.json --dry-run

# Resume
python -m provenance benchmark plan --plan benchmark_plan.json --out-dir data/run --resume

# Force re-run
python -m provenance benchmark plan --plan benchmark_plan.json --out-dir data/run --resume --force
```

#### JSON Output Contract

Reports use stable field names and include `schema_version`.
Field names are lowercase_snake_case.
Reports are JSON-serializable without custom Python objects.
