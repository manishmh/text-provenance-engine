# Text Provenance Engine

Phase 1 is a local Python engine for detecting text-level provenance signals.
It reports deterministic Unicode artifacts and evidence for known watermark
configurations. It does not classify arbitrary text as AI-generated or human.

## Setup

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

The source checkout also supports:

```bash
python -m provenance analyze fixtures/unicode.txt
```

## CLI

```bash
python -m provenance analyze fixtures/unicode.txt
python -m provenance analyze fixtures/unicode.txt --json
python -m provenance analyze sample.txt --detector unicode
python -m provenance analyze sample.txt --detector kgw --config configs/kgw.example.json
python -m provenance analyze sample.txt --detector kgw-reference --config configs/kgw.reference.example.json
python -m provenance analyze sample.txt --detector synthid --config configs/synthid.example.json
python -m provenance analyze sample.txt --detector synthid-reference --config configs/synthid.reference.example.json
```

`kgw` and `synthid` are controlled simulations and their JSON output includes
`implementation_kind: "simulation"` and `compatibility: "controlled-local-only"`.
Use `kgw-reference` for the model-token KGW research detector.
Use `synthid-reference` for the SynthID-Text reference detector (simple or HuggingFace tokenizer).

Watermark configs must match the configuration used to generate the text. A
positive KGW result means evidence for that configured signal only.

## Python

```python
from provenance import ProvenanceEngine

engine = ProvenanceEngine()
result = engine.analyze("plain text")
print(result.to_dict())
```

## Controlled Samples

Simulation samples:

```bash
python scripts/generate_kgw_samples.py --config configs/kgw.example.json --out-dir data/generated/kgw
python scripts/generate_synthid_samples.py --config configs/synthid.example.json --out-dir data/generated/synthid
```

Controlled (toy-tokenizer) reference-algorithm samples. These exercise the KGW
math over a local vocabulary and are labelled
`implementation_kind: "reference-adapted"`, `compatibility: "controlled-local-only"`
because the token IDs do not come from a real model:

```bash
python scripts/generate_kgw_reference_samples.py --config configs/kgw.reference.example.json --out-dir data/generated/kgw
python -m provenance analyze data/generated/kgw/watermarked.txt --detector kgw-reference --config configs/kgw.reference.example.json --json
python -m provenance analyze data/generated/kgw/unwatermarked.txt --detector kgw-reference --config configs/kgw.reference.example.json --json
```

Genuine model/tokenizer-backed KGW experiment (requires the optional `hf`
extra; downloads a small causal LM on first use). Only this path is labelled
`implementation_kind: "reference"`, `compatibility: "kgw-reference"`:

```bash
pip install -e '.[dev,hf]'
python scripts/generate_kgw_hf_samples.py --config configs/kgw.hf.example.json --out-dir data/generated/kgw/reference
python -m provenance analyze data/generated/kgw/reference/watermarked.txt   --detector kgw-reference --config configs/kgw.hf.example.json --json
python -m provenance analyze data/generated/kgw/reference/unwatermarked.txt --detector kgw-reference --config configs/kgw.hf.example.json --json
```

The generation pipeline is `prompt -> tokenizer -> causal LM -> logits -> KGW
logits processor -> sampling -> token IDs -> text`. The watermark influences the
model's logits; green tokens are never selected directly.

Generated metadata records the scheme, configuration id, prompt id, model and
tokenizer identifiers and revisions, KGW parameters (with a hash-key *identifier*
rather than the secret key), generation parameters, and the ground-truth
watermark flag.

A successful KGW detection demonstrates evidence consistent with the tested KGW
configuration. It does not identify the model provider or prove that text was
AI-generated.

## SynthID-Text Reference Detector

The reference-backed SynthID detector (`synthid-reference`) implements the
weighted-mean detection algorithm from Google DeepMind's public SynthID-Text
reference implementation. It supports two tokenizer backends:

| Tokenizer backend | `implementation_kind` | `compatibility` |
|-------------------|-----------------------|-----------------|
| simple/controlled vocabulary | `reference` | `synthid-reference` |
| Hugging Face (real model) | `reference` | `synthid-reference` |

Genuine model/tokenizer-backed SynthID experiment (requires the optional `hf`
extra; downloads a small causal LM on first use):

```bash
pip install -e '.[dev,hf]'
# Generate watermarked/unwatermarked samples with distilgpt2
python -c "
from provenance.configuration import ExperimentConfig
from provenance.detectors.reference.synthid import SynthIDReferenceConfig, SynthIDReferenceDetector
from provenance.generation.synthid import SynthIDGenerationConfig, SynthIDLogitsProcessor, generate_synthid_token_ids
from provenance.models import HuggingFaceCausalLM
from provenance.tokenizers import tokenizer_from_config

experiment = ExperimentConfig.from_file('configs/synthid.hf.example.json')
tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())
model = HuggingFaceCausalLM(experiment.model)
config = SynthIDReferenceConfig.from_file('configs/synthid.hf.example.json')
detector = SynthIDReferenceDetector(config, tokenizer=tokenizer)
processor = SynthIDLogitsProcessor(config)
gen_config = SynthIDGenerationConfig(max_new_tokens=50, seed=42)
prompt_ids = tokenizer.encode('The future of AI is shaped by')
wm_ids = generate_synthid_token_ids(model=model, prompt_token_ids=prompt_ids, processor=processor, generation_config=gen_config, watermarked=True)
result = detector.score_token_ids(wm_ids)
print(f'Score: {result.score:.4f}, Detected: {result.detected}')
"
```

The generation pipeline is `prompt -> tokenizer -> causal LM -> logits -> SynthID
logits processor -> sampling -> token IDs -> text`. The watermark influences the
model's logits through a weighted-mean scoring of g-values; tokens are never
selected directly by the watermark.

Classification: `implementation_kind="reference"`, `compatibility="synthid-reference"`.
This is evidence for the tested SynthID configuration only. It does NOT identify
Gemini, does NOT prove AI authorship, and does NOT detect production Gemini watermarks
(which require unavailable keys/configuration).

## KGW Reference Variant

The reference-backed KGW detector implements `kgw-python-left-v1`: a
MarkLLM-style left-hash KGW variant over exact token IDs with configurable
`gamma`, `delta`, `hash_key`, `prefix_length`, `f_scheme`, z-score, p-value, and
optional repeated n-gram handling.

The same detector runs in two modes, distinguished by the tokenizer backend:

| Tokenizer backend | `implementation_kind` | `compatibility` |
|-------------------|-----------------------|-----------------|
| simple/controlled vocabulary | `reference-adapted` | `controlled-local-only` |
| Hugging Face (real model) | `reference` | `kgw-reference` |

The `reference-adapted` toy path (`provenance-toy-causal-lm-v1` /
`kgw-reference-toy-tokenizer-v1`) runs offline and backs the deterministic unit
tests. The `reference` path requires the optional `hf` dependency; no large model
is downloaded during installation.

```bash
pip install -e '.[dev,hf]'
```

### Upstream compatibility: reference-aligned but RNG-incompatible

`kgw-python-left-v1` is a structural port of MarkLLM's KGW left-hash/additive
scheme — identical `gamma`, `delta`, greenlist size, seeding formula, z-score, and
p-value — but it uses a pure-Python `random.shuffle` permutation where MarkLLM
uses PyTorch `randperm`. That substitution was cross-checked against the genuine
upstream code (`scripts/upstream_kgw_crosscheck.py`, using `markllm`), and it is
decisive rather than cosmetic:

- Our recorded watermarked token IDs score **z=10.85** under our detector but
  **z=0.53 (undetected)** under the genuine MarkLLM detector.
- Directly-built greenlists have identical size but their overlap equals random
  chance — the greenlists are statistically independent.
- The two implementations **cannot detect each other's watermarks**.

Classification for `kgw-python-left-v1`: **reference-aligned but RNG-incompatible**. This is a valid,
self-consistent KGW research detector for text generated with *this* variant. It
is **not** interoperable with MarkLLM or `lm-watermarking`, and is **not**
production- or byte-compatible with any external KGW deployment. Full evidence:
[`docs/kgw-upstream-crosscheck.md`](docs/kgw-upstream-crosscheck.md).

To address interoperability, a second variant `kgw-markllm-v1` is provided. This variant
uses PyTorch `torch.randperm` to exactly reproduce MarkLLM 0.1.5's behavior. It has been
verified to have identical greenlists, equivalent detection scores, and full generation
interoperability with the upstream MarkLLM package.

## KGW Benchmark

The benchmark subsystem measures the MarkLLM-compatible detector
(`kgw-markllm-v1`) across token lengths and z-thresholds:

```bash
pip install -e '.[dev,hf]'
python -m provenance benchmark kgw \
  --config configs/kgw.markllm.hf.example.json \
  --lengths 50,100,200,500 \
  --samples 10 \
  --seed 42 \
  --out-dir data/benchmarks/kgw
```

For each length it generates matched watermarked and unwatermarked samples (the
pair shares a derived seed, so the only difference is the KGW logit bias), scores
each, and writes `records.jsonl`, `report.json`, and `report.txt` to the output
directory. The secret key is never stored --- records carry only `hash_key_id`.

The report separates **watermarked** and **unwatermarked** samples and reports:

- **TPR (true-positive / detection rate)** --- how often a *watermarked* sample
  clears a z-threshold. Higher is better.
- **FPR (false-positive rate)** --- how often an *unwatermarked* sample clears a
  z-threshold. Under the null, unwatermarked z-scores are ~`N(0, 1)`, so FPR at
  threshold `t` tracks the normal tail. Lower is better.
- **95% Wilson confidence intervals** on every TPR and FPR, alongside the sample
  count. An observed FPR of `0/20` is reported as `0.000 [0.000, 0.161]`: the
  observed rate is zero, but the true rate is only bounded below ~0.16 at this
  sample size. A zero count never means the true rate is zero.
- **ROC AUC** per length and pooled --- the probability a random watermarked
  z-score exceeds a random unwatermarked one (`1.0` = perfect separation, `0.5`
  = chance), plus the ROC curve points in `report.json` for plotting. AUC
  summarizes separability across *all* thresholds, independent of any single
  operating point.

Because KGW z-scores grow roughly with `sqrt(length)`, short texts carry less
signal and show lower TPR at a fixed threshold; evaluating several lengths makes
that limitation visible. No z-threshold is universally correct, so the report
gives a TPR/FPR table (with confidence intervals) at thresholds 2/3/4/5 to pick a
calibrated operating point. Use `--samples 50` (or more) for tighter intervals; a
16-prompt deterministic pool keeps samples from collapsing onto one continuation.

This benchmark evaluates a *known* KGW configuration. A high TPR means that
configured signal is detectable in text produced with that exact configuration.
It is not general AI-vs-human detection and does not attribute text to any model
provider.

## SynthID Benchmark

The benchmark infrastructure is scheme-agnostic and also supports SynthID-Text
evaluation:

```bash
python -m provenance benchmark synthid \
  --config configs/synthid.hf.example.json \
  --lengths 50,100,200 \
  --samples 10 \
  --seed 42 \
  --out-dir data/benchmarks/synthid
```

For each length it generates matched watermarked/unwatermarked samples using the
SynthID logits processor, scores each with the reference detector, and produces
the same report format (TPR/FPR, Wilson CIs, ROC AUC) as the KGW benchmark.

This evaluates detection of a *known* SynthID configuration. It is not general
AI detection and does not identify Gemini (production keys are unavailable).

## HTTP API (Phase 2)

The engine is exposed through a local REST API with pluggable persistence
(SQLite or PostgreSQL) and optional API-key authentication.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | SQLite at `data/provenance.db` |
| `PROVENANCE_API_KEY` | API key for `/v1/*` auth | unset (auth disabled) |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window for `/v1/*` | 60 |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window in seconds | 60 |
| `MAX_LIST_LIMIT` | Max `limit` param for `GET /v1/analyses` | 100 |
| `CORS_ORIGINS` | Comma-separated allowed origins | unset (restrictive) |

Copy `.env.example` to `.env` and fill in values for production.
Never commit `.env` with real credentials.

### Local Development

```bash
pip install -e '.[dev,api]'
# Optional: set an API key (without this, auth is disabled for local dev)
export PROVENANCE_API_KEY='your-secret-key'
uvicorn provenance.api.app:app --host 0.0.0.0 --port 8000
```

### Production (PostgreSQL)

```bash
pip install -e '.[api,api-pg]'
export DATABASE_URL='postgresql://user:password@localhost:5432/provenance'
export PROVENANCE_API_KEY='your-production-key'
uvicorn provenance.api.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
podman build -t provenance-api .   # or: docker build -t provenance-api .

# Run (SQLite, local dev)
podman run -p 8000:8000 provenance-api

# Run (PostgreSQL, production)
podman run -p 8000:8000 \
  -e DATABASE_URL='postgresql://user:password@db-host:5432/provenance' \
  -e PROVENANCE_API_KEY='your-production-key' \
  provenance-api
```

The image builds a minimal `python:3.10-slim` image, installs only
`fastapi`, `uvicorn`, and `psycopg2-binary`, and runs as a non-root user.

### Authentication

All `/v1/*` endpoints require an `X-API-Key` header when `PROVENANCE_API_KEY`
is set. The `/health` endpoint is always public.

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-secret-key' \
  -d '{"text": "Hello, world!", "detectors": ["unicode"]}'
```

| Condition | Status |
|-----------|--------|
| Missing `X-API-Key` header | 401 |
| Invalid API key | 403 |
| `PROVENANCE_API_KEY` unset | No auth required (local dev) |

### Health Check

`GET /health` and `GET /ready` are always unauthenticated:

```bash
curl http://localhost:8000/health
# {"status": "ok", "engine_version": "..."}

curl http://localhost:8000/ready
# {"status": "ready", "engine_version": "..."}
```

`/ready` verifies the persistence backend is usable.

### Request IDs

Every request receives an `X-Request-ID` header (UUID). If the client
provides one, it is echoed back. The ID appears in structured logs.

### Rate Limiting

`/v1/*` endpoints are rate-limited per client IP. Exceeding the limit
returns HTTP 429 with a `Retry-After` header. `/health` and `/ready`
are not rate-limited.

### CORS

Cross-origin requests are disabled by default. Set `CORS_ORIGINS` to a
comma-separated list of allowed origins to enable CORS.

### Endpoints

| Method | Path | Auth | Rate-limited | Description |
|--------|------|------|-------------|-------------|
| `GET` | `/health` | No | No | Service health check |
| `GET` | `/ready` | No | No | Readiness probe (checks DB) |
| `POST` | `/v1/analyze` | Yes | Yes | Analyze text with selected detectors |
| `GET` | `/v1/analyses/{id}` | Yes | Yes | Retrieve a persisted analysis |
| `GET` | `/v1/analyses` | Yes | Yes | List analyses (paginated) |

### Input limits

- Maximum text length: 200,000 characters.
- Detector names must be from: `unicode`, `kgw`, `kgw-reference`, `synthid`, `synthid-reference`.
- Watermark detectors (`kgw`, `kgw-reference`, `synthid`, `synthid-reference`) require `config_path`.

### Persistence

Analysis results are stored in a database selected via the `DATABASE_URL` environment variable.

| `DATABASE_URL` | Backend | Use case |
|----------------|---------|----------|
| unset (default) | SQLite at `data/provenance.db` | Local development |
| `sqlite:///path` | SQLite at path | Explicit SQLite |
| `postgresql://…` | PostgreSQL | Production |

The schema is initialized automatically on first startup.
PostgreSQL support requires `psycopg2-binary` (`pip install -e '.[api,api-pg]'`).

### Security & Logging Guarantees

- **Raw input text is never stored** (only its SHA-256 hash).
- **Raw watermark keys** are never exposed in API responses or database records.
- **Raw API keys** are never persisted or logged.
- Database passwords are never printed or included in connection strings.
- Only identifiers like `hash_key_id` appear in results.
- Structured logs include method, path, status, request ID, and duration only.
- Error responses return clean JSON without stack traces or internal details.
- CORS is disabled by default (no cross-origin requests allowed).

## Tests

```bash
pytest
```

Live PostgreSQL integration tests (when PostgreSQL is available):

```bash
export PROVENANCE_TEST_PG_DSN='postgresql://user@/dbname?options=-c search_path%3Dtest_schema'
pytest tests/test_api.py -k pg_live -v
```
