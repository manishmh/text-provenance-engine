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

## Quick Start

### 1. Start the API

```bash
pip install -e '.[api]'
cp .env.example .env  # then fill in local values
python -m uvicorn --env-file .env --factory provenance.api.app:create_app --host 0.0.0.0 --port 8000
```

### 2. Create an API key (optional, for managed keys)

```bash
export PROVENANCE_ADMIN_API_KEY='admin-secret'
curl -X POST http://localhost:8000/v1/api-keys \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: admin-secret' \
  -d '{"name": "My App Key"}'
# Returns: {"key_id": "...", "key": "<raw-secret-shown-once>"}
```

### 3. Run analysis

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: my-secret-key' \
  -d '{"text": "Hello, world!", "detectors": ["unicode"]}'
```

### 4. Run async analysis

```bash
# Submit
RESP=$(curl -s -X POST http://localhost:8000/v1/analyze/async \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: my-secret-key' \
  -d '{"text": "Long text..."}')
JOB_ID=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

# Poll for result
curl http://localhost:8000/v1/jobs/$JOB_ID -H 'X-API-Key: my-secret-key'
```

## Python SDK

Install the SDK (no external dependencies):

```bash
pip install -e .
```

Usage:

```python
from provenance_client import ProvenanceClient

client = ProvenanceClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
)

# Synchronous analysis
result = client.analyze(
    text="Hello, world!",
    detectors=["unicode"],
)
print(result.status, result.analysis_id)

# Async analysis + poll
job = client.analyze_async(text="Long text...")
final = client.wait_for_job(job.job_id)
print(final.status, final.result)

# Usage statistics
usage = client.get_usage()
print(f"{usage.total_requests} requests, {usage.total_characters} chars")
```

See `tests/test_sdk.py` for full SDK usage examples.

## Web Dashboard

A React + TypeScript dashboard is included for visualizing API data.

### Development

```bash
cd dashboard
npm install
npm run dev
# Opens at http://localhost:5173
```

### Production Build

```bash
cd dashboard
npm run build
dashboard/dist/ contains the static build
```

### Features

- **Overview**: Metrics, detector usage, recent analyses
- **Analyze**: Text input with detector selection and result display
- **History**: Paginated analysis history with detail view
- **Jobs**: View, cancel, and retry background analysis jobs
- **Usage**: Request/character usage with endpoint and detector breakdowns
- **Detectors**: Supported detectors with capabilities and configuration requirements (`GET /v1/detectors`)
- **Settings**: API server connection and developer setup; the connection form also appears once on first launch
- **Benchmarks**: Configure, queue, monitor, cancel, retry, and inspect robustness benchmark runs; drill into results in Robustness
- **Robustness**: Watermark robustness matrix, detector/model/category/length comparison, Wilson CIs.
  Reads CLI-produced `benchmark_results.jsonl` artifacts via `GET /v1/robustness/results` and
  `GET /v1/robustness/comparison`. Set `PROVENANCE_ROBUSTNESS_DIR` to the directory tree
  containing benchmark outputs (default `data/robustness`). All statistics are computed
  server-side by the Phase 5 reporting layer; the dashboard only displays them.

### Authentication

The dashboard stores your API base URL and key in `sessionStorage`.
- Keys are sent only to the configured API server via `X-API-Key` header
- Keys are never logged or stored in localStorage
- Disconnect clears the session

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

## Robustness Benchmark

Evaluate detector resilience against text transformations:

```bash
python -m provenance benchmark robustness \
  --text fixtures/unicode.txt \
  --detector unicode \
  --json
```

For watermark detectors:

```bash
python -m provenance benchmark robustness \
  --text fixtures/watermarked.txt \
  --detector kgw \
  --config configs/kgw.example.json
```

Available transformations: identity, whitespace_normalization,
unicode_normalization_nfc, unicode_normalization_nfd, lowercase, uppercase,
punctuation_normalization, strip_blank_lines,
add_leading_trailing_whitespace, double_spaces.

Select specific transforms:

```bash
python -m provenance benchmark robustness \
  --text fixtures/unicode.txt \
  --detector unicode \
  --transforms "lowercase,whitespace_normalization"
```

The output includes detection rate before/after each transformation andscore changes. Raw text is never persisted — only SHA-256 hashes.

### Watermark Robustness (KGW / SynthID)

Generate watermarked samples and evaluate detection robustness against
text transformations:

```bash
# KGW watermark robustness
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw-reference \
  --lengths 50,100 \
  --samples 3

# SynthID watermark robustness
python -m provenance benchmark robustness \
  --config configs/synthid.hf.example.json \
  --detector synthid-reference \
  --lengths 50,100 \
  --samples 3
```

Pipeline: generate watermarked text → detect original → apply transformation →
detect transformed → compare scores and detection results.

Report includes:
- Baseline detection rate (identity transform)
- Transformed detection rate
- Robustness rate (fraction of baseline-detected samples still detected after transformation)
- Mean baseline/transformed scores
- Mean score delta per transformation

**Important:** This evaluates robustness of the implemented detector under
the tested transformations. It does not establish resistance to adversarial
attacks, watermark removal, paraphrasing, or generic AI-text detection.

### Benchmark Reporting (Phase 5C)

Save benchmark results and generate aggregated comparison reports:

```bash
# Save results to disk
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw-reference \
  --lengths 50,100 \
  --samples 3 \
  --out-dir data/robustness/kgw

# Aggregate and compare saved results
python -m provenance benchmark robustness-report \
  --input data/robustness/ \
  --json
```

The report includes:
- Versioned machine-readable result schema
- Cross-experiment aggregation by detector, config, transform, length
- 95% Wilson confidence intervals for detection/robustness rates
- Robustness matrix (detectors × transforms)
- Per-transform and per-length breakdowns
- JSON output for programmatic consumption

**Limitations:** These measurements describe robustness against tested
deterministic transformations only. A high robustness score does not mean
the watermark is resistant to removal.

### Advanced Robustness Evaluation (Phase 5D)

Extended transformations organized by category with predefined profiles:

```bash
# Use a specific profile
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw-reference \
  --profile unicode \
  --lengths 50,100 \
  --samples 3

# Run all safe transformations
python -m provenance benchmark robustness \
  --config configs/kgw.hf.example.json \
  --detector kgw-reference \
  --profile all_safe \
  --lengths 50,100 \
  --samples 5 \
  --out-dir data/robustness/kgw
```

Available profiles: `formatting`, `unicode`, `whitespace`, `casing`,
`punctuation`, `lexical`, `tokenization-sensitive`, `all_safe`

Transformation categories:
- **formatting**: paragraph reflow, line wrapping, blank-line normalization
- **whitespace**: collapse, expansion, tab/space normalization
- **unicode**: NFC, NFD, NFKD normalization, punctuation normalization
- **casing**: lowercase, uppercase, title case
- **punctuation**: normalization, repeated punctuation
- **lexical**: conservative synonym substitution, contraction expansion
- **tokenization-sensitive**: formatting boundary insertion/removal

All transformations are deterministic and reproducible from a seed.
No external LLMs or online APIs are used.

**Important:** This evaluates robustness of implemented detectors under
controlled deterministic transformations. It does NOT implement adversarial
attacks, watermark removal, gradient-based evasion, or automated paraphrasing.
The purpose is measurement, not evasion.

### Benchmark Orchestration (Phase 5E)

Run multi-model benchmark plans with cross-model comparison:

```bash
# Create a benchmark plan (benchmark_plan.json)
# Then run it:
python -m provenance benchmark plan \
  --plan benchmark_plan.json \
  --out-dir data/benchmarks/run-001

# Dry run — see what would be executed
python -m provenance benchmark plan \
  --plan benchmark_plan.json --dry-run

# Resume interrupted run
python -m provenance benchmark plan \
  --plan benchmark_plan.json --out-dir data/benchmarks/run-001 --resume

# Force re-run of completed experiments
python -m provenance benchmark plan \
  --plan benchmark_plan.json --out-dir data/benchmarks/run-001 --resume --force
```

Example `benchmark_plan.json`:

```json
{
  "schema_version": "provenance-benchmark-plan-v1",
  "name": "watermark-robustness-v1",
  "specs": [
    {
      "detector": "kgw-reference",
      "config": "configs/kgw.hf.model_a.json",
      "lengths": [50, 100],
      "samples": 10,
      "seed": 42,
      "profile": "all_safe"
    },
    {
      "detector": "kgw-reference",
      "config": "configs/kgw.hf.model_b.json",
      "lengths": [50, 100],
      "samples": 10,
      "seed": 42,
      "profile": "all_safe"
    }
  ],
  "output_dir": "data/benchmarks/v1"
}
```

Features:
- Deterministic experiment IDs (SHA-256 of normalized spec)
- Run manifests with status tracking (including the plan ID, so resuming
  with a different plan fails fast instead of silently mixing runs)
- Resume/force support for interrupted runs (resume retries pending and
  failed experiments; completed ones are skipped unless --force is given)
- Failure isolation (one experiment failure doesn't stop others)
- Cross-model comparison reports with Wilson CIs
- No raw text or secrets stored in manifests

Each experiment runs the real pipeline (generate watermarked samples,
evaluate across transforms, write `results.jsonl` into
`<out-dir>/<experiment_id>/`). Watermark specs require an HF experiment
config and download the model on first use. The `unicode` detector cannot
generate samples, so `unicode` specs fail with guidance to use the
text-file robustness mode instead.

**Important:** This is experiment orchestration for measurement.
It does NOT implement adversarial optimization or watermark removal.

### Model Loading Cache (Phase 6B)

Watermark benchmarks reuse Hugging Face models/tokenizers through a bounded
process-global cache (`src/provenance/loading.py`):

- The first benchmark with a given configuration loads the model (seconds to
  minutes) and tokenizer; repeated runs, samples, and plan specs sharing the
  configuration reuse them with identical outputs.
- Cache identity covers every load-relevant field (model id, revision,
  device, dtype, download flags; full tokenizer factory dict), so one
  config can never contaminate another.
- Capacity is `PROVENANCE_MODEL_CACHE_SIZE` entries per kind (default 2,
  LRU eviction; `0` disables caching). Cache hits, misses, load durations,
  and evictions are logged via the `provenance.loading` logger.
- Model inference is serialized per cached instance; concurrent benchmark
  threads share safely at the cost of serialized forward passes.
- Cached references are released on API shutdown. Long-lived processes
  holding many distinct configs should keep the capacity small — each
  retained model costs roughly its weight size in RAM (e.g. hundreds of MB
  for small GPT-2-class models).
- Cheap `simple-vocabulary` tokenizers are never cached (built fresh).

### When Hugging Face Downloads Occur

- Simulation configs (`configs/kgw.example.json`, `configs/synthid.example.json`)
  need no downloads: detection and robustness text-file mode run fully offline.
- HF experiment configs (`configs/*.hf.example.json`) trigger a model and
  tokenizer download on first use (cached under `~/.cache/huggingface` by
  `transformers`), including `benchmark robustness` watermark mode,
  `benchmark kgw/synthid`, and `benchmark plan` with watermark specs.
- Set `HF_HUB_OFFLINE=1` to force cache-only operation (fails fast instead
  of downloading), or `local_files_only: true` in the config.

### Dashboard-Triggered Benchmark Runs (Phase 6C)

Start a run from the dashboard **Benchmarks** page (New benchmark run) or
directly:

```bash
curl -X POST http://localhost:8000/v1/benchmark-runs \
  -H "X-API-Key: $PROVENANCE_API_KEY" -H "Content-Type: application/json" \
  -d '{"detector": "kgw-reference", "config": "configs/kgw.hf.example.json",
       "profile": "all_safe", "lengths": [50, 100], "samples": 5, "seed": 42}'
```

Lifecycle: `queued → running → completed` or `failed`; `cancelled` via
`POST /v1/benchmark-runs/{id}/cancel` (queued runs stop immediately,
running runs finish the current step first); `failed`/`cancelled` runs can
be retried in place with `POST .../retry` (same run ID, artifacts resume
via the run manifest). Partial experiment failure marks the run `failed` —
never success-looking. Progress (`experiments_total/completed/failed`) is
derived from completed work only.

- Concurrency: at most `PROVENANCE_MAX_BENCHMARK_RUNS` runs execute at
  once (default 1); extras wait in the background queue.
- Results persist under `PROVENANCE_ROBUSTNESS_DIR/runs/<run_id>/` as
  standard Phase 5 artifacts, stamped with the run ID. Resolve them with
  `GET /v1/robustness/results?run_id=<run_id>` or the dashboard
  Robustness view ("View results in Robustness" on a completed run).
- Restart recovery: runs interrupted by an API restart are marked `failed`
  (never successful); artifacts are preserved and retry resumes safely.
- Privacy: run records hold configuration, progress, and aggregate
  summaries only — no raw text, watermark keys, or credentials. Config
  paths must be existing `.json` files (≤1MB); run output directories are
  server-chosen, never user-controlled.

Known limitations: cancellation is cooperative (long generation steps run
to a checkpoint first); only the reference detectors (`kgw-reference`,
`synthid-reference`) can complete generation runs — `unicode` and the
simulation detectors (`kgw`, `synthid`) are rejected with guidance;
`transforms` takes precedence when both `profile` and `transforms` are
given.


## Public Product (Phase 7A)

The consumer SaaS shell around the engine: a public website with anonymous
analysis (2 free/UTC-day, cookie-metered), Supabase Auth sign-in, and the
existing dashboard as the entitlement-gated workspace (`#/app`).

```bash
# Public analysis (no API key; sets the signed pv_visitor cookie)
curl -c cookies -X POST http://localhost:8000/v1/public/analyze \
  -H 'Content-Type: application/json' -d '{"text": "Hello, world!"}'

# Quota status + identity/entitlements
curl -b cookies http://localhost:8000/v1/public/quota
curl -b cookies http://localhost:8000/v1/me
```

- Anonymous identity is a server-signed `HttpOnly; SameSite=Lax` cookie
  (`PROVENANCE_ANON_COOKIE_SECRET`); no fingerprinting, no raw IPs stored.
- Quota is enforced atomically in the DB and consumed only by successful
  analyses; the UTC-day boundary is documented in `docs/architecture.md`.
- With Supabase configured, `/v1/*` also accepts a Supabase bearer token
  (anonymous access is then denied); without it, legacy API-key behavior
  is unchanged. `POST /v1/auth/sync` provisions the user row and migrates
  anonymous usage without resetting quota.
- Frontend: without `VITE_SUPABASE_URL` the dashboard is the legacy
  developer console; with it, the public site is the landing page and the
  workspace requires sign-in. Copy `dashboard/.env.example` to
  `dashboard/.env` to configure.
- Manual credentials needed: a Supabase project (URL = public,
  anon key = public/frontend, JWT secret = backend-only secret).
  Payments are not implemented; Pro is granted by setting `plan='pro'`
  on the `app_users` row until Phase 7C.

### Public analysis capability (Phase 7B.1)

`POST /v1/public/analyze` runs only the cheap deterministic Unicode artifact
detector on arbitrary pasted text. Its structured result separates checked and
detected signals from detectors that were not run. It does not return raw
detector evidence, scores, thresholds, keys, configuration paths, or debug data.

| Detector | Public classification | Public execution | Compute |
|---|---|---|---|
| `unicode` | Publicly usable on arbitrary input | Run | Cheap/deterministic |
| `kgw` | Benchmark-only controlled simulation | Not applicable | Cheap but configured |
| `synthid` | Benchmark-only controlled simulation | Not applicable | Cheap but configured |
| `kgw-reference` | Reference/config-specific | Unavailable without matching key/config | Potentially model-backed |
| `synthid-reference` | Reference/config-specific | Unavailable without matching key/config | Potentially model-backed |

A clean public result means only that the Unicode scan found no supported
artifact. It does not establish human authorship or the absence of KGW,
SynthID, vendor-specific, or unknown watermarks. Authenticated users can enter
the existing full analysis workflow; configured reference detection and
benchmarks remain deployment- and entitlement-dependent.

## HTTP API (Phase 2)

The engine is exposed through a local REST API with pluggable persistence
(SQLite or PostgreSQL) and optional API-key authentication.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | SQLite at `data/provenance.db` |
| `PROVENANCE_API_KEY` | Legacy API key for `/v1/*` auth | unset (auth disabled) |
| `RATE_LIMIT_MAX_REQUESTS` | Max requests per window for `/v1/*` | 60 |
| `RATE_LIMIT_WINDOW_SECONDS` | Rate limit window in seconds | 60 |
| `MAX_LIST_LIMIT` | Max `limit` param for `GET /v1/analyses` | 100 |
| `CORS_ORIGINS` | Comma-separated allowed origins | unset (restrictive) |
| `PROVENANCE_DAILY_REQUEST_LIMIT` | Max requests per key per day | unset (unlimited) |
| `PROVENANCE_DAILY_CHARACTER_LIMIT` | Max characters per key per day | unset (unlimited) |
| `PROVENANCE_ADMIN_API_KEY` | Admin key for key management endpoints | unset (disabled) |
| `PROVENANCE_MAX_BACKGROUND_JOBS` | Max concurrent background workers | 2 |
| `PROVENANCE_JOB_RETENTION_HOURS` | Hours to keep completed/failed jobs | 24 |
| `PROVENANCE_ROBUSTNESS_DIR` | Directory tree scanned for benchmark `*.jsonl` artifacts | `data/robustness` |
| `PROVENANCE_MODEL_CACHE_SIZE` | Cached models/tokenizers retained per kind (0 disables) | 2 |
| `PROVENANCE_MAX_BENCHMARK_RUNS` | Simultaneous dashboard-triggered benchmark executions | 1 |

Copy `.env.example` to `.env` and fill in values for production.
Never commit `.env` with real credentials.

### Local Development

```bash
pip install -e '.[dev,api]'
cp .env.example .env  # then fill in local values
python -m uvicorn --env-file .env --factory provenance.api.app:create_app --host 0.0.0.0 --port 8000
```

### Production (PostgreSQL)

```bash
pip install -e '.[api,api-pg]'
export DATABASE_URL='postgresql://user:password@localhost:5432/provenance'
export PROVENANCE_API_KEY='your-production-key'
python -m uvicorn --factory provenance.api.app:create_app --host 0.0.0.0 --port 8000
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

### API Keys (Phase 3A/3B)

API keys are stored as SHA-256 hashes in the database. Raw secrets are never
persisted or logged. The legacy `PROVENANCE_API_KEY` environment variable is
still supported for backwards compatibility.

**Creating keys** (requires `PROVENANCE_ADMIN_API_KEY`):

```bash
# Set the admin key
export PROVENANCE_ADMIN_API_KEY='your-admin-key'

# Create a new API key
curl -X POST http://localhost:8000/v1/api-keys \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-admin-key' \
  -d '{"name": "My App Key"}'
# Returns: {"key_id": "...", "key": "<raw-secret-shown-once>", ...}

# List all keys (admin only)
curl http://localhost:8000/v1/api-keys -H 'X-API-Key: your-admin-key'

# Revoke a key
curl -X DELETE http://localhost:8000/v1/api-keys/{key_id} \
  -H 'X-API-Key: your-admin-key'
```

**Important**: The raw API key is returned **only once** at creation time.
It cannot be recovered afterward.

| Key source | Behavior |
|------------|----------|
| stored key (active) | Requests accepted |
| stored key (revoked) | Requests rejected (403) |
| `PROVENANCE_API_KEY` env var | Accepted as legacy fallback |
| `PROVENANCE_ADMIN_API_KEY` | Admin access + normal endpoint access |

### API Key Lifecycle

1. Admin creates a key via `POST /v1/api-keys` → receives raw secret once
2. User stores the secret securely (e.g., environment variable)
3. User authenticates requests with `X-API-Key: <secret>`
4. Admin can revoke via `DELETE /v1/api-keys/{key_id}` → key is deactivated
5. Revoked key records remain for audit/usage history

### Usage Tracking (Phase 3A)

Every `/v1/*` request is recorded with: key ID, endpoint, timestamp, status
code, duration, character count, and detector used.

`GET /v1/usage` returns aggregate statistics for the calling key:

```bash
curl http://localhost:8000/v1/usage -H 'X-API-Key: your-key'
# {"total_requests": 42, "successful_requests": 40, "failed_requests": 2,
#  "total_characters": 12345, "by_endpoint": {"/v1/analyze": 40},
#  "by_detector": {"unicode": 38}}
```

### Usage Limits (Phase 3A)

Set daily limits per key:

```bash
export PROVENANCE_DAILY_REQUEST_LIMIT=1000
export PROVENANCE_DAILY_CHARACTER_LIMIT=1000000
```

When a limit is exceeded, requests return HTTP 429 with a descriptive error.
These limits are separate from the per-IP rate limiter.

### Endpoints

| Method | Path | Auth | Rate-limited | Description |
|--------|------|------|-------------|-------------|
| `GET` | `/health` | No | No | Service health check |
| `GET` | `/ready` | No | No | Readiness probe (checks DB + executor) |
| `GET` | `/metrics` | No | No | Application metrics (JSON) |
| `POST` | `/v1/analyze` | Yes | Yes | Analyze text with selected detectors |
| `GET` | `/v1/analyses/{id}` | Yes | Yes | Retrieve a persisted analysis |
| `GET` | `/v1/analyses` | Yes | Yes | List analyses (paginated) |
| `GET` | `/v1/usage` | Yes | Yes | Usage statistics for the calling key |
| `POST` | `/v1/api-keys` | Admin | Yes | Create a new API key |
| `GET` | `/v1/api-keys` | Admin | Yes | List all API keys |
| `DELETE` | `/v1/api-keys/{id}` | Admin | Yes | Revoke an API key |
| `POST` | `/v1/analyze/async` | Yes | Yes | Submit text for background analysis |
| `GET` | `/v1/jobs/{job_id}` | Yes | Yes | Get job status and result |
| `GET` | `/v1/jobs` | Yes | Yes | List jobs for the calling key |
| `DELETE` | `/v1/jobs/{job_id}` | Yes | Yes | Cancel a queued/running job |
| `POST` | `/v1/jobs/{job_id}/retry` | Yes | Yes | Retry a failed job (creates new job) |

### Response Metadata (Phase 3A)

`POST /v1/analyze` includes a `duration_ms` field with the processing time.

### Async Analysis (Phase 3C)

For expensive analysis, use the async endpoint:

```bash
curl -X POST http://localhost:8000/v1/analyze/async \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text": "Long text to analyze...", "detectors": ["unicode"]}'
# HTTP 202
# {"job_id": "...", "status": "queued", "message": "Analysis job submitted"}

# Check job status
curl http://localhost:8000/v1/jobs/{job_id} -H 'X-API-Key: your-key'
# {"job_id": "...", "status": "completed", "result": {...}, ...}

# List your jobs
curl http://localhost:8000/v1/jobs -H 'X-API-Key: your-key'
```

**Important**: The async endpoint is single-process only. It is NOT a
distributed job queue. Background workers are bounded by
`PROVENANCE_MAX_BACKGROUND_JOBS` (default 2). Completed jobs are retained
for `PROVENANCE_JOB_RETENTION_HOURS` (default 24) and cleaned up
opportunistically on startup.

Both sync (`/v1/analyze`) and async (`/v1/analyze/async`) share the same
`AnalysisService` — identical analysis logic, different execution model.

### Job Cancellation (Phase 3D)

Cancel a queued or running job:

```bash
# Cancel a queued job (will not execute)
curl -X DELETE http://localhost:8000/v1/jobs/{job_id} -H 'X-API-Key: your-key'
# {"job_id": "...", "status": "cancelled", "message": "Queued job cancelled"}

# Cancel a running job (best-effort, checked after analysis)
curl -X DELETE http://localhost:8000/v1/jobs/{job_id} -H 'X-API-Key: your-key'
# {"job_id": "...", "status": "cancellation_requested", ...}
```

- Only the owning API key (or admin) can cancel a job.
- Completed/failed jobs cannot be cancelled (returns HTTP 409).
- Already-cancelled jobs return HTTP 200 (idempotent).
- For running jobs, cancellation is best-effort: the worker checks after
  analysis completes but cannot forcibly kill Python threads.

### Job Retry (Phase 3D)

Retry a failed job by re-submitting the text:

```bash
curl -X POST http://localhost:8000/v1/jobs/{job_id}/retry \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"text": "The same text to re-analyze...", "detectors": ["unicode"]}'
# HTTP 202
# {"job_id": "<new-job-id>", "status": "queued", "message": "Retry submitted (attempt 1)"}
```

- Only failed jobs can be retried (returns HTTP 409 otherwise).
- Creates a **new** job linked to the original via `parent_job_id`.
- `retry_count` increments on each retry (supports chains).
- Normal rate limits, daily limits, and concurrency limits apply.
- The raw text is never stored — you must re-submit it.

### Job State Model

```
queued → running → completed
queued → cancelled
running → failed
running → cancellation_requested → cancelled
failed → retry creates NEW queued job
```

### Restart Recovery

On application startup, jobs stuck in `running` or `queued` from a
previous process are automatically marked as `failed` with the error
message "Job interrupted by application restart". This prevents jobs
from being permanently stuck if the process crashes.

### Metrics & Observability (Phase 3E)

`GET /metrics` returns machine-readable application metrics:

```bash
curl http://localhost:8000/metrics
# {"engine_version": "0.1.0", "total_requests": 42, "successful_requests": 40,
#  "failed_requests": 2, "total_characters": 12345, "avg_duration_ms": 150.5,
#  "by_endpoint": {"/v1/analyze": 40}, "by_detector": {"unicode": 38},
#  "by_status_code": {"200": 40, "429": 2},
#  "jobs_queued": 0, "jobs_running": 1, "jobs_completed": 5,
#  "jobs_failed": 1, "jobs_cancelled": 0,
#  "configured_worker_count": 2}
```

`GET /ready` reports operational status:

```bash
curl http://localhost:8000/ready
# {"status": "ready", "engine_version": "0.1.0",
#  "persistence": "ok", "executor": "ok"}
```

### Configuration Validation (Phase 3E)

Operational environment variables are validated at startup:

- `PROVENANCE_MAX_BACKGROUND_JOBS` — must be >= 1 (default 2)
- `PROVENANCE_JOB_RETENTION_HOURS` — must be >= 1 (default 24)
- `RATE_LIMIT_MAX_REQUESTS` — must be >= 1 (default 60)
- `RATE_LIMIT_WINDOW_SECONDS` — must be >= 1 (default 60)
- `MAX_LIST_LIMIT` — must be >= 1 (default 100)
- `PROVENANCE_DAILY_REQUEST_LIMIT` — must be >= 1 if set (default unlimited)
- `PROVENANCE_DAILY_CHARACTER_LIMIT` — must be >= 1 if set (default unlimited)

Invalid values log a warning at startup but do not crash the application.

### Graceful Shutdown (Phase 3E)

On shutdown:
1. New background jobs are no longer accepted
2. Currently running jobs are allowed to finish
3. Final job state is persisted to the database
4. The executor is shut down cleanly

### Input limits

- Maximum text length: 200,000 characters.
- Detector names must be from: `unicode`, `kgw`, `kgw-reference`, `synthid`, `synthid-reference`.
- Watermark detectors (`kgw`, `kgw-reference`, `synthid`, `synthid-reference`) require `config_path`.

### Persistence

Analysis results are stored in a database selected via the `DATABASE_URL` environment variable.

| `DATABASE_URL` | Backend | Use case |
|----------------|---------|----------|
| unset (default) | SQLite at `data/provenance.db` | Local development |
| `sqlite:///relative/path` | SQLite at repo-relative path | Explicit SQLite |
| `sqlite:////absolute/path` | SQLite at absolute path | Explicit SQLite |
| `postgresql://…` | PostgreSQL | Production |

The schema is initialized and migrated automatically whenever a repository is
created. PostgreSQL bootstrapping is a single transaction using the existing
`CREATE ... IF NOT EXISTS` definitions and idempotent `ADD COLUMN IF NOT
EXISTS` migrations; it never drops tables or data. PostgreSQL support requires
`psycopg2-binary` (`pip install -e '.[api,api-pg]'`).

There is no separate migration framework or version table in this project.
Normal API startup is the authoritative bootstrap command:

```bash
python -m uvicorn --env-file .env --factory provenance.api.app:create_app \
  --host 0.0.0.0 --port 8000
```

For a one-shot bootstrap without keeping the API process running:

```bash
dotenv -f .env run -- python -c \
  "from provenance.api.db import create_repository; r=create_repository(); r.close()"
```

The required application tables are `analyses`, `api_keys`, `usage_records`,
`jobs`, `benchmark_runs`, `app_users`, `anonymous_visitors`, and
`usage_events`. No subscription/payment tables exist yet.

### Production Deployment

**Environment variables** (required in production):

```bash
export DATABASE_URL='postgresql://user:password@db:5432/provenance'
export PROVENANCE_API_KEY='your-production-key'
export PROVENANCE_ADMIN_API_KEY='your-admin-key'
export CORS_ORIGINS='https://your-dashboard-domain.com'
export RATE_LIMIT_MAX_REQUESTS=60
export RATE_LIMIT_WINDOW_SECONDS=60
export PROVENANCE_DAILY_REQUEST_LIMIT=1000
export PROVENANCE_DAILY_CHARACTER_LIMIT=1000000
export PROVENANCE_MAX_BACKGROUND_JOBS=2
export PROVENANCE_JOB_RETENTION_HOURS=24
```

**PostgreSQL / fresh Supabase setup**:

```bash
# Use Supabase's session-pooler URI for an IPv4-only deployment.
# Put DATABASE_URL in .env, then run the one-shot command above or start API.

# Verify the resulting schema and required SaaS table:
dotenv -f .env run -- zsh -c 'psql "$DATABASE_URL" -c "\\dt"'
dotenv -f .env run -- zsh -c \
  'psql "$DATABASE_URL" -c "select count(*) from app_users"'
```

Supabase may automatically enable RLS on SQL-created tables. The application
does not create permissive policies and does not use the Supabase Data API or
frontend database access. FastAPI connects through `DATABASE_URL` as the
backend database owner; on the current Supabase session-pooler setup that role
owns these tables and has `BYPASSRLS`. If a restricted database role is used
instead, grant only the required backend privileges or define restrictive
server-side policies before switching credentials.

Bootstrap failure logs contain only the failed stage, exception type, and
PostgreSQL SQLSTATE—never the DSN or credentials. Because DDL is transactional,
a failed bootstrap rolls back as a unit. Before future non-additive migrations,
take a Supabase backup or `pg_dump`; rolling application code back does not
remove tables/columns created by this additive bootstrap.

**Docker Compose (local production-like testing)**:

```bash
docker-compose up -d
# API at http://localhost:8000
# Dashboard at http://localhost:5173 (after npm run dev in dashboard/)
```

**Dashboard for production**:

```bash
cd dashboard && npm run build
# Serve dist/ with nginx or any static file server
# Configure API base URL in the dashboard UI
```

**CORS configuration**:

Set `CORS_ORIGINS` to the dashboard origin (e.g., `https://dashboard.example.com`).
The dashboard uses `fetch()` which requires CORS for cross-origin requests.
`DELETE` is included in allowed methods (required for job cancel/revoke).

**Health/readiness checks**:

```bash
curl http://localhost:8000/health  # Lightweight, always 200
curl http://localhost:8000/ready   # Checks DB + executor
curl http://localhost:8000/metrics # Machine-readable metrics
```

### Security & Logging Guarantees

- **Raw input text is never stored** (only its SHA-256 hash).
- **Raw watermark keys** are never exposed in API responses or database records.
- **Raw API keys are never persisted or logged.** The raw secret is returned
  only once at creation time and cannot be recovered.
- Database passwords are never printed or included in connection strings.
- Only identifiers like `hash_key_id` appear in results.
- Structured logs include method, path, status, request ID, and duration only.
- Error responses return clean JSON without stack traces or internal details.
- CORS is disabled by default (no cross-origin requests allowed).
- API key management endpoints require a separate admin key.
- Revoked keys remain in the database for audit/usage history.

## Tests

```bash
pytest
```

Live PostgreSQL integration tests (when PostgreSQL is available):

```bash
export PROVENANCE_TEST_PG_DSN='postgresql://user@/dbname?options=-c search_path%3Dtest_schema'
pytest tests/test_api.py -k pg_live -v
```
