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
