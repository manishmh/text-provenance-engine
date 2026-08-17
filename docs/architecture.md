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
- structured result schemas;
- a local CLI and tests.

It does not implement an HTTP API, database, authentication, benchmarking
service, watermark removal, attack experiments, or generic AI detection.

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

## SynthID Assumptions

SynthID-Text detection requires a known watermark configuration. Phase 1
currently includes only a controlled simulation inspired by the weighted-mean
score shape. Reference-backed SynthID is intentionally not implemented yet. The
project does not assume access to Google's production Gemini keys or production
configuration, and it must not be used as a Gemini attribution detector.

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
