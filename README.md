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
```

`kgw` and `synthid` are controlled simulations and their JSON output includes
`implementation_kind: "simulation"` and `compatibility: "controlled-local-only"`.
Use `kgw-reference` for the model-token KGW research detector.

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

The current KGW reference variant does not claim byte-for-byte compatibility
with every KGW implementation because it uses a pure-Python deterministic
permutation rather than PyTorch `randperm`.

## Tests

```bash
pytest
```
