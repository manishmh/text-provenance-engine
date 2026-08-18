# KGW Upstream Cross-Check

This document records an upstream cross-check of our `kgw-python-left-v1`
detector against the genuine published KGW implementation. It closes the gap
identified in the audit: our previous "reference comparison"
(`tests/integration/test_kgw_hf_reference_compare.py`) only compared our scorer
against an independent reimplementation of the **same** pure-Python variant,
which uses the same `random.Random(seed).shuffle` RNG. That proves
self-consistency, not compatibility with upstream KGW.

Reproduce with:

```bash
pip install -e '.[hf]' markllm      # torch + transformers + genuine MarkLLM
python scripts/upstream_kgw_crosscheck.py            # Parts 2 and 3
python scripts/upstream_kgw_crosscheck.py --generate # Part 5 (needs distilgpt2)
```

Upstream code was installed from PyPI (`markllm==0.1.5`) and cross-checked
against the repositories at `github.com/THU-BPM/MarkLLM` and
`github.com/jwkirchenbauer/lm-watermarking`. No upstream code was modified; the
genuine `markllm.watermark.kgw.kgw.KGWUtils` class runs unchanged, fed a minimal
config stub carrying only the attributes it reads.

## 1. Upstream implementation selected — MarkLLM

Two upstream families exist for the KGW watermark:

- **`jwkirchenbauer/lm-watermarking`** — the original reference.
  - `watermark_processor.py` (`simple_1`/`lefthash`): seeds the RNG directly with
    `hash_key * prev_token` (no vocab modulo, no PRF table).
  - `extended_watermark_processor.py` + `alternative_prf_schemes.py`: seeds with
    `prf_lookup[type](input_ids[-context_width:], salt_key=hash_key) % (2**64-1)`.
    Its `additive_prf` multiplies the token sum by the salt key and applies a
    separate `hashint`/`fixed_table` avalanche hash. **No `prf = randperm(hash_key)`
    table, no `sum % vocab` lookup.**
- **`THU-BPM/MarkLLM`** (`watermark/kgw/kgw.py`) — a reimplementation whose
  left-hash scheme is the exact structure our variant reproduces.

Our `kgw-python-left-v1` is a structural port of **MarkLLM's** `KGWUtils`, so
MarkLLM is the correct upstream reference. Every one of the following matches
MarkLLM (see side-by-side below); none match lm-watermarking's PRF construction.

## 1a. Exact variant comparison

| Property | MarkLLM (`left`/`additive`) | Our `kgw-python-left-v1` | Match |
|---|---|---|---|
| Seeding scheme | left-hash: `seed = (hash_key · f(ctx)) % vocab` | identical formula | structure ✔ / value ✘ |
| Hash/key handling | single integer `hash_key` (default `15485863`) | same `hash_key = 15485863` | ✔ |
| Context width | `prefix_length` (=1 here) | `prefix_length` (=1) | ✔ |
| Window scheme | `left` (also `self`) | `left` only | ✔ (subset) |
| PRF table | `prf = torch.randperm(vocab, seed=hash_key)` | `prf = random.Random(hash_key).shuffle(range(vocab))` | **✘ different RNG** |
| f-scheme (`additive`) | `prf[ (Σ last L tokens) % vocab ]` | `prf[ (Σ last L tokens) % vocab ]` | structure ✔ / value ✘ (prf differs) |
| Greenlist generation | `randperm(vocab, seed)[:γ·vocab]` | `shuffle(range(vocab), seed)[:γ·vocab]` | structure ✔ / **members ✘** |
| γ (gamma) | configurable (0.25) | 0.25 | ✔ |
| δ (delta) | configurable (2.0) | 2.0 | ✔ |
| Repeated n-gram handling | none in `score_sequence` (scores every position) | optional `ignore_repeated_ngrams` (off here) | ✔ when off |
| Detection statistic | `z = (g − γT)/√(Tγ(1−γ))` | identical | ✔ |
| p-value | `norm.sf(z)` = `0.5·erfc(z/√2)` | `0.5·erfc(z/√2)` | ✔ |

The math (γ, δ, greenlist size, z-score, p-value) is **identical**. The single
divergence is the permutation source — `torch.randperm` vs Python
`random.shuffle` — which makes the PRF table differ, which makes the per-context
seed differ, which makes the greenlist **members** differ. One root cause (the
RNG), cascading through the whole pipeline.

## 2. Scorer comparison — same recorded token IDs

Scoring the exact IDs in `data/generated/kgw/reference/*.metadata.json` (no
decode/re-encode) with our scorer and with the genuine MarkLLM `KGWUtils`:

**Watermarked sample (200 tokens, 199 scored):**

```
our implementation:  green_count = 116   z_score = 10.845728   p_value = 1.04e-27
upstream (MarkLLM):  green_count =  53   z_score =  0.532055   p_value = 2.97e-01
```

**Unwatermarked sample:**

```
our implementation:  green_count = 49   z_score = -0.122782   p_value = 5.49e-01
upstream (MarkLLM):  green_count = 44   z_score = -0.941327   p_value = 8.27e-01
```

Our detector reports a decisive watermark on our watermarked sample (z=10.8);
**the genuine upstream detector reports z=0.53 — no watermark — on the same
tokens.** Green-mask agreement on the watermarked sample is 46.2%, *below* the
~62.5% expected of two independent greenlists, because tokens we forced into our
green set are in upstream's green set only with probability γ=0.25.

## 3. Direct greenlist comparison (no language model)

Fixed `hash_key=15485863, vocab=50257, γ=0.25, prefix_length=1, f=additive`:

```
context=[13]:    |ours|=12564 |upstream|=12564 intersection=3078 (chance≈3141)  seeds 7094 vs 32567  DIFFER
context=[262]:   |ours|=12564 |upstream|=12564 intersection=3141 (chance≈3141)  seeds 29459 vs 43245 DIFFER
context=[464]:   |ours|=12564 |upstream|=12564 intersection=3121 (chance≈3141)  seeds 25668 vs 333   DIFFER
context=[1000]:  |ours|=12564 |upstream|=12564 intersection=3136 (chance≈3141)  seeds 28499 vs 13905 DIFFER
context=[49087]: |ours|=12564 |upstream|=12564 intersection=3121 (chance≈3141)  seeds 43787 vs 45180 DIFFER
```

Greenlists have identical **size** (12564 = ⌊0.25·50257⌋) but their intersection
equals random chance (~3141) — the two greenlists are **statistically
independent**. Even the intermediate seed integer differs for every context,
because the `prf` table (torch vs Python RNG) differs before the seed is even
computed.

## 4. Generation comparison (Part 5)

Generated 200 tokens with distilgpt2 using MarkLLM's own `KGWLogitsProcessor`
(same γ/δ/hash_key/prefix), then scored both ways:

```
MarkLLM-watermarked text:
  MarkLLM detector : green=114/199  z=10.518  p=3.56e-26
  OUR detector     : green= 39/199  z=-1.760  p=9.61e-01
```

Combined with Part 2, the picture is symmetric:

| Text source | MarkLLM detector | Our detector |
|---|---|---|
| Our watermarked | z = 0.53 (undetected) | z = 10.85 (strong) |
| MarkLLM watermarked | z = 10.52 (strong) | z = −1.76 (undetected) |

Each implementation strongly detects **its own** watermark (z≈10.5–10.8) and is
**blind to the other's** (z≈0.5, −1.8). The watermarking *mechanism* and the
*detection statistic* agree; the *key material* (greenlist assignment) does not.

## 5. Exact discrepancies

1. **PRF permutation RNG.** MarkLLM: `torch.randperm(vocab, generator=Generator().manual_seed(hash_key))`. Ours: `random.Random(hash_key).shuffle(range(vocab))`. Different algorithms → different `prf` table from the same key.
2. **Per-context seed.** Because `f_additive` returns `prf[...]`, the differing `prf` yields a different seed integer `(hash_key·f) % vocab` for every context (confirmed in Part 3).
3. **Greenlist permutation RNG.** The same torch-vs-Python difference applies again when turning the seed into the vocab permutation, so even an identical seed would yield a different greenlist.

There are **no discrepancies** in gamma, delta, greenlist size, context width,
f-scheme formula, window scheme, z-score, or p-value.

## 6. Final compatibility classification

**Reference-aligned but RNG-incompatible** (classification #2).

The algorithm — seeding scheme, f-scheme, greenlist construction, and detection
statistic — is a faithful match to MarkLLM's KGW left-hash variant, and the
detection math is identical. But because the permutation source differs
(`random.shuffle` vs `torch.randperm`), the concrete greenlists are statistically
independent and the two implementations **cannot detect each other's
watermarks**. It is *not* upstream-compatible and must not be described as
production- or byte-compatible with MarkLLM or lm-watermarking.

## 7. Scientific readiness

`kgw-python-left-v1` **is** scientifically usable as a *self-consistent* research
detector: for text generated with this exact variant it produces correct,
theoretically-grounded KGW statistics (verified: watermarked z=10.8 vs
unwatermarked z≈0, and it correctly rejects MarkLLM-watermarked text as its own
negative). It is **not** valid for scoring text produced by MarkLLM,
lm-watermarking, or any external KGW deployment — those require their own RNG.
For a research detector claiming upstream interoperability, the permutation would
need to be swapped for `torch.randperm` with MarkLLM's exact seeding.
