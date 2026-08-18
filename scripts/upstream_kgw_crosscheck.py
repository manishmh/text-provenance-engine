#!/usr/bin/env python3
"""Upstream KGW cross-check: kgw-python-left-v1 vs genuine MarkLLM KGWUtils.

Purpose
-------
Our existing reference-comparison test (tests/integration/test_kgw_hf_reference_compare.py)
re-scores our recorded token IDs with an *independent Python reimplementation of
the same kgw-python-left-v1 variant*. That reimplementation also uses
``random.Random(seed).shuffle`` -- the same pure-Python RNG as our detector -- so
it proves self-consistency, NOT compatibility with the upstream KGW code.

This script closes that gap. It runs the GENUINE upstream MarkLLM detector logic
(``markllm.watermark.kgw.kgw.KGWUtils``, unmodified, installed from PyPI
``markllm``) against the EXACT token IDs recorded in
``data/generated/kgw/reference/`` and against directly-constructed greenlists,
independent of any language model.

MarkLLM is the correct upstream reference for kgw-python-left-v1: its
``_f_additive`` (``prf[sum(ctx) % vocab]``), left-hash seed
(``(hash_key * f) % vocab``), ``randperm[:greenlist_size]`` selection, and z-score
formula are the exact structures our variant reimplements. The only substitution
is ``torch.randperm`` (MarkLLM) -> ``random.Random.shuffle`` (ours).

Requirements
------------
``pip install markllm torch`` (the same optional stack used by the ``hf`` extra).
The script skips cleanly if they are not installed.

Optionally, with ``--generate`` and distilgpt2 available, it also generates a
small sample with MarkLLM's own logits processor and scores it both ways.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Configuration used to GENERATE data/generated/kgw/reference/ (kgw.hf.example.json).
HASH_KEY = 15485863
VOCAB = 50257
GAMMA = 0.25
DELTA = 2.0
PREFIX = 1
F_SCHEME = "additive"
WINDOW = "left"


def p_value(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2))


def _require_upstream():
    try:
        import torch  # noqa: F401
        from markllm.watermark.kgw.kgw import KGWUtils  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment guard
        print(f"[skip] upstream stack unavailable ({exc}); pip install markllm torch")
        raise SystemExit(0)


def make_ours():
    from provenance.detectors.reference.kgw import KGWReferenceConfig, KGWReferenceScorer

    cfg = KGWReferenceConfig(
        configuration_id="kgw-hf-distilgpt2-v1", version="1", gamma=GAMMA, delta=DELTA,
        hash_key=HASH_KEY, z_threshold=4.0, prefix_length=PREFIX,
        f_scheme=F_SCHEME, window_scheme=WINDOW,
    )
    return KGWReferenceScorer(cfg, vocab_size=VOCAB)


def make_markllm():
    from markllm.watermark.kgw.kgw import KGWUtils

    # Genuine upstream KGWUtils, fed a minimal config stub carrying exactly the
    # attributes the class reads. No upstream code is modified.
    stub = SimpleNamespace(
        device="cpu", hash_key=HASH_KEY, vocab_size=VOCAB, gamma=GAMMA, delta=DELTA,
        prefix_length=PREFIX, f_scheme=F_SCHEME, window_scheme=WINDOW,
    )
    return KGWUtils(stub)


def score_ours(scorer, ids):
    s = scorer.score_token_ids(ids)
    return dict(green=s.green_token_count, scored=s.scored_token_count,
                frac=s.green_fraction, z=s.z_score, p=s.p_value,
                mask=list(s.green_token_mask))


def score_markllm(utils, ids):
    import torch

    z, flags = utils.score_sequence(torch.tensor(ids, dtype=torch.long))
    scored_flags = flags[PREFIX:]
    green = sum(1 for f in scored_flags if f == 1)
    scored = len(scored_flags)
    return dict(green=green, scored=scored,
                frac=green / scored if scored else 0.0, z=z, p=p_value(z),
                mask=[bool(f == 1) for f in scored_flags])


def _print_pair(name, o, m):
    print(f"\n===== SCORER COMPARISON: {name} ({o['scored'] + PREFIX} token IDs) =====")
    for label, r in [("our implementation (kgw-python-left-v1)", o),
                     ("upstream (MarkLLM KGWUtils, left/additive)", m)]:
        print(f"  {label}:")
        print(f"    green_count    = {r['green']}")
        print(f"    scored_tokens  = {r['scored']}")
        print(f"    green_fraction = {r['frac']:.6f}")
        print(f"    z_score        = {r['z']:.6f}")
        print(f"    p_value        = {r['p']:.6e}")
    agree = sum(1 for a, b in zip(o["mask"], m["mask"]) if a == b)
    total = min(len(o["mask"]), len(m["mask"])) or 1
    baseline = 100 * (GAMMA ** 2 + (1 - GAMMA) ** 2)
    print(f"  green-mask agreement: {agree}/{total} ({100*agree/total:.1f}%)  "
          f"[independent-greenlist baseline ~{baseline:.1f}%]")


def compare_recorded(ours_scorer, ml_utils):
    ref_dir = ROOT / "data/generated/kgw/reference"
    for stem in ["watermarked", "unwatermarked"]:
        path = ref_dir / f"{stem}.metadata.json"
        if not path.exists():
            print(f"[skip] {path} missing; run scripts/generate_kgw_hf_samples.py")
            continue
        ids = json.loads(path.read_text())["generated_token_ids"]
        _print_pair(stem, score_ours(ours_scorer, ids), score_markllm(ml_utils, ids))


def compare_greenlists(ours_scorer, ml_utils):
    import torch

    print("\n===== DIRECT GREENLIST COMPARISON (no language model) =====")
    print(f"  fixed: hash_key={HASH_KEY} vocab={VOCAB} gamma={GAMMA} "
          f"prefix_length={PREFIX} f_scheme={F_SCHEME}")
    greenlist_size = int(VOCAB * GAMMA)
    chance = greenlist_size * GAMMA
    for ctx in [[13], [262], [464], [1000], [49087]]:
        ours = set(ours_scorer.greenlist_ids(ctx))
        ml = set(int(x) for x in ml_utils.get_greenlist_ids(
            torch.tensor(ctx, dtype=torch.long)).tolist())
        our_seed = (HASH_KEY * ours_scorer._f(ctx)) % VOCAB
        ml_seed = (HASH_KEY * ml_utils._f(torch.tensor(ctx, dtype=torch.long))) % VOCAB
        print(f"  context={ctx}: |ours|={len(ours)} |upstream|={len(ml)} "
              f"intersection={len(ours & ml)} (chance≈{chance:.0f})  "
              f"seeds ours={our_seed} up={ml_seed} "
              f"{'MATCH' if our_seed == ml_seed else 'DIFFER'}")


def generate_sample(ours_scorer, ml_utils):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
        from markllm.watermark.kgw.kgw import KGWLogitsProcessor
    except Exception as exc:
        print(f"[skip --generate] {exc}")
        return
    print("\n===== PART 5: GENERATION WITH GENUINE MARKLLM PROCESSOR (distilgpt2) =====")
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    stub = SimpleNamespace(device="cpu", hash_key=HASH_KEY, vocab_size=VOCAB, gamma=GAMMA,
                           delta=DELTA, prefix_length=PREFIX, f_scheme=F_SCHEME, window_scheme=WINDOW)
    proc = KGWLogitsProcessor(stub, ml_utils)
    prompt = "The history of lighthouses along the northern coast is a story of"
    ids = tok(prompt, return_tensors="pt").input_ids
    torch.manual_seed(42)
    out = model.generate(ids, logits_processor=LogitsProcessorList([proc]),
                         do_sample=True, top_p=0.95, temperature=1.0,
                         max_new_tokens=200, pad_token_id=tok.eos_token_id)
    gen = out[0][ids.shape[1]:].tolist()
    m = score_markllm(ml_utils, gen)
    o = score_ours(ours_scorer, gen)
    print(f"  MarkLLM-watermarked text, {len(gen)} tokens")
    print(f"    MarkLLM detector : green={m['green']}/{m['scored']} z={m['z']:.3f} p={m['p']:.3e}")
    print(f"    OUR detector     : green={o['green']}/{o['scored']} z={o['z']:.3f} p={o['p']:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true",
                        help="also generate a sample with MarkLLM's processor (needs distilgpt2)")
    args = parser.parse_args()

    _require_upstream()
    ours_scorer = make_ours()
    ml_utils = make_markllm()
    compare_recorded(ours_scorer, ml_utils)
    compare_greenlists(ours_scorer, ml_utils)
    if args.generate:
        generate_sample(ours_scorer, ml_utils)


if __name__ == "__main__":
    main()
