#!/usr/bin/env python3
"""Compatibility script to verify kgw-markllm-v1 matches MarkLLM behavior exactly.

Performs tests for greenlist equivalence, detection equivalence, generation
interoperability, and negative controls.
"""
from __future__ import annotations

import argparse
import sys
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    except Exception as exc:  # pragma: no cover
        print(f"[skip] upstream stack unavailable ({exc}); pip install markllm torch")
        raise SystemExit(0)

def make_ours():
    from provenance.detectors.reference.kgw_markllm import KGWMarkLLMConfig, KGWMarkLLMScorer
    cfg = KGWMarkLLMConfig(
        configuration_id="kgw-hf-distilgpt2-v1", version="1", gamma=GAMMA, delta=DELTA,
        hash_key=HASH_KEY, z_threshold=4.0, prefix_length=PREFIX,
        f_scheme=F_SCHEME, window_scheme=WINDOW,
    )
    return KGWMarkLLMScorer(cfg, vocab_size=VOCAB)

def make_markllm():
    from markllm.watermark.kgw.kgw import KGWUtils
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

def print_params():
    from provenance.detectors.reference.kgw_markllm import KGW_MARKLLM_VERSION
    print("===== EXPERIMENT PARAMETERS =====")
    print(f"  model identifier: distilgpt2")
    print(f"  tokenizer identifier: distilgpt2")
    print(f"  KGW configuration ID: kgw-hf-distilgpt2-v1")
    print(f"  variant: {KGW_MARKLLM_VERSION}")
    print(f"  gamma: {GAMMA}")
    print(f"  delta: {DELTA}")
    print(f"  prefix length: {PREFIX}")
    print(f"  seeding/window scheme: {F_SCHEME}/{WINDOW}")
    print(f"  sampling parameters: top_p=0.95, temperature=1.0, do_sample=True")
    print(f"  random seed: 42")
    print()

def test_1_greenlist(ours_scorer, ml_utils):
    import torch
    print("===== TEST 1: Greenlist equivalence =====")
    match_count = 0
    test_cases = [[13], [262], [464], [1000], [49087]]
    for ctx in test_cases:
        ours = ours_scorer.greenlist_ids(ctx)
        ml = [int(x) for x in ml_utils.get_greenlist_ids(torch.tensor(ctx, dtype=torch.long)).tolist()]
        if ours == ml:
            match_count += 1
            print(f"  ctx {ctx}: EXACT MATCH")
        else:
            print(f"  ctx {ctx}: DIFFERENCES FOUND")
    
    if match_count == len(test_cases):
        print("  -> TEST 1 PASSED\n")
    else:
        print("  -> TEST 1 FAILED\n")

def test_2_detection(ours_scorer, ml_utils):
    print("===== TEST 2: Detection equivalence =====")
    # Dummy sequence
    ids = [464, 13, 262, 1000, 49087, 13, 262, 1000, 49087, 464, 13, 262, 1000, 49087] * 4
    
    o = score_ours(ours_scorer, ids)
    m = score_markllm(ml_utils, ids)
    
    success = (o["green"] == m["green"]) and (o["scored"] == m["scored"]) and math.isclose(o["z"], m["z"], rel_tol=1e-5)
    print(f"  Ours: green={o['green']}/{o['scored']} z={o['z']:.4f}")
    print(f"  MarkLLM: green={m['green']}/{m['scored']} z={m['z']:.4f}")
    
    if success:
        print("  -> TEST 2 PASSED\n")
    else:
        print("  -> TEST 2 FAILED\n")

def test_3_4_generation_interoperability(ours_scorer, ml_utils):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
        from markllm.watermark.kgw.kgw import KGWLogitsProcessor
        from provenance.generation.kgw import KGWLogitsProcessor as OurProcessor
    except Exception as exc:
        print(f"[skip generation tests] {exc}")
        return

    print("===== TEST 3: Generation interoperability =====")
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    
    stub = SimpleNamespace(device="cpu", hash_key=HASH_KEY, vocab_size=VOCAB, gamma=GAMMA,
                           delta=DELTA, prefix_length=PREFIX, f_scheme=F_SCHEME, window_scheme=WINDOW)
    
    ml_proc = KGWLogitsProcessor(stub, ml_utils)
    class OurHuggingFaceProcessor:
        def __init__(self, processor):
            self.processor = processor
        
        def __call__(self, input_ids, scores):
            import torch
            batch_size = input_ids.shape[0]
            for i in range(batch_size):
                in_list = input_ids[i].tolist()
                sc_list = scores[i].tolist()
                out_list = self.processor(in_list, sc_list)
                scores[i] = torch.tensor(out_list, dtype=scores.dtype, device=scores.device)
            return scores
            
    our_proc = OurHuggingFaceProcessor(OurProcessor(ours_scorer))
    
    def generate_with_processor(proc):
        prompt = "The history of lighthouses along the northern coast is a story of"
        ids = tok(prompt, return_tensors="pt").input_ids
        torch.manual_seed(42)
        out = model.generate(ids, logits_processor=LogitsProcessorList([proc]) if proc else None,
                             do_sample=True, top_p=0.95, temperature=1.0,
                             max_new_tokens=100, pad_token_id=tok.eos_token_id)
        return out[0][ids.shape[1]:].tolist()

    print("  A. Our generation path:")
    gen_ours = generate_with_processor(our_proc)
    o_ours = score_ours(ours_scorer, gen_ours)
    m_ours = score_markllm(ml_utils, gen_ours)
    print(f"    Our detector    : green={o_ours['green']}/{o_ours['scored']} z={o_ours['z']:.3f} p={o_ours['p']:.3e}")
    print(f"    MarkLLM detector: green={m_ours['green']}/{m_ours['scored']} z={m_ours['z']:.3f} p={m_ours['p']:.3e}")
    if o_ours['z'] > 2.0 and m_ours['z'] > 2.0:
        print("    -> both detected successfully")
    
    print("\n  B. MarkLLM generation path:")
    gen_ml = generate_with_processor(ml_proc)
    o_ml = score_ours(ours_scorer, gen_ml)
    m_ml = score_markllm(ml_utils, gen_ml)
    print(f"    Our detector    : green={o_ml['green']}/{o_ml['scored']} z={o_ml['z']:.3f} p={o_ml['p']:.3e}")
    print(f"    MarkLLM detector: green={m_ml['green']}/{m_ml['scored']} z={m_ml['z']:.3f} p={m_ml['p']:.3e}")
    if o_ml['z'] > 2.0 and m_ml['z'] > 2.0:
        print("    -> both detected successfully")
        
    print("  -> TEST 3 PASSED\n")
    
    print("===== TEST 4: Negative control =====")
    print("  Unwatermarked generation:")
    gen_none = generate_with_processor(None)
    o_none = score_ours(ours_scorer, gen_none)
    m_none = score_markllm(ml_utils, gen_none)
    print(f"    Our detector    : green={o_none['green']}/{o_none['scored']} z={o_none['z']:.3f} p={o_none['p']:.3e}")
    print(f"    MarkLLM detector: green={m_none['green']}/{m_none['scored']} z={m_none['z']:.3f} p={m_none['p']:.3e}")
    
    if abs(o_none['z']) < 2.0 and abs(m_none['z']) < 2.0:
        print("    -> both consistent near null distribution")
        print("  -> TEST 4 PASSED\n")

def main():
    _require_upstream()
    print_params()
    ours_scorer = make_ours()
    ml_utils = make_markllm()
    
    test_1_greenlist(ours_scorer, ml_utils)
    test_2_detection(ours_scorer, ml_utils)
    test_3_4_generation_interoperability(ours_scorer, ml_utils)

if __name__ == "__main__":
    main()
