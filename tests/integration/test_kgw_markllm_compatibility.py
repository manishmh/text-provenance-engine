"""Compatibility tests verifying kgw-markllm-v1 matches MarkLLM behavior exactly.

Tests require MarkLLM to be installed.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

try:
    import torch
    from markllm.watermark.kgw.kgw import KGWUtils
    from types import SimpleNamespace
    MARKLLM_AVAILABLE = True
except ImportError:
    MARKLLM_AVAILABLE = False


from provenance.detectors.reference.kgw_markllm import KGWMarkLLMConfig, KGWMarkLLMScorer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    return KGWMarkLLMConfig(
        configuration_id="kgw-hf-distilgpt2-v1",
        version="1",
        gamma=0.25,
        delta=2.0,
        hash_key=15485863,
        z_threshold=4.0,
        prefix_length=1,
        f_scheme="additive",
        window_scheme="left",
    )


@pytest.fixture
def markllm_utils(config):
    if not MARKLLM_AVAILABLE:
        pytest.skip("markllm not installed")
    stub = SimpleNamespace(
        device="cpu",
        hash_key=config.hash_key,
        vocab_size=50257,
        gamma=config.gamma,
        delta=config.delta,
        prefix_length=config.prefix_length,
        f_scheme=config.f_scheme,
        window_scheme=config.window_scheme,
    )
    return KGWUtils(stub)


@pytest.fixture
def our_scorer(config):
    return KGWMarkLLMScorer(config, vocab_size=50257)


@pytest.mark.skipif(not MARKLLM_AVAILABLE, reason="markllm not installed")
def test_greenlist_equivalence(our_scorer, markllm_utils):
    """TEST 1: Greenlist equivalence."""
    test_cases = [[13], [262], [464], [1000], [49087]]
    for ctx in test_cases:
        ours = our_scorer.greenlist_ids(ctx)
        ml = [int(x) for x in markllm_utils.get_greenlist_ids(torch.tensor(ctx, dtype=torch.long)).tolist()]
        assert ours == ml


@pytest.mark.skipif(not MARKLLM_AVAILABLE, reason="markllm not installed")
def test_detection_equivalence(our_scorer, markllm_utils):
    """TEST 2: Detection equivalence."""
    # Dummy tokens
    token_ids = [464, 13, 262, 1000, 49087, 13, 262, 1000, 49087, 464, 13, 262, 1000, 49087] * 2
    
    ours = our_scorer.score_token_ids(token_ids)
    
    ml_z, ml_flags = markllm_utils.score_sequence(torch.tensor(token_ids, dtype=torch.long))
    ml_scored_flags = ml_flags[1:] # prefix=1
    ml_green = sum(1 for f in ml_scored_flags if f == 1)
    ml_scored = len(ml_scored_flags)
    ml_p = 0.5 * math.erfc(ml_z / math.sqrt(2))
    
    assert ours.green_token_count == ml_green
    assert ours.scored_token_count == ml_scored
    assert math.isclose(ours.z_score, ml_z, rel_tol=1e-5)
    assert math.isclose(ours.p_value, ml_p, rel_tol=1e-5)
