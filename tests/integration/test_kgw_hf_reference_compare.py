"""Reference-comparison test over genuine model/tokenizer-backed token IDs.

This does NOT require torch/transformers: it reads the token IDs recorded when
the model-backed samples were generated and re-scores them two independent ways.

Level B of the correctness suite: for identical token IDs and identical KGW
configuration, our scorer must agree with an independent reimplementation of the
same `kgw-python-left-v1` variant -- exact match, no tolerances.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from provenance.detectors.reference.kgw import KGWReferenceConfig, KGWReferenceScorer

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "kgw.hf.example.json"
REFERENCE_DIR = ROOT / "data" / "generated" / "kgw" / "reference"


# --- independent reimplementation of kgw-python-left-v1 (no shared code) ------

def _permutation(seed, vocab_size):
    values = list(range(vocab_size))
    random.Random(int(seed)).shuffle(values)
    return values


def _f(config, prf, vocab_size, prefix):
    if config.f_scheme == "additive":
        return prf[sum(prefix) % vocab_size]
    if config.f_scheme == "time":
        value = 1
        for token_id in prefix:
            value *= token_id
        return prf[value % vocab_size]
    if config.f_scheme == "skip":
        return prf[prefix[0] % vocab_size]
    return min(prf[token_id % vocab_size] for token_id in prefix)


def _independent_score(config, vocab_size, token_ids):
    prf = _permutation(config.hash_key, vocab_size)
    green_size = int(vocab_size * config.gamma)
    green = 0
    scored = 0
    flags = []
    seen = set()
    for index in range(config.prefix_length, len(token_ids)):
        context = token_ids[index - config.prefix_length : index]
        token_id = token_ids[index]
        ngram = tuple(context + [token_id])
        if config.ignore_repeated_ngrams and ngram in seen:
            flags.append(None)
            continue
        seen.add(ngram)
        prefix = context[-config.prefix_length :]
        seed = (config.hash_key * _f(config, prf, vocab_size, prefix)) % vocab_size
        greenlist = set(_permutation(seed, vocab_size)[:green_size])
        is_green = token_id in greenlist
        flags.append(is_green)
        scored += 1
        green += int(is_green)
    z = (green - config.gamma * scored) / math.sqrt(scored * config.gamma * (1 - config.gamma))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return green, scored, flags, z, p


def _load_config():
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return KGWReferenceConfig.from_dict(data)


@pytest.mark.parametrize("stem", ["watermarked", "unwatermarked"])
def test_our_scoring_matches_independent_reference_on_model_token_ids(stem):
    metadata_path = REFERENCE_DIR / f"{stem}.metadata.json"
    if not metadata_path.exists():
        pytest.skip(
            "model-backed KGW fixtures missing; run "
            "scripts/generate_kgw_hf_samples.py --config configs/kgw.hf.example.json"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_ids = metadata["generated_token_ids"]
    vocab_size = metadata["vocab_size"]

    config = _load_config()
    scorer = KGWReferenceScorer(config, vocab_size=vocab_size)
    ours = scorer.score_token_ids(token_ids)

    exp_green, exp_scored, exp_flags, exp_z, exp_p = _independent_score(
        config, vocab_size, token_ids
    )

    assert ours.green_token_mask == exp_flags          # green mask: MATCH
    assert ours.green_token_count == exp_green          # green count: MATCH
    assert ours.scored_token_count == exp_scored
    assert ours.z_score == exp_z                        # z-score: MATCH (exact)
    assert ours.p_value == exp_p                        # p-value: MATCH (exact)
