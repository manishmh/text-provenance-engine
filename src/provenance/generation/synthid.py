"""Reference-backed SynthID-Text generation pipeline.

Two generation paths:

1. **Synthetic** (``generate_reference_synthid_text``): selects tokens from a
   discrete vocabulary to maximise/minimise the weighted-mean score.  No model
   is involved.  Intended for offline testing and fixture generation.

2. **Model-backed** (``generate_synthid_token_ids``): uses real HuggingFace
   model logits, applying the SynthID logits processor to bias sampling toward
   watermarked tokens.  The detector scores the exact generated token IDs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from provenance.detectors.reference.synthid import (
    SynthIDReferenceConfig,
    SynthIDReferenceDetector,
    _compute_g_values,
    _default_weights,
    _g_value_for_ngram,
)
from provenance.preprocessing.text import stable_hash_int


class LogitModel(Protocol):
    identifier: str
    vocab_size: int

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the exact input token IDs."""


@dataclass(frozen=True)
class SynthIDGenerationConfig:
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = 0.95
    seed: int = 42


class SynthIDLogitsProcessor:
    """Apply SynthID weighted-mean watermark bias to model logits.

    For each candidate token, this processor computes the g-value vector
    for the full n-gram (context + candidate) at each watermarking depth,
    applies the weighted-mean scoring, and adds the score as a logit bias
    to favour watermarked tokens.

    Performance: computing g-values for all ~50k vocab tokens is expensive
    in pure Python.  The processor limits watermarking to the top-k
    candidates (default 400) by score, applying the watermark bias only
    to those tokens.  This matches the reference implementation's approach
    of only watermarking within a sparse top-k window.
    """

    def __init__(self, config: SynthIDReferenceConfig, *, watermark_top_k: int = 200):
        self.config = config
        self._hash_iv = _hash_iv_from_keys(config.keys)
        self._weights = (
            list(config.weights)
            if config.weights is not None
            else _default_weights(len(config.keys))
        )
        self._watermark_top_k = watermark_top_k

    def __call__(self, input_ids: list[int], logits: list[float]) -> list[float]:
        if len(input_ids) < self.config.ngram_len - 1:
            return logits

        context = tuple(input_ids[-(self.config.ngram_len - 1) :])
        depth = len(self.config.keys)

        # Find top-k candidates to watermark (sparse top-k approach).
        top_k = min(self._watermark_top_k, len(logits))
        indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:top_k]

        result = list(logits)
        for token_id in indices:
            ngram = context + (token_id,)
            g_vals = [
                _g_value_for_ngram(
                    ngram,
                    hash_iv=self._hash_iv,
                    key=key,
                    depth=depth_i,
                )
                for depth_i, key in enumerate(self.config.keys)
            ]
            score = sum(g * w for g, w in zip(g_vals, self._weights)) / depth
            result[token_id] = score

        return result


def _hash_iv_from_keys(keys: tuple[int, ...]) -> int:
    """Derive the hash IV from keys (same as in synthid.py)."""
    import hashlib as _hashlib

    _INT64_MAX = (1 << 63) - 1
    key_bytes = b"".join(
        k.to_bytes(8, byteorder="little", signed=False) for k in keys
    )
    digest = _hashlib.sha256(key_bytes).digest()
    return int.from_bytes(digest, byteorder="big") % _INT64_MAX


def generate_reference_synthid_text(
    detector: SynthIDReferenceDetector,
    *,
    token_count: int,
    watermarked: bool,
) -> str:
    """Generate deterministic reference-backed SynthID watermarked text.

    Uses the same g-value computation as the reference detector, but selects
    tokens to maximise (watermarked) or minimise (unwatermarked) the
    weighted-mean score.  Token selection among ties is deterministic via a
    stable hash.

    Parameters
    ----------
    detector:
        A configured ``SynthIDReferenceDetector``.
    token_count:
        Number of tokens to generate.
    watermarked:
        If ``True``, bias selection toward high weighted-mean scores.

    Returns
    -------
    str
        Space-joined token text.
    """
    if token_count < detector.config.ngram_len:
        raise ValueError("token_count must be at least ngram_len")

    weights = (
        list(detector.config.weights)
        if detector.config.weights is not None
        else _default_weights(len(detector.config.keys))
    )
    hash_iv = detector._hash_iv
    ngram_len = detector.config.ngram_len
    keys = detector.config.keys

    # Start with the first ngram_len - 1 tokens from the vocabulary.
    tokens = list(detector.config.vocabulary[: ngram_len - 1])
    token_ids = [detector.tokenizer.token_id(t) for t in tokens]

    while len(tokens) < token_count:
        context = tuple(token_ids[-(ngram_len - 1) :])

        best_token = None
        best_score = None

        for token in detector.config.vocabulary:
            tid = detector.tokenizer.token_id(token)
            ngram = context + (tid,)
            g_vals = [
                _g_value_for_ngram(ngram, hash_iv=hash_iv, key=key, depth=depth)
                for depth, key in enumerate(keys)
            ]
            depth = len(keys)
            score = sum(g * w for g, w in zip(g_vals, weights)) / depth

            if best_score is None:
                best_token = token
                best_score = score
            elif watermarked and score > best_score:
                best_token = token
                best_score = score
            elif not watermarked and score < best_score:
                best_token = token
                best_score = score
            elif score == best_score:
                # Tie-break deterministically.
                tie_id = stable_hash_int(
                    [
                        "synthid-ref-select",
                        detector.config.configuration_id,
                        detector.config.sampling_table_seed,
                        watermarked,
                        *token_ids[-ngram_len:],
                        tid,
                    ]
                )
                current_tie_id = stable_hash_int(
                    [
                        "synthid-ref-select",
                        detector.config.configuration_id,
                        detector.config.sampling_table_seed,
                        watermarked,
                        *token_ids[-ngram_len:],
                        detector.tokenizer.token_id(best_token),
                    ]
                )
                if tie_id < current_tie_id:
                    best_token = token
                    best_score = score

        tokens.append(best_token)
        token_ids.append(detector.tokenizer.token_id(best_token))

    return " ".join(tokens)


def generate_synthid_token_ids(
    *,
    model: LogitModel,
    prompt_token_ids: list[int],
    processor: SynthIDLogitsProcessor,
    generation_config: SynthIDGenerationConfig,
    watermarked: bool,
) -> list[int]:
    """Generate token IDs from model logits, optionally applying SynthID bias.

    Follows the same pattern as ``generate_kgw_token_ids``: the watermark
    modifies model logits through a LogitsProcessor-style integration, and
    the token is sampled from the resulting distribution.
    """
    from provenance.generation.kgw import sample_from_logits

    rng = random.Random(generation_config.seed)
    input_ids = list(prompt_token_ids)
    generated: list[int] = []

    for _ in range(generation_config.max_new_tokens):
        logits = model.next_token_logits(input_ids)
        if len(logits) != model.vocab_size:
            raise ValueError("model returned logits with unexpected vocabulary size")
        if watermarked:
            logits = processor(input_ids, logits)
        token_id = sample_from_logits(
            logits,
            rng=rng,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
        )
        generated.append(token_id)
        input_ids.append(token_id)

    return generated
