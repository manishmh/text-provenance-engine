"""Controlled KGW generation over model logits."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Protocol

from provenance.detectors.reference.kgw import KGWReferenceScorer
from provenance.detectors.reference.kgw_markllm import KGWMarkLLMScorer

KGWScorerType = KGWReferenceScorer | KGWMarkLLMScorer


class LogitModel(Protocol):
    identifier: str
    vocab_size: int

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the exact input token IDs."""


@dataclass(frozen=True)
class KGWGenerationConfig:
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_k: int | None = None
    top_p: float | None = None
    seed: int = 0


class KGWLogitsProcessor:
    """Apply KGW green-token logit bias to model logits."""

    def __init__(self, scorer: KGWScorerType):
        self.scorer = scorer

    def __call__(self, input_ids: list[int], logits: list[float]) -> list[float]:
        return self.scorer.bias_logits(input_ids, logits)


def generate_kgw_token_ids(
    *,
    model: LogitModel,
    prompt_token_ids: list[int],
    scorer: KGWScorerType,
    generation_config: KGWGenerationConfig,
    watermarked: bool,
) -> list[int]:
    """Generate token IDs from model logits, optionally applying KGW bias."""

    rng = random.Random(generation_config.seed)
    input_ids = list(prompt_token_ids)
    generated: list[int] = []
    processor = KGWLogitsProcessor(scorer)

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


def sample_from_logits(
    logits: list[float],
    *,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> int:
    """Temperature / top-k / nucleus (top-p) sampling over a logit vector.

    The KGW logit bias is already applied to ``logits`` before this is called, so
    the watermark influences sampling exclusively through the model's logit
    distribution -- green tokens are never selected directly.
    """
    if temperature <= 0:
        return max(range(len(logits)), key=lambda index: logits[index])

    candidates = list(range(len(logits)))
    if top_k is not None and 0 < top_k < len(candidates):
        candidates.sort(key=lambda index: logits[index], reverse=True)
        candidates = candidates[:top_k]

    scaled = [logits[index] / temperature for index in candidates]
    max_logit = max(scaled)
    weights = [math.exp(value - max_logit) for value in scaled]

    if top_p is not None and 0 < top_p < 1:
        order = sorted(range(len(candidates)), key=lambda i: weights[i], reverse=True)
        total_weight = sum(weights)
        cumulative_p = 0.0
        kept: list[int] = []
        for i in order:
            kept.append(i)
            cumulative_p += weights[i] / total_weight
            if cumulative_p >= top_p:
                break
        candidates = [candidates[i] for i in kept]
        weights = [weights[i] for i in kept]

    total = sum(weights)
    target = rng.random() * total
    cumulative = 0.0
    for token_id, weight in zip(candidates, weights):
        cumulative += weight
        if cumulative >= target:
            return token_id
    return candidates[-1]
