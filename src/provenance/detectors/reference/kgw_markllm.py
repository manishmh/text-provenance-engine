"""Reference-backed KGW detector over model token IDs.

Implemented variant:
    KGW left-hash detector following the MarkLLM KGW structure, precisely
    reproducing MarkLLM's `torch.randperm` permutation behavior.

Compatibility note:
    This variant matches genuine MarkLLM 0.1.5 behavior.
"""

from __future__ import annotations

import json
import math
import torch
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from provenance.configuration import ExperimentConfig
from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.schemas import DetectionResult
from provenance.tokenizers import ProvenanceTokenizer, tokenizer_from_config


KGW_MARKLLM_VERSION = "kgw-markllm-v1"

KGW_MARKLLM_LIMITATIONS = [
    "A successful KGW detection demonstrates evidence consistent with the tested KGW configuration. It does not identify the model provider or prove that text was AI-generated.",
    "Evidence is for this explicit KGW configuration only.",
    "A negative result is not evidence of human authorship.",
    "Requires torch for permutation compatibility with MarkLLM.",
]


def classify_backend(tokenizer: "ProvenanceTokenizer") -> tuple[str, str]:
    """Return (implementation_kind, compatibility) for the detector run.

    The KGW math is identical either way; what differs is whether the token IDs
    come from a real model tokenizer or from the controlled local vocabulary.

    - Real Hugging Face tokenizer -> a genuine model/tokenizer-backed KGW run:
      ``("reference", "kgw-reference")``.
    - Simple/controlled tokenizer -> the algorithm running over toy token IDs,
      which must NOT be advertised as a real reference run:
      ``("reference-adapted", "controlled-local-only")``.
    """
    if getattr(tokenizer, "backend", "simple") == "huggingface":
        return "reference", "MarkLLM-compatible"
    return "reference-adapted", "controlled-local-only"


@dataclass(frozen=True)
class KGWMarkLLMConfig:
    configuration_id: str
    version: str
    gamma: float
    delta: float
    hash_key: int
    z_threshold: float
    prefix_length: int
    f_scheme: str = "time"
    window_scheme: str = "left"
    min_scored_tokens: int = 20
    ignore_repeated_ngrams: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KGWMarkLLMConfig":
        if "watermark" in data or "detector" in data:
            configuration_id = str(data["configuration_id"])
            version = str(data.get("version", "1"))
            watermark = dict(data.get("watermark", {}))
            detector = dict(data.get("detector", {}))
            merged = {**watermark, **detector}
            merged.setdefault("configuration_id", configuration_id)
            merged.setdefault("version", version)
            return cls.from_dict(merged)

        required = [
            "configuration_id",
            "version",
            "gamma",
            "delta",
            "hash_key",
            "z_threshold",
            "prefix_length",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetectorConfigurationError(
                f"missing KGW reference config fields: {', '.join(missing)}"
            )

        config = cls(
            configuration_id=str(data["configuration_id"]),
            version=str(data["version"]),
            gamma=float(data["gamma"]),
            delta=float(data["delta"]),
            hash_key=int(data["hash_key"]),
            z_threshold=float(data["z_threshold"]),
            prefix_length=int(data["prefix_length"]),
            f_scheme=str(data.get("f_scheme", "time")),
            window_scheme=str(data.get("window_scheme", "left")),
            min_scored_tokens=int(data.get("min_scored_tokens", 20)),
            ignore_repeated_ngrams=bool(data.get("ignore_repeated_ngrams", False)),
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "KGWMarkLLMConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if not 0 < self.gamma < 1:
            raise DetectorConfigurationError("KGW gamma must be between 0 and 1")
        if self.delta < 0:
            raise DetectorConfigurationError("KGW delta must be non-negative")
        if self.prefix_length < 1:
            raise DetectorConfigurationError("KGW prefix_length must be at least 1")
        if self.min_scored_tokens < 1:
            raise DetectorConfigurationError("KGW min_scored_tokens must be at least 1")
        if self.f_scheme not in {"time", "additive", "skip", "min"}:
            raise DetectorConfigurationError(f"unsupported KGW f_scheme: {self.f_scheme}")
        if self.window_scheme != "left":
            raise DetectorConfigurationError("KGW reference detector currently supports left window scheme only")


@dataclass(frozen=True)
class KGWScore:
    token_count: int
    scored_token_count: int
    green_token_count: int
    green_token_mask: list[bool | None]
    ignored_repeated_count: int
    z_score: float
    p_value: float

    @property
    def green_fraction(self) -> float:
        if self.scored_token_count == 0:
            return 0.0
        return self.green_token_count / self.scored_token_count


class KGWMarkLLMScorer:
    """KGW math utilities independent from text tokenization."""

    def __init__(self, config: KGWMarkLLMConfig, *, vocab_size: int, device: str = "cpu"):
        if vocab_size < 2:
            raise DetectorConfigurationError("KGW vocab_size must be at least 2")
        self.config = config
        self.vocab_size = vocab_size
        self.device = device
        self.greenlist_size = int(vocab_size * config.gamma)
        if self.greenlist_size < 1:
            raise DetectorConfigurationError("KGW gamma yields an empty greenlist")
        
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(config.hash_key)
        self._prf = torch.randperm(self.vocab_size, device=self.device, generator=self._rng).tolist()

    def score_token_ids(self, token_ids: list[int]) -> KGWScore:
        token_count = len(token_ids)
        if token_count <= self.config.prefix_length:
            return KGWScore(
                token_count=token_count,
                scored_token_count=0,
                green_token_count=0,
                green_token_mask=[],
                ignored_repeated_count=0,
                z_score=0.0,
                p_value=1.0,
            )

        green_count = 0
        scored_count = 0
        ignored_repeated_count = 0
        flags: list[bool | None] = []
        seen_ngrams: set[tuple[int, ...]] = set()

        for index in range(self.config.prefix_length, token_count):
            context = token_ids[index - self.config.prefix_length : index]
            token_id = token_ids[index]
            ngram = tuple(context + [token_id])
            if self.config.ignore_repeated_ngrams and ngram in seen_ngrams:
                ignored_repeated_count += 1
                flags.append(None)
                continue
            seen_ngrams.add(ngram)

            is_green = self.is_green_token(context, token_id)
            flags.append(is_green)
            scored_count += 1
            if is_green:
                green_count += 1

        if scored_count == 0:
            z_score = 0.0
            p_value = 1.0
        else:
            z_score = self.z_score(green_count, scored_count)
            p_value = self.p_value(z_score)

        return KGWScore(
            token_count=token_count,
            scored_token_count=scored_count,
            green_token_count=green_count,
            green_token_mask=flags,
            ignored_repeated_count=ignored_repeated_count,
            z_score=z_score,
            p_value=p_value,
        )

    def greenlist_ids(self, context_ids: list[int]) -> list[int]:
        if len(context_ids) < self.config.prefix_length:
            raise ValueError("KGW context shorter than prefix_length")
        seed = (self.config.hash_key * self._f(context_ids)) % self.vocab_size
        return self._permutation(seed)[: self.greenlist_size]

    def is_green_token(self, context_ids: list[int], token_id: int) -> bool:
        return token_id in set(self.greenlist_ids(context_ids))

    def bias_logits(self, context_ids: list[int], logits: list[float]) -> list[float]:
        if len(logits) != self.vocab_size:
            raise ValueError("logit vector length must equal KGW vocab_size")
        if len(context_ids) < self.config.prefix_length:
            return list(logits)
        biased = list(logits)
        for token_id in self.greenlist_ids(context_ids):
            biased[token_id] += self.config.delta
        return biased

    def z_score(self, green_count: int, scored_count: int) -> float:
        expected = self.config.gamma * scored_count
        denominator = math.sqrt(scored_count * self.config.gamma * (1 - self.config.gamma))
        return (green_count - expected) / denominator

    def p_value(self, z_score: float) -> float:
        return 0.5 * math.erfc(z_score / math.sqrt(2))

    def _f(self, context_ids: list[int]) -> int:
        prefix = context_ids[-self.config.prefix_length :]
        if self.config.f_scheme == "time":
            value = 1
            for token_id in prefix:
                value *= token_id
            return self._prf[value % self.vocab_size]
        if self.config.f_scheme == "additive":
            return self._prf[sum(prefix) % self.vocab_size]
        if self.config.f_scheme == "skip":
            return self._prf[prefix[0] % self.vocab_size]
        return min(self._prf[token_id % self.vocab_size] for token_id in prefix)

    @lru_cache(maxsize=4096)
    def _permutation(self, seed: int) -> list[int]:
        self._rng.manual_seed(int(seed))
        return torch.randperm(self.vocab_size, device=self.device, generator=self._rng).tolist()


class KGWMarkLLMDetector(WatermarkDetector):
    name = "kgw-markllm"
    version = KGW_MARKLLM_VERSION

    def __init__(self, config: KGWMarkLLMConfig, tokenizer: ProvenanceTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.scorer = KGWMarkLLMScorer(config, vocab_size=tokenizer.vocabulary_size)
        self.implementation_kind, self.compatibility = classify_backend(tokenizer)

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        tokenizer: ProvenanceTokenizer | None = None,
    ) -> "KGWMarkLLMDetector":
        experiment = ExperimentConfig.from_file(path)
        resolved_tokenizer = tokenizer or tokenizer_from_config(experiment.tokenizer.to_factory_dict())
        config = KGWMarkLLMConfig.from_dict(
            {
                "configuration_id": experiment.configuration_id,
                "version": experiment.version,
                **experiment.watermark_configuration,
                **experiment.detector_configuration,
            }
        )
        return cls(config, resolved_tokenizer)

    def detect(self, text: str) -> DetectionResult:
        token_ids = self.tokenizer.encode(text)
        score = self.scorer.score_token_ids(token_ids)

        if score.scored_token_count < self.config.min_scored_tokens:
            return DetectionResult(
                detector=self.name,
                detector_version=self.version,
                status="insufficient_text",
                detected=None,
                implementation_kind=self.implementation_kind,
                compatibility=self.compatibility,
                score=None,
                threshold=self.config.z_threshold,
                confidence="insufficient_evidence",
                evidence=self._evidence(score, token_ids),
                text_requirements={
                    "minimum_scored_tokens": self.config.min_scored_tokens,
                    "actual_scored_tokens": score.scored_token_count,
                    "prefix_length": self.config.prefix_length,
                    "tokenizer_identifier": self.tokenizer.identifier,
                },
                limitations=KGW_MARKLLM_LIMITATIONS,
            )

        detected = score.z_score >= self.config.z_threshold
        return DetectionResult(
            detector=self.name,
            detector_version=self.version,
            status="ok",
            detected=detected,
            implementation_kind=self.implementation_kind,
            compatibility=self.compatibility,
            score=score.z_score,
            threshold=self.config.z_threshold,
            confidence=self._confidence(score.z_score, score.p_value, detected),
            evidence=self._evidence(score, token_ids),
            text_requirements={
                "minimum_scored_tokens": self.config.min_scored_tokens,
                "actual_scored_tokens": score.scored_token_count,
                "prefix_length": self.config.prefix_length,
                "tokenizer_identifier": self.tokenizer.identifier,
                "model_identifier": self.tokenizer.model_identifier,
            },
            limitations=KGW_MARKLLM_LIMITATIONS,
            metadata={
                "scheme": "kgw",
                "variant": KGW_MARKLLM_VERSION,
                "gamma": self.config.gamma,
                "delta": self.config.delta,
                "f_scheme": self.config.f_scheme,
                "window_scheme": self.config.window_scheme,
                "ignore_repeated_ngrams": self.config.ignore_repeated_ngrams,
            },
        )

    def _evidence(self, score: KGWScore, token_ids: list[int]) -> dict[str, Any]:
        return {
            "scheme": "kgw",
            "configuration_id": self.config.configuration_id,
            "configuration_version": self.config.version,
            "tokenizer_identifier": self.tokenizer.identifier,
            "tokenizer_backend": getattr(self.tokenizer, "backend", "simple"),
            "tokenizer_revision": getattr(self.tokenizer, "revision", None),
            "model_identifier": self.tokenizer.model_identifier,
            "token_ids": token_ids,
            "token_count": score.token_count,
            "scored_token_count": score.scored_token_count,
            "green_token_count": score.green_token_count,
            "green_fraction": score.green_fraction,
            "expected_green_fraction": self.config.gamma,
            "green_token_mask": score.green_token_mask,
            "ignored_repeated_count": score.ignored_repeated_count,
            "z_score": score.z_score,
            "p_value": score.p_value,
            "implementation_kind": self.implementation_kind,
            "compatibility": self.compatibility,
            "variant": KGW_MARKLLM_VERSION,
        }

    def _confidence(self, z_score: float, p_value: float, detected: bool) -> str:
        if detected and p_value <= 1e-6:
            return "high"
        if detected:
            return "medium"
        if z_score <= 0:
            return "low"
        return "insufficient_evidence"
