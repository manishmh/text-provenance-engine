"""Controlled SynthID-Text simulation detector.

This is a self-consistent local simulation inspired by the weighted-mean score
shape. It is not compatible with Google's SynthID-Text reference implementation
or production Gemini watermark detection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.preprocessing.text import stable_hash_float, stable_hash_int
from provenance.schemas import DetectionResult
from provenance.tokenizers import SimpleVocabularyTokenizer


DEFAULT_SYNTHID_LIMITATIONS = [
    "Requires the same SynthID-Text configuration used during generation.",
    "Demonstrates detection for known local configurations only.",
    "Does not detect arbitrary production Gemini output or infer Google production keys.",
    "A negative result is not evidence of human authorship.",
]


@dataclass(frozen=True)
class SynthIDConfig:
    configuration_id: str
    version: str
    keys: tuple[int, ...]
    ngram_len: int
    threshold: float
    vocabulary: tuple[str, ...]
    tokenizer: str = "simple"
    case_sensitive: bool = False
    sampling_table_size: int = 4096
    sampling_table_seed: int = 0
    context_history_size: int = 1024
    weights: tuple[float, ...] | None = None
    min_usable_tokens: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthIDConfig":
        required = [
            "configuration_id",
            "version",
            "keys",
            "ngram_len",
            "threshold",
            "vocabulary",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetectorConfigurationError(
                f"missing SynthID config fields: {', '.join(missing)}"
            )

        weights = data.get("weights")
        config = cls(
            configuration_id=str(data["configuration_id"]),
            version=str(data["version"]),
            keys=tuple(int(key) for key in data["keys"]),
            ngram_len=int(data["ngram_len"]),
            threshold=float(data["threshold"]),
            vocabulary=tuple(str(token) for token in data["vocabulary"]),
            tokenizer=str(data.get("tokenizer", "simple")),
            case_sensitive=bool(data.get("case_sensitive", False)),
            sampling_table_size=int(data.get("sampling_table_size", 4096)),
            sampling_table_seed=int(data.get("sampling_table_seed", 0)),
            context_history_size=int(data.get("context_history_size", 1024)),
            weights=tuple(float(weight) for weight in weights) if weights is not None else None,
            min_usable_tokens=int(data.get("min_usable_tokens", 20)),
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "SynthIDConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.tokenizer != "simple":
            raise DetectorConfigurationError("Phase 1 SynthID detector supports tokenizer='simple' only")
        if self.ngram_len < 2:
            raise DetectorConfigurationError("SynthID ngram_len must be at least 2")
        if not self.keys:
            raise DetectorConfigurationError("SynthID keys must not be empty")
        if self.weights is not None and len(self.weights) != len(self.keys):
            raise DetectorConfigurationError("SynthID weights must match number of keys")
        if len(self.vocabulary) < 2:
            raise DetectorConfigurationError("SynthID vocabulary must contain at least two tokens")
        if self.min_usable_tokens < 1:
            raise DetectorConfigurationError("SynthID min_usable_tokens must be at least 1")
        if not 0 <= self.threshold <= 1:
            raise DetectorConfigurationError("SynthID weighted-mean threshold must be in [0, 1]")


class SynthIDTextDetector(WatermarkDetector):
    name = "synthid"
    version = "1.0.0-simulation"

    def __init__(self, config: SynthIDConfig):
        self.config = config
        self.tokenizer = SimpleVocabularyTokenizer(
            vocabulary=config.vocabulary,
            tokenizer_identifier=f"{config.configuration_id}:simple-regex",
            case_sensitive=config.case_sensitive,
            namespace="synthid-simulation-token",
        )
        self._vocab_index = {
            self._normalize_token(token): index for index, token in enumerate(config.vocabulary)
        }
        self._weights = self._normalized_weights()

    @classmethod
    def from_config_file(cls, path: str | Path) -> "SynthIDTextDetector":
        return cls(SynthIDConfig.from_file(path))

    def detect(self, text: str) -> DetectionResult:
        token_ids = self.tokenizer.encode(text)
        g_values, usable_mask = self._g_values_and_mask(token_ids)
        usable_count = sum(usable_mask)

        if usable_count < self.config.min_usable_tokens:
            return DetectionResult(
                detector=self.name,
                detector_version=self.version,
                status="insufficient_text",
                detected=None,
                implementation_kind="simulation",
                compatibility="controlled-local-only",
                score=None,
                threshold=self.config.threshold,
                confidence="insufficient_evidence",
                evidence={
                    "token_count": len(token_ids),
                    "usable_token_count": usable_count,
                    "usable_mask": usable_mask,
                    "configuration_id": self.config.configuration_id,
                    "configuration_version": self.config.version,
                },
                text_requirements={
                    "minimum_usable_tokens": self.config.min_usable_tokens,
                    "actual_usable_tokens": usable_count,
                    "ngram_len": self.config.ngram_len,
                },
                limitations=DEFAULT_SYNTHID_LIMITATIONS,
            )

        score = self._weighted_mean_score(g_values, usable_mask)
        detected = score >= self.config.threshold
        confidence = self._confidence(score, detected)

        return DetectionResult(
            detector=self.name,
            detector_version=self.version,
            status="ok",
            detected=detected,
            implementation_kind="simulation",
            compatibility="controlled-local-only",
            score=score,
            threshold=self.config.threshold,
            confidence=confidence,
            evidence={
                "raw_score": score,
                "token_count": len(token_ids),
                "usable_token_count": usable_count,
                "usable_mask": usable_mask,
                "g_values": g_values,
                "watermarking_depth": len(self.config.keys),
                "scorer": "weighted_mean",
                "configuration_id": self.config.configuration_id,
                "configuration_version": self.config.version,
                "implementation_kind": "simulation",
                "compatibility": "controlled-local-only",
            },
            text_requirements={
                "minimum_usable_tokens": self.config.min_usable_tokens,
                "actual_usable_tokens": usable_count,
                "ngram_len": self.config.ngram_len,
                "tokenizer": self.config.tokenizer,
            },
            limitations=DEFAULT_SYNTHID_LIMITATIONS,
            metadata={
                "sampling_table_size": self.config.sampling_table_size,
                "sampling_table_seed": self.config.sampling_table_seed,
                "context_history_size": self.config.context_history_size,
                "weights": self._weights,
                "tokenizer_identifier": self.tokenizer.identifier,
            },
        )

    def candidate_weighted_value(self, token_ids: list[int], token_id: int) -> float:
        prospective = token_ids + [token_id]
        if len(prospective) < self.config.ngram_len:
            return 0.0
        context = prospective[-self.config.ngram_len :]
        g_values = self._g_values_for_context(context)
        return sum(value * weight for value, weight in zip(g_values, self._weights)) / len(
            self.config.keys
        )

    def select_token(self, token_ids: list[int], *, prefer_high_score: bool) -> str:
        scored = []
        for token in self.config.vocabulary:
            token_id = self._token_id(token)
            scored.append((self.candidate_weighted_value(token_ids, token_id), token))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=prefer_high_score)
        best_score = scored[0][0]
        tied = [token for score, token in scored if score == best_score]
        selector = stable_hash_int(
            [
                "synthid-select",
                self.config.configuration_id,
                self.config.sampling_table_seed,
                prefer_high_score,
                *token_ids[-self.config.ngram_len :],
            ]
        )
        return tied[selector % len(tied)]

    def _g_values_and_mask(self, token_ids: list[int]) -> tuple[list[list[int]], list[bool]]:
        g_values: list[list[int]] = []
        usable_mask: list[bool] = []
        seen_contexts: list[tuple[int, ...]] = []

        for index in range(len(token_ids)):
            if index < self.config.ngram_len - 1:
                g_values.append([0 for _ in self.config.keys])
                usable_mask.append(False)
                continue

            context = tuple(token_ids[index - self.config.ngram_len + 1 : index + 1])
            repeated = context in seen_contexts[-self.config.context_history_size :]
            seen_contexts.append(context)
            g_values.append(self._g_values_for_context(list(context)))
            usable_mask.append(not repeated)

        return g_values, usable_mask

    def _g_values_for_context(self, context_ids: list[int]) -> list[int]:
        return [
            self._g_value_for_key(context_ids, key, depth)
            for depth, key in enumerate(self.config.keys)
        ]

    def _g_value_for_key(self, context_ids: list[int], key: int, depth: int) -> int:
        value = stable_hash_float(
            [
                "synthid-g",
                self.config.configuration_id,
                self.config.sampling_table_seed,
                self.config.sampling_table_size,
                key,
                depth,
                *context_ids,
            ]
        )
        return 1 if value >= 0.5 else 0

    def _weighted_mean_score(self, g_values: list[list[int]], usable_mask: list[bool]) -> float:
        usable_count = sum(usable_mask)
        if usable_count == 0:
            return 0.0
        weighted_sum = 0.0
        for values, usable in zip(g_values, usable_mask):
            if not usable:
                continue
            weighted_sum += sum(value * weight for value, weight in zip(values, self._weights))
        return weighted_sum / (len(self.config.keys) * usable_count)

    def _normalized_weights(self) -> list[float]:
        depth = len(self.config.keys)
        if self.config.weights is None:
            weights = [10.0 - i * (9.0 / max(depth - 1, 1)) for i in range(depth)]
        else:
            weights = list(self.config.weights)
        total = sum(weights)
        return [weight * depth / total for weight in weights]

    def _normalize_token(self, token: str) -> str:
        return token if self.config.case_sensitive else token.lower()

    def _token_id(self, token: str) -> int:
        return self.tokenizer.token_id(token)

    def _confidence(self, score: float, detected: bool) -> str:
        margin = abs(score - self.config.threshold)
        if detected and margin >= 0.15:
            return "high"
        if detected:
            return "medium"
        if margin < 0.05:
            return "insufficient_evidence"
        return "low"


def generate_controlled_synthid_text(
    detector: SynthIDTextDetector,
    *,
    token_count: int,
    watermarked: bool,
) -> str:
    """Generate deterministic local SynthID samples for tests and fixtures."""

    if token_count < detector.config.ngram_len:
        raise ValueError("token_count must be at least ngram_len")

    tokens = list(detector.config.vocabulary[: detector.config.ngram_len - 1])
    token_ids = [detector._token_id(token) for token in tokens]

    while len(tokens) < token_count:
        token = detector.select_token(token_ids, prefer_high_score=watermarked)
        tokens.append(token)
        token_ids.append(detector._token_id(token))

    return " ".join(tokens)
