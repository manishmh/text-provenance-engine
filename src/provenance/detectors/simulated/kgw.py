"""Controlled KGW simulation detector.

This is a self-consistent local simulation. It is useful for exercising the
engine and result schema, but it is not compatible with published KGW
implementations because it uses a regex tokenizer and synthetic token selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.preprocessing.text import stable_hash_float, stable_hash_int
from provenance.schemas import DetectionResult
from provenance.tokenizers import SimpleVocabularyTokenizer


DEFAULT_KGW_LIMITATIONS = [
    "Requires the same KGW configuration used during generation.",
    "A negative result is not evidence of human authorship.",
    "A positive result is evidence for the configured KGW watermark, not for a specific model provider.",
    "The Phase 1 adapter uses a deterministic local tokenizer; it is not a universal detector for arbitrary text.",
]


@dataclass(frozen=True)
class KGWConfig:
    configuration_id: str
    version: str
    hash_key: str
    gamma: float
    z_threshold: float
    prefix_length: int
    vocabulary: tuple[str, ...]
    tokenizer: str = "simple"
    case_sensitive: bool = False
    f_scheme: str = "additive"
    window_scheme: str = "left"
    min_scored_tokens: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KGWConfig":
        required = [
            "configuration_id",
            "version",
            "hash_key",
            "gamma",
            "z_threshold",
            "prefix_length",
            "vocabulary",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetectorConfigurationError(f"missing KGW config fields: {', '.join(missing)}")

        config = cls(
            configuration_id=str(data["configuration_id"]),
            version=str(data["version"]),
            hash_key=str(data["hash_key"]),
            gamma=float(data["gamma"]),
            z_threshold=float(data["z_threshold"]),
            prefix_length=int(data["prefix_length"]),
            vocabulary=tuple(str(token) for token in data["vocabulary"]),
            tokenizer=str(data.get("tokenizer", "simple")),
            case_sensitive=bool(data.get("case_sensitive", False)),
            f_scheme=str(data.get("f_scheme", "additive")),
            window_scheme=str(data.get("window_scheme", "left")),
            min_scored_tokens=int(data.get("min_scored_tokens", 20)),
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "KGWConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.tokenizer != "simple":
            raise DetectorConfigurationError("Phase 1 KGW detector supports tokenizer='simple' only")
        if self.window_scheme != "left":
            raise DetectorConfigurationError("Phase 1 KGW detector supports window_scheme='left' only")
        if self.f_scheme not in {"additive", "time", "skip", "min"}:
            raise DetectorConfigurationError(f"unsupported KGW f_scheme: {self.f_scheme}")
        if not 0 < self.gamma < 1:
            raise DetectorConfigurationError("KGW gamma must be between 0 and 1")
        if self.prefix_length < 1:
            raise DetectorConfigurationError("KGW prefix_length must be at least 1")
        if self.min_scored_tokens < 1:
            raise DetectorConfigurationError("KGW min_scored_tokens must be at least 1")
        if len(self.vocabulary) < 2:
            raise DetectorConfigurationError("KGW vocabulary must contain at least two tokens")


class KGWDetector(WatermarkDetector):
    name = "kgw"
    version = "1.0.0-simulation"

    def __init__(self, config: KGWConfig):
        self.config = config
        self.tokenizer = SimpleVocabularyTokenizer(
            vocabulary=config.vocabulary,
            tokenizer_identifier=f"{config.configuration_id}:simple-regex",
            case_sensitive=config.case_sensitive,
            namespace="kgw-simulation-token",
        )
        self._vocab_index = {
            self._normalize_token(token): index for index, token in enumerate(config.vocabulary)
        }

    @classmethod
    def from_config_file(cls, path: str | Path) -> "KGWDetector":
        return cls(KGWConfig.from_file(path))

    def detect(self, text: str) -> DetectionResult:
        token_ids = self.tokenizer.encode(text)
        scored_tokens = max(0, len(token_ids) - self.config.prefix_length)

        if scored_tokens < self.config.min_scored_tokens:
            return DetectionResult(
                detector=self.name,
                detector_version=self.version,
                status="insufficient_text",
                detected=None,
                implementation_kind="simulation",
                compatibility="controlled-local-only",
                score=None,
                threshold=self.config.z_threshold,
                confidence="insufficient_evidence",
                evidence={
                    "token_count": len(token_ids),
                    "scored_token_count": scored_tokens,
                    "configuration_id": self.config.configuration_id,
                    "configuration_version": self.config.version,
                },
                text_requirements={
                    "minimum_scored_tokens": self.config.min_scored_tokens,
                    "actual_scored_tokens": scored_tokens,
                    "prefix_length": self.config.prefix_length,
                },
                limitations=DEFAULT_KGW_LIMITATIONS,
            )

        green_flags = []
        green_count = 0
        for index in range(self.config.prefix_length, len(token_ids)):
            context = token_ids[index - self.config.prefix_length : index]
            is_green = self.is_green_token(context, token_ids[index])
            green_flags.append(is_green)
            if is_green:
                green_count += 1

        z_score = self._z_score(green_count, scored_tokens)
        detected = z_score >= self.config.z_threshold
        confidence = self._confidence(z_score, detected)

        return DetectionResult(
            detector=self.name,
            detector_version=self.version,
            status="ok",
            detected=detected,
            implementation_kind="simulation",
            compatibility="controlled-local-only",
            score=z_score,
            threshold=self.config.z_threshold,
            confidence=confidence,
            evidence={
                "token_count": len(token_ids),
                "scored_token_count": scored_tokens,
                "green_token_count": green_count,
                "green_fraction": green_count / scored_tokens,
                "expected_green_fraction": self.config.gamma,
                "z_score": z_score,
                "green_token_mask": green_flags,
                "configuration_id": self.config.configuration_id,
                "configuration_version": self.config.version,
                "implementation_kind": "simulation",
                "compatibility": "controlled-local-only",
            },
            text_requirements={
                "minimum_scored_tokens": self.config.min_scored_tokens,
                "actual_scored_tokens": scored_tokens,
                "prefix_length": self.config.prefix_length,
                "tokenizer": self.config.tokenizer,
            },
            limitations=DEFAULT_KGW_LIMITATIONS,
            metadata={
                "gamma": self.config.gamma,
                "f_scheme": self.config.f_scheme,
                "window_scheme": self.config.window_scheme,
                "tokenizer_identifier": self.tokenizer.identifier,
            },
        )

    def is_green_token(self, context_ids: list[int], token_id: int) -> bool:
        seed_value = self._f(context_ids)
        value = stable_hash_float(
            [
                "kgw",
                self.config.configuration_id,
                self.config.hash_key,
                self.config.window_scheme,
                seed_value,
                token_id,
            ]
        )
        return value < self.config.gamma

    def select_token(self, context_ids: list[int], *, prefer_green: bool) -> str:
        candidates = []
        for token in self.config.vocabulary:
            token_id = self._token_id(token)
            is_green = self.is_green_token(context_ids, token_id)
            if is_green == prefer_green:
                candidates.append(token)
        if not candidates:
            candidates = list(self.config.vocabulary)

        selector = stable_hash_int(
            [
                "kgw-select",
                self.config.configuration_id,
                self.config.hash_key,
                prefer_green,
                *context_ids,
            ]
        )
        return candidates[selector % len(candidates)]

    def _normalize_token(self, token: str) -> str:
        return token if self.config.case_sensitive else token.lower()

    def _token_id(self, token: str) -> int:
        return self.tokenizer.token_id(token)

    def _f(self, context_ids: list[int]) -> int:
        if len(context_ids) < self.config.prefix_length:
            raise ValueError("KGW context shorter than prefix_length")
        if self.config.f_scheme == "additive":
            return sum(context_ids)
        if self.config.f_scheme == "time":
            value = 1
            for token_id in context_ids:
                value *= max(token_id, 1)
            return value
        if self.config.f_scheme == "skip":
            return context_ids[0]
        return min(context_ids)

    def _z_score(self, green_count: int, scored_tokens: int) -> float:
        expected = self.config.gamma * scored_tokens
        denominator = sqrt(scored_tokens * self.config.gamma * (1 - self.config.gamma))
        return (green_count - expected) / denominator

    def _confidence(self, z_score: float, detected: bool) -> str:
        if detected and z_score >= self.config.z_threshold + 2:
            return "high"
        if detected:
            return "medium"
        if z_score <= 0:
            return "low"
        return "insufficient_evidence"


def generate_controlled_kgw_text(
    detector: KGWDetector,
    *,
    token_count: int,
    watermarked: bool,
) -> str:
    """Generate deterministic local KGW samples for tests and fixtures."""

    if token_count <= detector.config.prefix_length:
        raise ValueError("token_count must exceed prefix_length")

    tokens = list(detector.config.vocabulary[: detector.config.prefix_length])
    token_ids = [detector._token_id(token) for token in tokens]

    while len(tokens) < token_count:
        context = token_ids[-detector.config.prefix_length :]
        token = detector.select_token(context, prefer_green=watermarked)
        tokens.append(token)
        token_ids.append(detector._token_id(token))

    return " ".join(tokens)
