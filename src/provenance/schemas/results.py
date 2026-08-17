"""Typed result objects returned by detectors and the engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class UnicodeFinding:
    """A deterministic Unicode artifact found at one character position."""

    codepoint: str
    name: str
    position: int
    category: str
    severity: str
    reason: str
    character: str

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class DetectionResult:
    """Unified result shape for every signal detector."""

    detector: str
    detector_version: str
    status: str
    detected: bool | None
    implementation_kind: str = "unavailable"
    compatibility: str = "unavailable"
    score: float | int | None = None
    threshold: float | int | None = None
    confidence: str = "unavailable"
    evidence: dict[str, Any] = field(default_factory=dict)
    text_requirements: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(frozen=True)
class AnalysisResult:
    """Top-level result returned by ProvenanceEngine.analyze()."""

    engine_version: str
    status: str
    text_stats: dict[str, Any]
    results: list[DetectionResult]
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)
