"""Typed response models for the SDK.

These are lightweight dataclasses that mirror the API response schemas.
They are created from raw JSON dicts returned by the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """One detector's finding."""

    detector: str
    detector_version: str
    status: str
    detected: bool | None = None
    implementation_kind: str = "unavailable"
    compatibility: str = "unavailable"
    score: float | int | None = None
    threshold: float | int | None = None
    confidence: str = "unavailable"
    evidence: dict[str, Any] = field(default_factory=dict)
    text_requirements: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AnalyzeResult:
    """Full analysis result from ``POST /v1/analyze``."""

    analysis_id: str
    engine_version: str
    status: str
    text_stats: dict[str, Any] = field(default_factory=dict)
    results: list[DetectionResult] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AnalyzeResult:
        results = [DetectionResult.from_dict(r) for r in d.get("results", [])]
        return cls(
            analysis_id=d["analysis_id"],
            engine_version=d["engine_version"],
            status=d["status"],
            text_stats=d.get("text_stats", {}),
            results=results,
            limitations=d.get("limitations", []),
            metadata=d.get("metadata", {}),
            duration_ms=d.get("duration_ms"),
        )


@dataclass
class AsyncAnalyzeResult:
    """Result from ``POST /v1/analyze/async``."""

    job_id: str
    status: str = "queued"
    message: str = "Analysis job submitted"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AsyncAnalyzeResult:
        return cls(
            job_id=d["job_id"],
            status=d.get("status", "queued"),
            message=d.get("message", ""),
        )


@dataclass
class JobResult:
    """Full job details from ``GET /v1/jobs/{job_id}``."""

    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    character_count: int = 0
    detectors: list[str] = field(default_factory=list)
    config_path: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    result: dict[str, Any] | None = None
    retry_count: int = 0
    parent_job_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        """True if the job is in a final state (completed, failed, cancelled)."""
        return self.status in ("completed", "failed", "cancelled")

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JobSummary:
    """Compact job metadata for listing."""

    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    character_count: int = 0
    detectors: list[str] = field(default_factory=list)
    error_message: str | None = None
    duration_ms: float | None = None
    retry_count: int = 0
    parent_job_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobSummary:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class UsageResult:
    """Usage statistics from ``GET /v1/usage``."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_characters: int = 0
    by_endpoint: dict[str, int] = field(default_factory=dict)
    by_detector: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UsageResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CancelResult:
    """Result from ``DELETE /v1/jobs/{job_id}``."""

    job_id: str
    status: str
    message: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CancelResult:
        return cls(
            job_id=d["job_id"],
            status=d["status"],
            message=d.get("message", ""),
        )
