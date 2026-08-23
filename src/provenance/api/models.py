"""Pydantic request/response models for the provenance HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# Detectors the engine can instantiate without external model access.
_VALID_DETECTORS = frozenset({
    "unicode",
    "kgw",
    "kgw-reference",
    "synthid",
    "synthid-reference",
})

# Watermark detectors that require a config file.
_WATERMARK_DETECTORS = _VALID_DETECTORS - {"unicode"}

# Maximum accepted text length (characters).  Kept conservative to avoid
# accidental abuse of a local service.
MAX_TEXT_LENGTH = 200_000


class AnalyzeRequest(BaseModel):
    """Request body for POST /v1/analyze."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="UTF-8 text to analyze",
    )
    detectors: list[str] | None = Field(
        default=None,
        description=(
            "Detectors to run. Defaults to ['unicode']. "
            "Supported: unicode, kgw, kgw-reference, synthid, synthid-reference."
        ),
    )
    config_path: str | None = Field(
        default=None,
        description="Path to detector configuration JSON (required for watermark detectors).",
    )

    @model_validator(mode="after")
    def _validate_detectors(self) -> "AnalyzeRequest":
        names = self.detectors or ["unicode"]
        unknown = set(names) - _VALID_DETECTORS
        if unknown:
            raise ValueError(
                f"Unknown detector(s): {', '.join(sorted(unknown))}. "
                f"Valid options: {', '.join(sorted(_VALID_DETECTORS))}"
            )
        needs_config = set(names) & _WATERMARK_DETECTORS
        if needs_config and self.config_path is None:
            raise ValueError(
                f"config_path is required when using watermark detector(s): "
                f"{', '.join(sorted(needs_config))}"
            )
        return self


class DetectionResultItem(BaseModel):
    """One detector's result, flattened from DetectionResult.to_dict()."""

    detector: str
    detector_version: str
    status: str
    detected: bool | None = None
    implementation_kind: str = "unavailable"
    compatibility: str = "unavailable"
    score: float | int | None = None
    threshold: float | int | None = None
    confidence: str = "unavailable"
    evidence: dict[str, Any] = {}
    text_requirements: dict[str, Any] = {}
    limitations: list[str] = []
    metadata: dict[str, Any] = {}


class AnalyzeResponse(BaseModel):
    """Response body for POST /v1/analyze."""

    analysis_id: str
    engine_version: str
    status: str
    text_stats: dict[str, Any]
    results: list[DetectionResultItem]
    limitations: list[str] = []
    metadata: dict[str, Any] = {}


class AnalysisSummary(BaseModel):
    """Compact summary for list endpoint."""

    analysis_id: str
    timestamp: str
    engine_version: str
    text_hash: str
    character_count: int
    token_count: int
    status: str
    detector_count: int


class AnalysisListResponse(BaseModel):
    """Response for GET /v1/analyses."""

    analyses: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    engine_version: str


class ReadyResponse(BaseModel):
    """Response for GET /ready."""

    status: str
    engine_version: str
