"""Pydantic request/response models for the provenance HTTP API.

All error responses use :class:`ErrorResponse` with a consistent structure.
Sync and async analysis return equivalent detection results.
Pagination responses use :class:`AnalysisListResponse` / :class:`JobListResponse`.
"""

from __future__ import annotations

from enum import Enum
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
    duration_ms: float | None = None


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
    """Paginated response for ``GET /v1/analyses``."""

    analyses: list[AnalysisSummary]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """Lightweight health check response.

    Always returns 200.  Does not check persistence or executor.
    Use ``GET /ready`` for operational readiness.
    """

    status: str = "ok"
    engine_version: str


class UsageSummary(BaseModel):
    """Usage statistics summary (internal)."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_characters: int = 0
    by_endpoint: dict[str, int] = {}
    by_detector: dict[str, int] = {}


class UsageResponse(BaseModel):
    """Response for ``GET /v1/usage``.

    Returns aggregate usage statistics scoped to the calling API key.
    Includes breakdowns by endpoint and detector.
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_characters: int = 0
    by_endpoint: dict[str, int] = {}
    by_detector: dict[str, int] = {}


class CreateApiKeyRequest(BaseModel):
    """Request body for POST /v1/api-keys."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Human-readable label for the key",
    )


class CreateApiKeyResponse(BaseModel):
    """Response for ``POST /v1/api-keys``.

    The raw API key secret is returned **only in this response**.
    It cannot be recovered afterward.
    """

    key_id: str
    name: str
    status: str
    created_at: str
    key: str = Field(
        ...,
        description="The raw API key secret. Shown only once — cannot be recovered.",
    )


class ApiKeySummary(BaseModel):
    """Safe metadata for a stored API key (never includes the secret)."""

    key_id: str
    name: str
    status: str
    created_at: str


class ApiKeyListResponse(BaseModel):
    """Response for ``GET /v1/api-keys``.

    Lists all API keys (admin only).  Never returns raw secrets.
    """

    keys: list[ApiKeySummary]
    total: int


class JobStatus(str, Enum):
    """Valid job statuses."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLATION_REQUESTED = "cancellation_requested"


class AsyncAnalyzeResponse(BaseModel):
    """Response for ``POST /v1/analyze/async``.

    Returns a ``job_id`` that can be used to poll for results.
    Use ``GET /v1/jobs/{job_id}`` to check status.
    """

    job_id: str
    status: str = "queued"
    message: str = "Analysis job submitted"


class JobSummary(BaseModel):
    """Compact job metadata for listing."""

    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    character_count: int = 0
    detectors: list[str] = []
    error_message: str | None = None
    duration_ms: float | None = None
    retry_count: int = 0
    parent_job_id: str | None = None


class JobResponse(BaseModel):
    """Response for ``GET /v1/jobs/{job_id}``.

    Full job details including result (when completed) or error (when failed).
    Only the job owner (or admin) can access this endpoint.
    """

    job_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    character_count: int = 0
    detectors: list[str] = []
    config_path: str | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    result: dict[str, Any] | None = None
    retry_count: int = 0
    parent_job_id: str | None = None


class JobListResponse(BaseModel):
    """Paginated response for ``GET /v1/jobs``."""

    jobs: list[JobSummary]
    total: int
    limit: int
    offset: int


class ReadyResponse(BaseModel):
    """Response for ``GET /ready``.

    Reports operational readiness.  ``status`` is ``"ready"`` only when
    both persistence and executor are ``"ok"``.  Returns HTTP 200 regardless;
    check the ``status`` field programmatically.
    """

    status: str
    engine_version: str
    persistence: str = "ok"
    executor: str = "ok"


class MetricsResponse(BaseModel):
    """Response for ``GET /metrics``.

    Public, machine-readable application metrics.  No authentication required.
    Never exposes API keys, text hashes, job IDs, or other secrets.
    """

    engine_version: str
    # Analysis totals
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_characters: int = 0
    avg_duration_ms: float | None = None
    # Breakdowns
    by_endpoint: dict[str, int] = {}
    by_detector: dict[str, int] = {}
    by_status_code: dict[str, int] = {}
    # Job queue
    jobs_queued: int = 0
    jobs_running: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cancelled: int = 0
    jobs_cancellation_requested: int = 0
    # Executor config
    configured_worker_count: int = 0


# ---------------------------------------------------------------------------
# Error responses
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response.

    All API errors return this structure.  ``detail`` is a human-readable
    message.  ``request_id`` is included when available.
    """

    detail: str
    request_id: str | None = None
