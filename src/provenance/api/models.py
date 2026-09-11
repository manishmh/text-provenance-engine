"""Pydantic request/response models for the provenance HTTP API.

All error responses use :class:`ErrorResponse` with a consistent structure.
Sync and async analysis return equivalent detection results.
Pagination responses use :class:`AnalysisListResponse` / :class:`JobListResponse`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Build detector sets from the centralized registry.
def _build_detector_sets() -> tuple[frozenset[str], frozenset[str]]:
    """Build valid-detector and watermark-detector sets from the registry."""
    from provenance.detectors.registry import get_registry

    reg = get_registry()
    valid = frozenset(reg.names())
    watermark = frozenset(
        name for name, cap in zip(reg.names(), reg.capabilities())
        if cap.requires_config
    )
    return valid, watermark

_VALID_DETECTORS, _WATERMARK_DETECTORS = _build_detector_sets()

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


class BenchmarkRunConfig(BaseModel):
    """Validated benchmark-run configuration (embedded in run responses)."""

    detector: str
    config: str | None = None
    profile: str | None = None
    transforms: list[str] | None = None
    lengths: list[int]
    samples: int
    seed: int


class PublicAnalyzeRequest(BaseModel):
    """Request body for ``POST /v1/public/analyze`` (no API key)."""

    text: str = Field(..., min_length=1, description="UTF-8 text to analyze")


class PublicSignal(BaseModel):
    """One detector family's contribution to a public result (safe subset)."""

    detector: str
    display_name: str
    signal_type: Literal["hidden_unicode_provenance", "statistical_watermark", "provenance_signal"]
    status: Literal["detected", "not_detected", "inconclusive"]
    detected: bool | None = None
    confidence: str = "unavailable"
    evidence: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class PublicDetectorAvailability(BaseModel):
    """Why a registered detector was not run on anonymous arbitrary text."""

    detector: str
    display_name: str
    signal_type: Literal["hidden_unicode_provenance", "statistical_watermark", "provenance_signal"]
    classification: Literal[
        "publicly_usable_arbitrary_input", "reference_config_specific", "benchmark_only"
    ]
    status: Literal["available", "not_applicable", "unavailable_without_key_or_config"]
    compute_class: Literal[
        "cheap_deterministic", "cheap_configured", "potentially_model_backed"
    ]
    reason: str


class QuotaInfo(BaseModel):
    """Remaining-quota metadata returned by public endpoints."""

    plan: str
    limit: int
    used: int
    remaining: int
    max_chars_per_analysis: int


class PublicAnalyzeResponse(BaseModel):
    """Simplified public analysis result (no secrets, no debug fields)."""

    overall_result: Literal[
        "signal_detected", "no_supported_signal_detected", "inconclusive"
    ]
    verdict: str
    signals_checked: list[PublicSignal]
    signals_detected: list[str]
    unavailable_detectors: list[PublicDetectorAvailability]
    character_count: int
    quota: QuotaInfo
    limitations: list[str]
    disclaimer: str
    upgrade_hint: str


class AuthSyncResponse(BaseModel):
    """Result of ``POST /v1/auth/sync`` (post-sign-in provisioning)."""

    auth_user_id: str
    email: str | None
    plan: str
    claimed_events: int
    quota: QuotaInfo


class MeResponse(BaseModel):
    """Identity + entitlements for ``GET /v1/me``."""

    kind: str  # "anonymous" | "user"
    auth_user_id: str | None = None
    email: str | None = None
    plan: str
    quota: QuotaInfo
    entitlements: dict[str, Any]


class BillingCheckoutRequest(BaseModel):
    """Request a server-selected Razorpay Pro subscription checkout.

    The only accepted product selector is ``pro``; price identifiers are
    deliberately not accepted from browsers.
    """

    plan: str = Field(default="pro", max_length=16)


class BillingCheckoutResponse(BaseModel):
    provider: str
    checkout_url: str | None = None
    checkout_data: dict[str, Any] | None = None


class BillingStatusResponse(BaseModel):
    provider: str
    configured: bool
    checkout_mode: str | None = None


class BillingSubscriptionResponse(BaseModel):
    plan: str
    subscription: dict[str, Any] | None = None
    webhook_processing: bool = False


class BenchmarkRunCreate(BaseModel):
    """Request body for ``POST /v1/benchmark-runs``.

    Structural typing only — cross-field and registry validation happens
    server-side in ``provenance.api.benchmarks`` (authoritative). Unknown
    fields are passed through (not dropped) so the validator can reject
    them with structured errors.
    """

    model_config = ConfigDict(extra="allow")

    detector: str = Field(description="Detector name (must support sample generation)")
    config: str | None = Field(
        default=None,
        description="Detector config JSON path (required for watermark detectors)",
    )
    profile: str | None = Field(
        default=None,
        description="Transform profile name (e.g. all_safe); omit to use transforms or all baseline transforms",
    )
    transforms: list[str] | None = Field(
        default=None,
        description="Explicit transform names; takes precedence over profile when both are given",
    )
    lengths: list[int] = Field(description="Sample text lengths to generate")
    samples: int = Field(default=5, description="Samples per length")
    seed: int = Field(default=42, description="Base random seed")


class BenchmarkRunProgress(BaseModel):
    """Honest progress derived from completed/total experiments only."""

    experiments_total: int = 0
    experiments_completed: int = 0
    experiments_failed: int = 0
    current_experiment: str | None = None


class BenchmarkRunSummary(BaseModel):
    """Compact benchmark-run metadata for listing."""

    run_id: str
    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    config: BenchmarkRunConfig
    progress: BenchmarkRunProgress | None = None
    error_message: str | None = None
    duration_ms: float | None = None
    retry_count: int = 0


class BenchmarkRunResponse(BenchmarkRunSummary):
    """Full benchmark-run details including result summary (when completed)."""

    out_dir: str
    result: dict[str, Any] | None = None


class BenchmarkRunListResponse(BaseModel):
    """Paginated response for ``GET /v1/benchmark-runs``."""

    runs: list[BenchmarkRunSummary]
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
