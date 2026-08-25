"""FastAPI route handlers for the provenance HTTP API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from provenance.api.auth import RequireAdminKey, RequireAPIKey
from provenance.api.db import AnalysisRepository  # Protocol
from provenance.api.keys import generate_api_key, generate_key_id, hash_api_key
from provenance.api.middleware import rate_limit_dependency
from provenance.api.state import get_repo, set_repo
from provenance.api.models import (
    AnalysisListResponse,
    AnalysisSummary,
    AnalyzeRequest,
    AnalyzeResponse,
    ApiKeyListResponse,
    ApiKeySummary,
    AsyncAnalyzeResponse,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    DetectionResultItem,
    ErrorResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    JobSummary,
    MetricsResponse,
    ReadyResponse,
    UsageResponse,
)
from provenance.engine import ENGINE_VERSION
from provenance.schemas import DetectionResult

if TYPE_CHECKING:
    from provenance.detectors import WatermarkDetector

logger = logging.getLogger("provenance.api")
router = APIRouter()

# Valid terminal job statuses that cannot be cancelled
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
# Valid statuses for retry
_RETRYABLE_STATUSES = frozenset({"failed"})
# Valid cancellation transitions
_CANCEL_FROM = frozenset({"queued", "running", "cancellation_requested"})


# Backward-compatible alias for configure_repo
configure_repo = set_repo


def _get_repo() -> AnalysisRepository:
    return get_repo()


def _get_daily_limits() -> tuple[int | None, int | None]:
    """Read daily request/character limits from environment.

    Returns (max_requests, max_characters). None means unlimited.
    """
    req_str = os.environ.get("PROVENANCE_DAILY_REQUEST_LIMIT")
    char_str = os.environ.get("PROVENANCE_DAILY_CHARACTER_LIMIT")
    req_limit = int(req_str) if req_str else None
    char_limit = int(char_str) if char_str else None
    return req_limit, char_limit


def _unconfigured_result(detector_name: str) -> DetectionResult:
    return DetectionResult(
        detector=detector_name,
        detector_version="unavailable",
        status="not_configured",
        detected=None,
        implementation_kind="unavailable",
        compatibility="unavailable",
        confidence="unavailable",
        evidence={},
        text_requirements={"configuration_required": True},
        limitations=[
            f"{detector_name} detection requires an explicit known watermark configuration.",
            "No AI or human authorship conclusion is available.",
        ],
    )


def _build_detectors(
    detector_names: list[str], config_path: str | None
) -> tuple[list[WatermarkDetector], list[DetectionResult]]:
    """Instantiate detectors and collect unconfigured results."""
    # Lazy imports so the API module loads without torch/transformers
    from provenance.detectors import (
        KGWDetector,
        KGWReferenceDetector,
        SynthIDReferenceDetector,
        SynthIDTextDetector,
    )

    detectors: list[WatermarkDetector] = []
    extra: list[DetectionResult] = []

    for name in detector_names:
        if name == "unicode":
            continue
        if config_path is None:
            extra.append(_unconfigured_result(name))
            continue
        if name == "kgw":
            detectors.append(KGWDetector.from_config_file(config_path))
        elif name == "kgw-reference":
            detectors.append(KGWReferenceDetector.from_config_file(config_path))
        elif name == "synthid":
            detectors.append(SynthIDTextDetector.from_config_file(config_path))
        elif name == "synthid-reference":
            detectors.append(SynthIDReferenceDetector.from_config_file(config_path))

    return detectors, extra


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Lightweight health check. Always returns 200 if the server is running.",
    tags=["operations"],
)
def health() -> HealthResponse:
    return HealthResponse(engine_version=ENGINE_VERSION)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description=(
        "Verifies persistence (database) and executor (background worker) are usable. "
        "Returns 'status: ready' only when all subsystems are OK. "
        "Returns HTTP 200 regardless; check the 'status' field programmatically."
    ),
    tags=["operations"],
)
def ready() -> ReadyResponse:
    """Readiness probe — verifies persistence and executor are usable."""
    persistence_status = "ok"
    executor_status = "ok"

    # Check persistence
    try:
        repo = _get_repo()
        repo.list_analyses(limit=1, offset=0)
    except Exception:
        persistence_status = "error"

    # Check executor (only if shutting down)
    try:
        from provenance.api.jobs import is_shutting_down
        if is_shutting_down():
            executor_status = "shutting_down"
    except Exception:
        pass

    is_ready = persistence_status == "ok" and executor_status == "ok"
    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        engine_version=ENGINE_VERSION,
        persistence=persistence_status,
        executor=executor_status,
    )


# ---------------------------------------------------------------------------
# Metrics (public)
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Application metrics",
    description=(
        "Machine-readable application metrics.  No authentication required. "
        "Includes analysis totals, duration averages, job queue counts, "
        "and breakdowns by endpoint, detector, and status code. "
        "Never exposes API keys, text hashes, or job IDs."
    ),
    tags=["operations"],
)
def metrics() -> MetricsResponse:
    """Public machine-readable application metrics."""
    repo = _get_repo()

    # Usage aggregates (global, no key filter)
    usage = repo.get_usage_summary()

    # Average duration
    avg_duration = None
    try:
        avg_duration = repo.get_avg_duration()
    except Exception:
        pass

    # Job counts
    job_counts = repo.get_job_counts()

    # Executor config
    try:
        from provenance.api.jobs import _get_max_workers
        worker_count = _get_max_workers()
    except Exception:
        worker_count = 2

    # Usage by status code
    by_status_code: dict[str, int] = {}
    try:
        by_status_code = repo.get_usage_by_status()
    except Exception:
        pass

    return MetricsResponse(
        engine_version=ENGINE_VERSION,
        total_requests=usage.get("total_requests", 0),
        successful_requests=usage.get("successful_requests", 0),
        failed_requests=usage.get("failed_requests", 0),
        total_characters=usage.get("total_characters", 0),
        avg_duration_ms=avg_duration,
        by_endpoint=usage.get("by_endpoint", {}),
        by_detector=usage.get("by_detector", {}),
        by_status_code=by_status_code,
        jobs_queued=job_counts.get("queued", 0),
        jobs_running=job_counts.get("running", 0),
        jobs_completed=job_counts.get("completed", 0),
        jobs_failed=job_counts.get("failed", 0),
        jobs_cancelled=job_counts.get("cancelled", 0),
        jobs_cancellation_requested=job_counts.get("cancellation_requested", 0),
        configured_worker_count=worker_count,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_id(auth: dict) -> str:
    """Extract the stable key identifier from an auth identity dict."""
    return auth.get("id", "_none")


def _check_daily_limits(key_id: str, text_len: int) -> None:
    """Check daily request/character limits; raise 429 if exceeded."""
    req_limit, char_limit = _get_daily_limits()
    if not req_limit and not char_limit:
        return
    repo = _get_repo()
    today_reqs, today_chars = repo.get_usage_today(key_id)
    job_reqs, job_chars = repo.get_usage_today_jobs(key_id)
    total_reqs = today_reqs + job_reqs
    total_chars = today_chars + job_chars
    # SQLite: SELECTs may hold implicit transactions; release them.
    repo.commit_if_needed()
    if req_limit and total_reqs >= req_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily request limit ({req_limit}) exceeded",
            headers={
                "X-Usage-Requests": str(total_reqs),
                "X-Usage-Request-Limit": str(req_limit),
            },
        )
    if char_limit and total_chars + text_len > char_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily character limit ({char_limit}) exceeded",
            headers={
                "X-Usage-Characters": str(total_chars),
                "X-Usage-Character-Limit": str(char_limit),
            },
        )


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v1/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze text",
    description=(
        "Run detection on the provided text using the specified detectors. "
        "Returns analysis results including detector findings, text statistics, "
        "and an 'analysis_id' for retrieval."
    ),
    response_description="Analysis results with detector findings",
    tags=["analysis"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid request (unknown detector, missing config)"},
        429: {"model": ErrorResponse, "description": "Rate limit or daily usage limit exceeded"},
    },
)
def analyze(request: AnalyzeRequest, auth: RequireAPIKey = None, _rl: None = Depends(rate_limit_dependency)) -> Response:
    key_id = _auth_id(auth)
    _check_daily_limits(key_id, len(request.text))

    start_time = time.monotonic()

    from provenance.api.service import run_analysis
    detector_names = request.detectors or ["unicode"]
    result_dict = run_analysis(request.text, detector_names, request.config_path)

    # Persist
    repo = _get_repo()
    analysis_id = repo.save(
        text=request.text,
        engine_version=ENGINE_VERSION,
        detectors=detector_names,
        result_dict=result_dict,
    )

    duration_ms = round((time.monotonic() - start_time) * 1000, 2)

    # Record usage
    char_count = result_dict.get("text_stats", {}).get("character_count", len(request.text))
    repo.record_usage(
        key_id=key_id,
        endpoint="/v1/analyze",
        status_code=200,
        duration_ms=duration_ms,
        character_count=char_count,
        success=True,
        detector=detector_names[0] if len(detector_names) == 1 else None,
    )

    # Build usage headers
    resp_model = AnalyzeResponse(
        analysis_id=analysis_id,
        engine_version=result_dict.get("engine_version", ENGINE_VERSION),
        status=result_dict.get("status", "ok"),
        text_stats=result_dict.get("text_stats", {}),
        results=[DetectionResultItem(**r) for r in result_dict.get("results", [])],
        limitations=result_dict.get("limitations", []),
        metadata=result_dict.get("metadata", {}),
        duration_ms=duration_ms,
    )
    headers = _usage_headers(key_id)
    return Response(
        content=resp_model.model_dump_json(),
        media_type="application/json",
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Async analysis
# ---------------------------------------------------------------------------


@router.post(
    "/v1/analyze/async",
    response_model=AsyncAnalyzeResponse,
    status_code=202,
    summary="Submit async analysis",
    description=(
        "Submit text for background analysis.  Returns immediately with HTTP 202. "
        "Use 'GET /v1/jobs/{job_id}' to poll for the result. "
        "The job lifecycle is: queued -> running -> completed/failed. "
        "Use 'DELETE /v1/jobs/{job_id}' to cancel a queued or running job."
    ),
    response_description="Job submission confirmation",
    tags=["jobs"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"model": ErrorResponse, "description": "Invalid request"},
        429: {"model": ErrorResponse, "description": "Rate limit or daily usage limit exceeded"},
    },
)
def analyze_async(
    request: AnalyzeRequest,
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
) -> AsyncAnalyzeResponse:
    """Submit text for background analysis. Returns immediately with 202."""
    key_id = _auth_id(auth)
    _check_daily_limits(key_id, len(request.text))

    detector_names = request.detectors or ["unicode"]
    job_id = str(uuid.uuid4())
    input_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()

    repo = _get_repo()
    repo.create_job(
        job_id=job_id,
        key_id=key_id,
        detectors=detector_names,
        config_path=request.config_path,
        input_hash=input_hash,
        character_count=len(request.text),
    )

    # Submit to background executor
    from provenance.api.jobs import get_executor, run_job_background
    executor = get_executor()
    executor.submit(
        run_job_background,
        job_id=job_id,
        text=request.text,
        detector_names=detector_names,
        config_path=request.config_path,
        key_id=key_id,
    )

    return AsyncAnalyzeResponse(
        job_id=job_id,
        status="queued",
        message="Analysis job submitted",
    )


@router.get(
    "/v1/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get job status",
    description=(
        "Get job status and result.  Only the job owner (or admin) can access it. "
        "When status is 'completed', the 'result' field contains the full analysis. "
        "When status is 'failed', the 'error_message' field explains the failure."
    ),
    response_description="Job details with result or error",
    tags=["jobs"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found (or not owned by caller)"},
    },
)
def get_job(
    job_id: str,
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
) -> Response:
    """Get job status and result. Only the job owner (or admin) can access it."""
    key_id = _auth_id(auth)
    repo = _get_repo()
    job = repo.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Ownership check: only the key that created the job (or admin) can see it
    job_key = job.get("key_id", "")
    if key_id != "_admin" and job_key != key_id:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    resp = JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        character_count=job.get("character_count", 0),
        detectors=job.get("detectors", []),
        config_path=job.get("config_path"),
        error_message=job.get("error_message"),
        duration_ms=job.get("duration_ms"),
        result=job.get("result"),
        retry_count=job.get("retry_count", 0),
        parent_job_id=job.get("parent_job_id"),
    )
    return Response(
        content=resp.model_dump_json(),
        media_type="application/json",
        headers=_usage_headers(key_id),
    )


@router.get(
    "/v1/jobs",
    response_model=JobListResponse,
    summary="List jobs",
    description="List jobs for the calling API key.  Paginated.",
    response_description="Paginated list of jobs",
    tags=["jobs"],
    responses={
        400: {"model": ErrorResponse, "description": "Requires a stored API key (not admin or legacy)"},
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    },
)
def list_jobs(
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
) -> JobListResponse:
    """List jobs for the calling key."""
    key_id = _auth_id(auth)
    if key_id in ("_none", "_admin"):
        raise HTTPException(status_code=400, detail="Job listing requires an API key")
    repo = _get_repo()
    summaries, total = repo.list_jobs_by_key(key_id=key_id, limit=limit, offset=offset)
    return JobListResponse(
        jobs=[JobSummary(**s) for s in summaries],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete(
    "/v1/jobs/{job_id}",
    summary="Cancel a job",
    description=(
        "Cancel a queued or running job.  Only the job owner (or admin) can cancel.\n\n"
        "- queued -> cancelled (will not execute)\n"
        "- running -> cancellation_requested (worker checks after analysis)\n"
        "- completed/failed -> HTTP 409 Conflict\n"
        "- cancelled -> HTTP 200 (idempotent)"
    ),
    response_description="Cancellation confirmation",
    tags=["jobs"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Cannot cancel (job already completed/failed)"},
    },
)
def cancel_job(
    job_id: str,
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
) -> dict:
    """Cancel a job. Only the job owner (or admin) can cancel.

    - queued → cancelled (will not execute)
    - running → cancellation_requested (worker checks after analysis)
    - completed/failed/cancelled → 409 Conflict
    """
    key_id = _auth_id(auth)
    repo = _get_repo()
    job = repo.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Ownership check
    job_key = job.get("key_id", "")
    if key_id != "_admin" and job_key != key_id:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    status = job["status"]

    # Already cancelled is idempotent — return 200
    if status == "cancelled":
        return {"job_id": job_id, "status": "cancelled", "message": "Job already cancelled"}

    if status in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job in '{status}' status",
        )

    if status == "queued":
        # Atomically transition queued → cancelled.
        # Race: worker may have moved it to running already.
        # Use a conditional update: only cancel if still queued.
        repo.cancel_job_if_status(job_id, expected_status="queued", new_status="cancelled")
        # Re-read to see what actually happened
        updated_job = repo.get_job(job_id)
        final_status = updated_job["status"] if updated_job else "cancelled"
        if final_status == "cancelled":
            logger.info("Job %s cancelled (was queued)", job_id)
            return {"job_id": job_id, "status": "cancelled", "message": "Queued job cancelled"}
        # Worker raced ahead — treat as running cancellation
        repo.update_job_status(job_id=job_id, status="cancellation_requested")
        logger.info("Job %s cancellation requested (raced to running)", job_id)
        return {"job_id": job_id, "status": "cancellation_requested", "message": "Cancellation requested for running job"}

    if status == "running":
        repo.update_job_status(job_id=job_id, status="cancellation_requested")
        logger.info("Job %s cancellation requested (was running)", job_id)
        return {"job_id": job_id, "status": "cancellation_requested", "message": "Cancellation requested for running job"}

    if status == "cancellation_requested":
        return {"job_id": job_id, "status": "cancellation_requested", "message": "Cancellation already requested"}

    # Should not reach here
    raise HTTPException(status_code=409, detail=f"Cannot cancel job in '{status}' status")


@router.post(
    "/v1/jobs/{job_id}/retry",
    response_model=AsyncAnalyzeResponse,
    status_code=202,
    summary="Retry a failed job",
    description=(
        "Retry a failed job by re-submitting text.  Creates a **new** job linked "
        "to the original via 'parent_job_id'.  Only 'failed' jobs can be retried. "
        "The raw text is never stored — you must re-submit it."
    ),
    response_description="New job submission confirmation",
    tags=["jobs"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        409: {"model": ErrorResponse, "description": "Job is not in a retryable (failed) state"},
        429: {"model": ErrorResponse, "description": "Rate limit or daily usage limit exceeded"},
    },
)
def retry_job(
    job_id: str,
    request: AnalyzeRequest,
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
) -> AsyncAnalyzeResponse:
    """Retry a failed job. Creates a new job with the provided text.

    Only failed jobs can be retried. The new job is linked to the original
    via parent_job_id. Normal rate limits, daily limits, and concurrency
    limits apply.
    """
    key_id = _auth_id(auth)
    repo = _get_repo()
    original_job = repo.get_job(job_id)

    if original_job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Ownership check
    job_key = original_job.get("key_id", "")
    if key_id != "_admin" and job_key != key_id:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if original_job["status"] not in _RETRYABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Only failed jobs can be retried (current status: {original_job['status']})",
        )

    # Check daily limits
    _check_daily_limits(key_id, len(request.text))

    # Determine detectors and config from request (or fallback to original)
    detector_names = request.detectors or original_job.get("detectors", ["unicode"])
    config_path = request.config_path or original_job.get("config_path")
    input_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()
    retry_count = original_job.get("retry_count", 0) + 1

    new_job_id = str(uuid.uuid4())
    repo.create_job(
        job_id=new_job_id,
        key_id=key_id,
        detectors=detector_names,
        config_path=config_path,
        input_hash=input_hash,
        character_count=len(request.text),
        retry_count=retry_count,
        parent_job_id=job_id,
    )

    # Submit to background executor
    from provenance.api.jobs import get_executor, run_job_background
    executor = get_executor()
    executor.submit(
        run_job_background,
        job_id=new_job_id,
        text=request.text,
        detector_names=detector_names,
        config_path=config_path,
        key_id=key_id,
    )

    logger.info(
        "Job %s retried as %s (retry_count=%d)",
        job_id, new_job_id, retry_count,
    )

    return AsyncAnalyzeResponse(
        job_id=new_job_id,
        status="queued",
        message=f"Retry submitted (attempt {retry_count})",
    )


# ---------------------------------------------------------------------------
# GET /v1/analyses/{analysis_id}
# ---------------------------------------------------------------------------


@router.get(
    "/v1/analyses/{analysis_id}",
    summary="Get analysis by ID",
    description="Retrieve a previously persisted analysis by its ID.",
    response_description="Full analysis record",
    tags=["analysis"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        404: {"model": ErrorResponse, "description": "Analysis not found"},
    },
)
def get_analysis(analysis_id: str, auth: RequireAPIKey = None, _rl: None = Depends(rate_limit_dependency)) -> Response:
    key_id = _auth_id(auth)
    start_time = time.monotonic()
    repo = _get_repo()
    record = repo.get(analysis_id)
    duration_ms = round((time.monotonic() - start_time) * 1000, 2)
    if record is None:
        repo.record_usage(
            key_id=key_id,
            endpoint="/v1/analyses/{id}",
            status_code=404,
            duration_ms=duration_ms,
            character_count=0,
            success=False,
        )
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    repo.record_usage(
        key_id=key_id,
        endpoint="/v1/analyses/{id}",
        status_code=200,
        duration_ms=duration_ms,
        character_count=0,
        success=True,
    )
    headers = _usage_headers(key_id)
    return Response(
        content=json.dumps(record),
        media_type="application/json",
        headers=headers,
    )


def _get_max_list_limit() -> int:
    return int(os.environ.get("MAX_LIST_LIMIT", "100"))


@router.get(
    "/v1/analyses",
    response_model=AnalysisListResponse,
    summary="List analyses",
    description=(
        "List persisted analyses.  Paginated with 'limit' and 'offset'. "
        "The 'limit' parameter is capped by MAX_LIST_LIMIT (default 100)."
    ),
    response_description="Paginated list of analyses",
    tags=["analysis"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
        422: {"model": ErrorResponse, "description": "limit exceeds MAX_LIST_LIMIT"},
    },
)
def list_analyses(
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
) -> AnalysisListResponse:
    key_id = _auth_id(auth)
    max_limit = _get_max_list_limit()
    if limit > max_limit:
        raise HTTPException(status_code=422, detail=f"limit must be <= {max_limit}")
    start_time = time.monotonic()
    repo = _get_repo()
    summaries, total = repo.list_analyses(limit=limit, offset=offset)
    duration_ms = round((time.monotonic() - start_time) * 1000, 2)
    repo.record_usage(
        key_id=key_id,
        endpoint="/v1/analyses",
        status_code=200,
        duration_ms=duration_ms,
        character_count=0,
        success=True,
    )
    return AnalysisListResponse(
        analyses=[AnalysisSummary(**s) for s in summaries],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/v1/usage",
    response_model=UsageResponse,
    summary="Usage statistics",
    description=(
        "Return aggregate usage statistics for the calling API key. "
        "Includes total requests, characters analyzed, and breakdowns "
        "by endpoint and detector.  Only your own usage is visible."
    ),
    response_description="Usage statistics for the calling key",
    tags=["usage"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    },
)
def get_usage(
    auth: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
) -> UsageResponse:
    """Return aggregate usage statistics for the calling key."""
    key_id = _auth_id(auth)
    repo = _get_repo()
    summary = repo.get_usage_summary(key_id=key_id if key_id not in ("_none", "_admin") else None)
    return UsageResponse(**summary)


# ---------------------------------------------------------------------------
# Usage headers helper
# ---------------------------------------------------------------------------


def _usage_headers(key_id: str) -> dict[str, str]:
    """Build usage-limit headers from the database for the given key."""
    headers: dict[str, str] = {}
    req_limit_str = os.environ.get("PROVENANCE_DAILY_REQUEST_LIMIT")
    char_limit_str = os.environ.get("PROVENANCE_DAILY_CHARACTER_LIMIT")
    if not req_limit_str and not char_limit_str:
        return headers
    try:
        repo = _get_repo()
        today_reqs, today_chars = repo.get_usage_today(key_id)
        if req_limit_str:
            headers["X-Usage-Request-Limit"] = req_limit_str
            remaining = max(0, int(req_limit_str) - today_reqs)
            headers["X-Usage-Requests-Remaining"] = str(remaining)
        if char_limit_str:
            headers["X-Usage-Character-Limit"] = char_limit_str
            remaining = max(0, int(char_limit_str) - today_chars)
            headers["X-Usage-Characters-Remaining"] = str(remaining)
    except Exception:
        pass  # Don't let header generation break the response
    return headers


# ---------------------------------------------------------------------------
# API Key Management (admin-only)
# ---------------------------------------------------------------------------


@router.post(
    "/v1/api-keys",
    response_model=CreateApiKeyResponse,
    summary="Create API key",
    description=(
        "Create a new API key.  The raw secret is returned **only in this response** "
        "and cannot be recovered afterward.  Requires admin key."
    ),
    response_description="New API key (raw secret shown once)",
    tags=["api-keys"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid admin key"},
        403: {"model": ErrorResponse, "description": "Admin key management not configured"},
    },
)
def create_api_key(
    request: CreateApiKeyRequest,
    auth: RequireAdminKey = None,
) -> CreateApiKeyResponse:
    """Create a new API key. The raw secret is returned only in this response."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    key_id = generate_key_id()
    repo = _get_repo()
    repo.create_api_key(key_id=key_id, key_hash=key_hash, name=request.name)

    # Record when the key was last used (at creation time)
    repo.update_api_key_last_used(key_id)

    logger.info("API key created: key_id=%s name=%s", key_id, request.name)

    return CreateApiKeyResponse(
        key_id=key_id,
        name=request.name,
        status="active",
        created_at=repo.get_api_key_by_hash(key_hash)["created_at"],
        key=key,
    )


@router.get(
    "/v1/api-keys",
    response_model=ApiKeyListResponse,
    summary="List API keys",
    description="List all API keys.  Admin only.  Never returns raw secrets.",
    response_description="List of API key summaries",
    tags=["api-keys"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid admin key"},
        403: {"model": ErrorResponse, "description": "Admin key management not configured"},
    },
)
def list_api_keys(
    auth: RequireAdminKey = None,
) -> ApiKeyListResponse:
    """List all API keys (admin only). Never returns raw secrets."""
    repo = _get_repo()
    keys = repo.list_api_keys()
    return ApiKeyListResponse(
        keys=[ApiKeySummary(**k) for k in keys],
        total=len(keys),
    )


@router.delete(
    "/v1/api-keys/{key_id}",
    summary="Revoke API key",
    description=(
        "Revoke (deactivate) an API key.  Admin only. "
        "The key record remains for audit/usage history."
    ),
    response_description="Revocation confirmation",
    tags=["api-keys"],
    responses={
        401: {"model": ErrorResponse, "description": "Missing or invalid admin key"},
        403: {"model": ErrorResponse, "description": "Admin key management not configured"},
        404: {"model": ErrorResponse, "description": "API key not found"},
    },
)
def revoke_api_key(
    key_id: str,
    auth: RequireAdminKey = None,
) -> dict:
    """Revoke an API key (admin only). The key is deactivated, not deleted."""
    repo = _get_repo()
    # Verify the key exists before revoking
    existing = repo.get_api_key_by_id(key_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"API key {key_id} not found")
    repo.revoke_api_key(key_id)
    logger.info("API key revoked: key_id=%s", key_id)
    return {"key_id": key_id, "status": "revoked"}
