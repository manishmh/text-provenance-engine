"""Background job executor for expensive provenance analysis.

Provides a local ThreadPoolExecutor-based background worker for async
analysis jobs.  NOT a distributed job queue — single-process only.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from provenance.api.db import AnalysisRepository

logger = logging.getLogger("provenance.api")

# Module-level executor (initialized lazily)
_executor: ThreadPoolExecutor | None = None

# Graceful shutdown flag — checked before accepting new jobs
_shutting_down: bool = False

# Jobs in these statuses should not be executed
_CANCELLATION_STATUSES = frozenset({"cancelled", "cancellation_requested"})


def _get_max_workers() -> int:
    """Read PROVENANCE_MAX_BACKGROUND_JOBS from environment."""
    return max(1, int(os.environ.get("PROVENANCE_MAX_BACKGROUND_JOBS", "2")))


def get_executor() -> ThreadPoolExecutor:
    """Get or create the background executor."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=_get_max_workers())
    return _executor


def is_shutting_down() -> bool:
    """Return True if the application is shutting down."""
    return _shutting_down


def shutdown_executor(wait: bool = True) -> None:
    """Begin graceful shutdown: set flag, then wait for running jobs."""
    global _executor, _shutting_down
    _shutting_down = True
    if _executor is not None:
        # Wait for running jobs to finish; no new submissions accepted
        _executor.shutdown(wait=wait)
        _executor = None


def run_job_background(
    job_id: str,
    text: str,
    detector_names: list[str],
    config_path: str | None,
    key_id: str,
) -> None:
    """Execute an analysis job in the background.

    This runs in a worker thread.  Updates the job status in the database.
    Checks for cancellation before and after analysis.
    """
    import json as _json

    from provenance.api.service import run_analysis
    from provenance.api.state import get_repo

    repo = get_repo()

    # Pre-check: job may have been cancelled before worker picked it up
    current_job = repo.get_job(job_id)
    if current_job is None:
        logger.warning("Job %s disappeared before execution", job_id)
        return
    if current_job["status"] in _CANCELLATION_STATUSES:
        logger.info("Job %s was cancelled before execution started", job_id)
        return

    # Mark as running
    repo.update_job_status(
        job_id=job_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    start_time = time.monotonic()

    try:
        result_dict = run_analysis(text, detector_names, config_path)
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)

        char_count = result_dict.get("text_stats", {}).get("character_count", len(text))

        # Post-check: was cancellation requested during analysis?
        post_job = repo.get_job(job_id)
        if post_job and post_job["status"] == "cancellation_requested":
            # Mark as cancelled instead of completed
            repo.update_job_status(job_id=job_id, status="cancelled")
            repo.fail_job(
                job_id=job_id,
                error_message="Job cancelled by user",
                duration_ms=duration_ms,
            )
            # Override the status that fail_job set
            repo.update_job_status(job_id=job_id, status="cancelled")
            logger.info("Job %s cancelled after analysis completed", job_id)
            return

        repo.complete_job(
            job_id=job_id,
            result_json=_json.dumps(result_dict),
            duration_ms=duration_ms,
        )

        # Record usage (exactly once per job)
        repo.record_usage(
            key_id=key_id,
            endpoint="/v1/analyze/async",
            status_code=200,
            duration_ms=duration_ms,
            character_count=char_count,
            success=True,
            detector=detector_names[0] if len(detector_names) == 1 else None,
        )

        logger.info("Job %s completed in %.1fms", job_id, duration_ms)

    except Exception as exc:
        duration_ms = round((time.monotonic() - start_time) * 1000, 2)
        error_msg = type(exc).__name__ + ": " + str(exc)
        # Sanitize error — don't expose internals
        safe_error = _sanitize_error(error_msg)

        # Check if cancellation was requested
        post_job = repo.get_job(job_id)
        if post_job and post_job["status"] == "cancellation_requested":
            repo.update_job_status(job_id=job_id, status="cancelled")
            repo.fail_job(
                job_id=job_id,
                error_message="Job cancelled by user",
                duration_ms=duration_ms,
            )
            repo.update_job_status(job_id=job_id, status="cancelled")
            logger.info("Job %s cancelled (error was: %s)", job_id, safe_error)
            return

        repo.fail_job(
            job_id=job_id,
            error_message=safe_error,
            duration_ms=duration_ms,
        )

        # Record usage for failed job too
        repo.record_usage(
            key_id=key_id,
            endpoint="/v1/analyze/async",
            status_code=500,
            duration_ms=duration_ms,
            character_count=0,
            success=False,
        )

        logger.warning("Job %s failed: %s", job_id, safe_error)


def _sanitize_error(msg: str) -> str:
    """Remove sensitive info from error messages."""
    # Remove potential connection strings
    import re
    msg = re.sub(r'postgresql://[^\s]+@[^\s]+/\S+', '[REDACTED]', msg)
    msg = re.sub(r'Password=[^\s]+', 'Password=[REDACTED]', msg)
    # Truncate very long messages
    if len(msg) > 500:
        msg = msg[:500] + "..."
    return msg


def cleanup_old_jobs(repo: AnalysisRepository) -> int:
    """Remove completed/failed/cancelled jobs older than retention period.

    Returns the number of jobs deleted.
    """
    retention_hours = int(os.environ.get("PROVENANCE_JOB_RETENTION_HOURS", "24"))
    cutoff = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    from datetime import timedelta
    cutoff = cutoff - timedelta(hours=retention_hours)
    cutoff_str = cutoff.isoformat()

    return repo.cleanup_jobs(older_than=cutoff_str)
