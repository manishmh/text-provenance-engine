"""FastAPI application factory for the provenance HTTP API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from provenance.api.db import create_repository
from provenance.api.middleware import (
    LoggingMiddleware,
    RequestIDMiddleware,
    install_exception_handler,
    rate_limit_dependency,
)
from provenance.api.routes import configure_repo, router
from provenance.api.state import set_repo

logger = logging.getLogger("provenance.api")


def _validate_startup_config() -> None:
    """Validate configuration at startup. Logs warning on invalid values."""
    try:
        from provenance.api.config import validate_config
        validate_config()
    except Exception as exc:
        logger.warning("Configuration validation failed: %s", exc)


def create_app(db_url: str | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Parameters
    ----------
    db_url:
        Optional database URL override.  When *None*, the factory reads
        ``DATABASE_URL`` from the environment (or defaults to SQLite).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Phase 3E: Validate configuration at startup
        _validate_startup_config()

        repo = create_repository(db_url)
        set_repo(repo)
        configure_repo(repo)

        # Phase 3D: Recover jobs left in running/queued state from a previous process
        try:
            recovered = repo.recover_stale_jobs()
            if recovered:
                logger.info("Recovered %d stale jobs on startup", recovered)
        except Exception:
            logger.debug("Stale job recovery skipped", exc_info=True)

        # Phase 6C: Recover benchmark runs interrupted by a previous process.
        # Interrupted runs are marked failed (never successful); artifacts on
        # disk are preserved and retry resumes via the run manifest.
        try:
            recovered_runs = repo.recover_stale_benchmark_runs()
            if recovered_runs:
                logger.info("Recovered %d stale benchmark runs on startup", recovered_runs)
        except Exception:
            logger.debug("Stale benchmark run recovery skipped", exc_info=True)

        # Opportunistic job cleanup on startup
        try:
            from provenance.api.jobs import cleanup_old_jobs
            cleaned = cleanup_old_jobs(repo)
            if cleaned:
                logger.info("Cleaned up %d old jobs on startup", cleaned)
        except Exception:
            pass

        yield

        # Phase 3E: Graceful shutdown — stop accepting new jobs,
        # allow running jobs to finish, persist final state
        try:
            from provenance.api.jobs import shutdown_executor
            shutdown_executor(wait=True)
            logger.info("Background executor shut down cleanly")
        except Exception:
            logger.debug("Executor shutdown skipped", exc_info=True)

        # Phase 6B: release cached models/tokenizers so the process does
        # not retain gigabytes of weights after serving stops.
        try:
            from provenance.loading import clear_caches
            released = clear_caches()
            logger.info("Model/tokenizer caches released: %s", released)
        except Exception:
            logger.debug("Cache release skipped", exc_info=True)

        repo.close()

    app = FastAPI(
        title="Text Provenance Engine",
        version="0.1.0",
        description=(
            "REST API for detecting text-level provenance signals. "
            "Reports deterministic Unicode artifacts and evidence for known "
            "watermark configurations (KGW, SynthID). "
            "Does NOT classify text as AI-generated."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {
                "name": "operations",
                "description": (
                    "Health, readiness, and metrics endpoints. "
                    "No authentication required."
                ),
            },
            {
                "name": "analysis",
                "description": (
                    "Synchronous text analysis and result retrieval. "
                    "Requires API key (``X-API-Key`` header)."
                ),
            },
            {
                "name": "jobs",
                "description": (
                    "Background analysis jobs: submit, poll, cancel, and retry. "
                    "Requires API key (``X-API-Key`` header)."
                ),
            },
            {
                "name": "usage",
                "description": "Usage statistics for the calling API key.",
            },
            {
                "name": "api-keys",
                "description": "API key management. Requires admin key (``PROVENANCE_ADMIN_API_KEY``).",
            },
            {
                "name": "detectors",
                "description": (
                    "Detector discovery: list supported detectors and capabilities. "
                    "Requires API key (``X-API-Key`` header)."
                ),
            },
            {
                "name": "robustness",
                "description": (
                    "Stored robustness benchmark results and cross-model "
                    "comparison for dashboard visualization. Read-only. "
                    "Requires API key (``X-API-Key`` header)."
                ),
            },
            {
                "name": "benchmark-runs",
                "description": (
                    "Dashboard-triggered benchmark execution: configure, "
                    "queue, monitor, cancel, and retry robustness benchmark "
                    "runs. Requires API key (``X-API-Key`` header)."
                ),
            },
        ],
    )

    # Middleware (order matters — first added = outermost)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Exception handling
    install_exception_handler(app)

    # Rate limiting dependency
    app.dependency_overrides[rate_limit_dependency] = rate_limit_dependency

    # CORS (disabled by default — set CORS_ORIGINS to enable)
    from provenance.api.middleware import configure_cors
    configure_cors(app)

    # Routes
    app.include_router(router)

    return app
