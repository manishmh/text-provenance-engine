"""FastAPI application factory for the provenance HTTP API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from provenance.api.db import create_repository
from provenance.api.middleware import (
    LoggingMiddleware,
    RequestIDMiddleware,
    configure_cors,
    install_exception_handler,
)
from provenance.api.routes import configure_repo, router


def create_app(db_url: str | None = None) -> FastAPI:
    """Build the FastAPI application with pluggable persistence.

    Parameters
    ----------
    db_url:
        Connection string overriding ``DATABASE_URL``.  ``None`` means use
        the environment variable (or default SQLite).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        repo = create_repository(db_url)
        configure_repo(repo)
        yield
        repo.close()

    app = FastAPI(
        title="Text Provenance Engine",
        description=(
            "Local REST API for detecting text-level provenance signals. "
            "Reports deterministic Unicode artifacts and evidence for known "
            "watermark configurations. Does NOT classify text as AI-generated."
        ),
        version="0.3.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: outermost = first applied)
    configure_cors(app)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Exception handler
    install_exception_handler(app)

    app.include_router(router)
    return app


app = create_app()
