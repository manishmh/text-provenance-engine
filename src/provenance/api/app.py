"""FastAPI application factory for the provenance HTTP API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from provenance.api.db import AnalysisRepository
from provenance.api.routes import configure_repo, router


def create_app(db_path: str | Path = "data/provenance.db") -> FastAPI:
    """Build the FastAPI application with SQLite persistence."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        repo = AnalysisRepository(db_path)
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
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
