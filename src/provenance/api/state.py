"""Shared application state for the provenance HTTP API.

Holds the repository reference so both auth and routes can access it
without circular imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from provenance.api.db import AnalysisRepository

_repo: AnalysisRepository | None = None


def get_repo() -> AnalysisRepository:
    """Get the configured repository. Raises RuntimeError if not set."""
    if _repo is None:
        raise RuntimeError("AnalysisRepository not configured")
    return _repo


def set_repo(repo: AnalysisRepository) -> None:
    """Set the repository reference (called at app startup)."""
    global _repo
    _repo = repo
