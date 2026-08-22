"""API key authentication for the provenance HTTP API.

Reads the expected key from the ``PROVENANCE_API_KEY`` environment variable.
When the variable is unset or empty, authentication is **disabled** (local
development convenience).  In production, always set the variable.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, Header, HTTPException


def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """FastAPI dependency: verify the ``X-API-Key`` header.

    Raises
    ------
    HTTPException
        401 if the header is missing and auth is enabled.
        403 if the header value does not match.
    """
    expected = os.environ.get("PROVENANCE_API_KEY") or None
    if expected is None:
        # No key configured — allow everything (local dev).
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


# Type alias for route signatures.
RequireAPIKey = Annotated[None, Depends(require_api_key)]
