"""API key authentication for the provenance HTTP API.

Supports three auth modes:

1. **Legacy mode**: ``PROVENANCE_API_KEY`` env var — a single global key.
   When unset, authentication is disabled (local dev).
2. **Managed mode**: keys stored in the database via the repository.
   Checked first; falls back to the legacy env var.
3. **Admin mode**: ``PROVENANCE_ADMIN_API_KEY`` env var — a separate key
   that grants access to API-key management endpoints.

The raw API key is never stored or logged.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException

from provenance.api.keys import get_legacy_key_hash, hash_api_key, validate_api_key
from provenance.api.state import get_repo


def _has_any_auth_configured() -> bool:
    """Return True if any authentication method is configured."""
    return (
        os.environ.get("PROVENANCE_API_KEY") is not None
        or os.environ.get("PROVENANCE_ADMIN_API_KEY") is not None
    )


def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> Any:
    """FastAPI dependency: verify the ``X-API-Key`` header.

    SaaS mode: when Supabase Auth is configured, a valid Supabase bearer
    token is accepted as an alternative credential (source ``supabase``),
    and anonymous access to ``/v1/*`` is denied.  When Supabase is not
    configured, behavior is exactly the legacy behavior below.

    Returns a key identity dict ``{"id": ..., "source": ...}`` on success.
    Sources: ``stored``, ``legacy``, ``admin``, ``supabase``, ``none``.
    """
    from provenance.api.supabase_auth import supabase_configured, verify_supabase_token

    if x_api_key is None and supabase_configured():
        # SaaS deployments: API keys are for service/admin use; product
        # users authenticate with Supabase sessions.  Never trust a
        # frontend-supplied user ID — identity comes from the verified JWT.
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                claims = verify_supabase_token(token.strip())
                return {
                    "id": f"supabase:{claims.get('sub', '')}",
                    "source": "supabase",
                    "auth_user_id": str(claims.get("sub", "")),
                    "email": claims.get("email") if isinstance(claims.get("email"), str) else None,
                }
        raise HTTPException(status_code=401, detail="Sign-in required")

    if not _has_any_auth_configured() and x_api_key is None:
        return {"id": "_none", "source": "none"}

    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    # Check admin key first (admin keys can also access normal endpoints)
    admin_key = os.environ.get("PROVENANCE_ADMIN_API_KEY")
    if admin_key and x_api_key == admin_key:
        return {"id": "_admin", "source": "admin"}

    try:
        repo = get_repo()
        is_valid, key_id = validate_api_key(x_api_key, repo)
    except RuntimeError:
        expected = os.environ.get("PROVENANCE_API_KEY")
        if expected and x_api_key == expected:
            return {"id": "_legacy", "source": "legacy"}
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not is_valid:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if key_id and key_id.startswith("_"):
        # Legacy key from validate_api_key
        return {"id": "_legacy", "source": "legacy"}
    return {"id": key_id, "source": "stored"}


def require_admin_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """FastAPI dependency: require the admin API key.

    Returns ``{"id": "_admin", "source": "admin"}`` on success.
    """
    admin_key = os.environ.get("PROVENANCE_ADMIN_API_KEY")
    if admin_key is None:
        # Admin endpoint not configured — deny access
        raise HTTPException(
            status_code=403,
            detail="Admin API key management is not configured",
        )
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if x_api_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    return {"id": "_admin", "source": "admin"}


# Type aliases for route signatures.
RequireAPIKey = Annotated[dict, Depends(require_api_key)]
RequireAdminKey = Annotated[dict, Depends(require_admin_key)]
