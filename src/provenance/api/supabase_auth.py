"""Supabase Auth integration (verification side only).

The frontend uses the Supabase JS client for sign-in (email/password,
magic link, Google OAuth).  This backend never handles passwords: it only
verifies Supabase-issued JWTs (HS256, signed with the project's JWT
secret) and derives a stable ``auth_user_id`` (the ``sub`` claim).

Required backend configuration:
- ``SUPABASE_URL`` — used to check the ``iss`` claim.
- ``SUPABASE_JWT_SECRET`` — HMAC secret used to verify signatures.

Frontend configuration (``dashboard/.env`` — never the backend):
- ``VITE_SUPABASE_URL``, ``VITE_SUPABASE_ANON_KEY``.
"""
from __future__ import annotations

import os
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException


def supabase_configured() -> bool:
    return bool(os.environ.get("SUPABASE_URL", "").strip()) and bool(
        os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    )


def verify_supabase_token(token: str) -> dict[str, Any]:
    """Verify a Supabase access token; return its claims.

    Raises HTTPException 401 on any failure (invalid, expired, wrong
    issuer/audience).  Never trusts frontend-supplied user IDs — the
    identity always comes from the verified ``sub`` claim.
    """
    secret = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    base_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    if not secret or not base_url:
        raise HTTPException(status_code=503, detail="Authentication is not configured")
    options_issuer = f"{base_url}/auth/v1"
    try:
        # Supabase user sessions carry aud="authenticated".
        claims = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            issuer=options_issuer,
            audience="authenticated",
            options={"require": ["exp", "iss", "sub"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired; please sign in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session token")
    return claims


async def optional_supabase_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any] | None:
    """Return verified Supabase claims, or None when no bearer token."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return verify_supabase_token(token.strip())


OptionalSupabaseUser = Depends(optional_supabase_user)


async def require_supabase_user(
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Require a valid Supabase session; 401 otherwise."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Sign-in required")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="Sign-in required")
    return verify_supabase_token(token.strip())


RequireSupabaseUser = Depends(require_supabase_user)
