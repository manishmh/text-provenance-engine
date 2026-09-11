"""Middleware and dependencies for the provenance HTTP API.

Provides:
- Rate limiting (in-memory, configurable via env vars)
- X-Request-ID generation / propagation
- Structured request logging
- CORS configuration
- Global exception handler (sanitized JSON errors)
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default=""        )


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("provenance.api")

# Fields that must NEVER appear in logs
_SENSITIVE_KEYS = frozenset({
    "x-api-key", "authorization", "password", "secret",
    "database_url", "dsn", "text", "raw_text",
})


def trust_proxy_headers() -> bool:
    """Whether the deployment's reverse proxy is trusted to set X-Forwarded-For."""
    return os.environ.get("PROVENANCE_TRUST_PROXY", "0").strip().lower() in {"1", "true", "yes"}


def request_client_ip(request: Request) -> str:
    """Return client IP, trusting forwarding headers only after explicit opt-in."""
    client_ip = request.client.host if request.client else "unknown"
    if trust_proxy_headers():
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
    return client_ip


def _sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of headers with sensitive values redacted."""
    out = {}
    for k, v in headers.items():
        if k.lower() in _SENSITIVE_KEYS:
            out[k] = "[REDACTED]"
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate or accept X-Request-ID for every request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request_id_var.set(rid)
        request.state.request_id = rid

        try:
            response = await call_next(request)
        except Exception:
            # Re-raise but the request_id is already set on request.state
            raise
        response.headers["X-Request-ID"] = rid
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Reject declared oversized request bodies before JSON parsing or model work."""

    async def dispatch(self, request: Request, call_next) -> Response:
        raw_size = request.headers.get("content-length")
        if raw_size:
            try:
                size = int(raw_size)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
            try:
                maximum = int(os.environ.get("PROVENANCE_MAX_REQUEST_BODY_BYTES", "1048576"))
            except ValueError:
                maximum = 1_048_576
            if size > max(1024, maximum):
                rid = getattr(request.state, "request_id", "unknown")
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body exceeds the service limit", "request_id": rid},
                )
        return await call_next(request)


# ---------------------------------------------------------------------------
# Structured logging middleware
# ---------------------------------------------------------------------------


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, request ID, and duration for every request.

    Never logs: raw text, API keys, watermark keys, DATABASE_URL credentials.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.monotonic()
        rid = getattr(request.state, "request_id", "unknown")

        try:
            response = await call_next(request)
        except HTTPException:
            # Let HTTPExceptions propagate — FastAPI handles them
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            logger.info(
                "%s %s %d %.2fms rid=%s",
                request.method, request.url.path, 500, duration_ms, rid,
            )
            raise
        except Exception as exc:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            logger.error(
                "%s %s 500 %.2fms rid=%s error_type=%s",
                request.method, request.url.path, duration_ms, rid, type(exc).__name__,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": rid},
            )

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "%s %s %d %.2fms rid=%s",
            request.method, request.url.path, response.status_code, duration_ms, rid,
        )
        return response


# ---------------------------------------------------------------------------
# Rate limiting (in-memory sliding window)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Simple per-IP sliding-window rate limiter (no external deps)."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = {}

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        hits = self._hits.get(key, [])
        # Prune old entries
        hits = [t for t in hits if t > cutoff]
        self._hits[key] = hits

        if len(hits) >= self.max_requests:
            retry = int(self.window_seconds - (now - hits[0])) + 1
            return False, max(retry, 1)

        hits.append(now)
        return True, 0


def _get_rate_limit_config() -> tuple[int, int]:
    """Read rate limit config from environment.

    Returns (max_requests, window_seconds).
    """
    max_req = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "60"))
    window = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
    return max(1, max_req), max(1, window)


_limiter: _RateLimiter | None = None


def _get_limiter() -> _RateLimiter:
    global _limiter
    if _limiter is None:
        mw, ws = _get_rate_limit_config()
        _limiter = _RateLimiter(mw, ws)
    return _limiter


def rate_limit_dependency(request: Request) -> None:
    """FastAPI dependency: enforce rate limits on /v1/* endpoints."""
    limiter = _get_limiter()
    client_ip = request_client_ip(request)

    allowed, retry_after = limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )





# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


def configure_cors(app: FastAPI) -> None:
    """Add CORS middleware configured via CORS_ORIGINS env var.

    Default is restrictive: no origins allowed (same-origin only).
    Set CORS_ORIGINS to a comma-separated list to enable cross-origin.
    """
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        # Restrictive default: disallow all cross-origin requests
        allow_origins: list[str] = []
    else:
        allow_origins = [o.strip() for o in raw.split(",") if o.strip()]

    if "*" in allow_origins:
        raise RuntimeError("CORS_ORIGINS must not use wildcard when browser credentials are enabled")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=bool(allow_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


def install_exception_handler(app: FastAPI) -> None:
    """Install a handler that returns clean JSON for unexpected errors.

    Does NOT expose stack traces, credentials, DB connection strings,
    or internal secrets.  HTTPException (401/403/404/422/429) is left
    to FastAPI's built-in handler.
    """

    @app.exception_handler(Exception)
    async def _handle_exception(request: Request, exc: Exception) -> JSONResponse:
        # Let FastAPI/Starlette handle HTTPExceptions normally
        if isinstance(exc, HTTPException):
            raise exc  # type: ignore[misc]
        rid = getattr(request.state, "request_id", "unknown")
        logger.error("Unhandled exception rid=%s error_type=%s", rid, type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": rid,
            },
        )
