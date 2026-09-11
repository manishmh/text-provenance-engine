"""Configuration validation for the provenance HTTP API.

Validates operational environment variables at startup so invalid values
fail clearly rather than causing undefined behavior at runtime.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger("provenance.api")


class ConfigError(Exception):
    """Raised when an environment variable has an invalid value."""


def deployment_environment() -> str:
    """Return the explicit deployment tier (development by default)."""
    value = os.environ.get("PROVENANCE_ENVIRONMENT", "development").strip().lower()
    if value not in {"development", "staging", "production"}:
        raise ConfigError("PROVENANCE_ENVIRONMENT must be development, staging, or production")
    return value


def serverless_runtime() -> bool:
    """Whether this process is running on an ephemeral serverless runtime.

    Vercel supplies ``VERCEL=1`` to Functions.  The explicit provenance
    setting keeps the behavior easy to exercise in staging tests without
    coupling production code to a platform-specific import.
    """
    configured = os.environ.get("PROVENANCE_RUNTIME", "").strip().lower()
    return configured in {"serverless", "vercel"} or os.environ.get("VERCEL") == "1"


def validate_config() -> None:
    """Validate all operational environment variables.

    Raises ConfigError with a clear message on the first invalid value found.
    Safe to call at application startup.
    """
    _validate_int_env("PROVENANCE_MAX_BACKGROUND_JOBS", min_val=1, default=2)
    _validate_int_env("PROVENANCE_JOB_RETENTION_HOURS", min_val=1, default=24)
    _validate_int_env("RATE_LIMIT_MAX_REQUESTS", min_val=1, default=60)
    _validate_int_env("RATE_LIMIT_WINDOW_SECONDS", min_val=1, default=60)
    _validate_int_env("MAX_LIST_LIMIT", min_val=1, default=100)
    _validate_int_env("PROVENANCE_DAILY_REQUEST_LIMIT", min_val=1, required=False)
    _validate_int_env("PROVENANCE_DAILY_CHARACTER_LIMIT", min_val=1, required=False)
    _validate_int_env("PROVENANCE_PUBLIC_DAILY_LIMIT", min_val=1, default=2)
    _validate_int_env("PROVENANCE_PUBLIC_MAX_CHARS", min_val=1, default=5000)
    _validate_int_env("PROVENANCE_FREE_DAILY_LIMIT", min_val=1, default=50)
    _validate_int_env("PROVENANCE_FREE_MAX_CHARS", min_val=1, default=20000)
    _validate_int_env("PROVENANCE_PRO_DAILY_LIMIT", min_val=1, default=1000)
    _validate_int_env("PROVENANCE_PRO_MAX_CHARS", min_val=1, default=100000)
    _validate_int_env("PROVENANCE_MAX_REQUEST_BODY_BYTES", min_val=1024, default=1_048_576)
    _validate_bool_env("PROVENANCE_TRUST_PROXY", default=False)
    _validate_bool_env("PROVENANCE_ANON_COOKIE_SECURE", default=False)
    _validate_cors_origins()

    env = deployment_environment()
    same_site = os.environ.get("PROVENANCE_COOKIE_SAMESITE", "lax").strip().lower()
    if same_site not in {"lax", "strict", "none"}:
        raise ConfigError("PROVENANCE_COOKIE_SAMESITE must be lax, strict, or none")
    secure_cookie = _bool_env("PROVENANCE_ANON_COOKIE_SECURE", default=False)
    if same_site == "none" and not secure_cookie:
        raise ConfigError("PROVENANCE_COOKIE_SAMESITE=none requires PROVENANCE_ANON_COOKIE_SECURE=1")
    if env == "production":
        secret = os.environ.get("PROVENANCE_ANON_COOKIE_SECRET", "").strip()
        if len(secret) < 32:
            raise ConfigError("PROVENANCE_ANON_COOKIE_SECRET must be at least 32 characters in production")
        if not secure_cookie:
            raise ConfigError("PROVENANCE_ANON_COOKIE_SECURE must be enabled in production")
        if not os.environ.get("DATABASE_URL", "").strip().startswith(("postgresql://", "postgres://")):
            raise ConfigError("DATABASE_URL must use PostgreSQL in production")
        if not os.environ.get("CORS_ORIGINS", "").strip():
            raise ConfigError("CORS_ORIGINS must name explicit HTTPS frontend origins in production")
        if not os.environ.get("SUPABASE_URL", "").strip():
            raise ConfigError("SUPABASE_URL must be configured in production")


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes"}


def _validate_bool_env(name: str, *, default: bool) -> None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return
    if raw.strip().lower() not in {"0", "1", "true", "false", "yes", "no"}:
        raise ConfigError(f"{name} must be a boolean")


def _validate_cors_origins() -> None:
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        return
    env = deployment_environment()
    for origin in (value.strip() for value in raw.split(",") if value.strip()):
        parsed = urlparse(origin)
        if origin == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ConfigError("CORS_ORIGINS must contain explicit origin URLs, never wildcard or paths")
        if env == "production" and (parsed.scheme != "https" or parsed.hostname in {"localhost", "127.0.0.1"}):
            raise ConfigError("CORS_ORIGINS must use non-localhost HTTPS origins in production")


def _validate_int_env(
    name: str,
    *,
    min_val: int = 1,
    default: int | None = None,
    required: bool = False,
) -> None:
    """Validate that an environment variable is a positive integer.

    Parameters
    ----------
    name:
        Environment variable name.
    min_val:
        Minimum allowed value (inclusive).
    default:
        Value to use when the variable is unset.  None means skip validation
        when the variable is absent.
    required:
        If True, the variable must be set.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if required:
            raise ConfigError(f"{name} must be set")
        if default is not None:
            logger.debug("Using default %s=%s", name, default)
        return

    try:
        value = int(raw.strip())
    except ValueError:
        raise ConfigError(
            f"{name} must be a positive integer, got: {raw!r}"
        ) from None

    if value < min_val:
        raise ConfigError(
            f"{name} must be >= {min_val}, got: {value}"
        )
