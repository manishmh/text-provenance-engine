"""Configuration validation for the provenance HTTP API.

Validates operational environment variables at startup so invalid values
fail clearly rather than causing undefined behavior at runtime.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("provenance.api")


class ConfigError(Exception):
    """Raised when an environment variable has an invalid value."""


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
