"""SDK exception hierarchy.

Maps HTTP status codes to typed exceptions so callers can handle errors
without inspecting raw HTTP responses.
"""

from __future__ import annotations


class ProvenanceAPIError(Exception):
    """Base exception for all SDK errors.

    Attributes:
        status_code: HTTP status code (or 0 for transport errors).
        detail: Human-readable error message from the API.
        request_id: Request ID from the response headers, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        detail: str = "",
        request_id: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail or message
        self.request_id = request_id
        super().__init__(message)


class AuthenticationError(ProvenanceAPIError):
    """HTTP 401 — Missing or invalid API key."""


class ForbiddenError(ProvenanceAPIError):
    """HTTP 403 — Invalid API key or admin access required."""


class NotFoundError(ProvenanceAPIError):
    """HTTP 404 — Resource not found."""


class ConflictError(ProvenanceAPIError):
    """HTTP 409 — Conflict (e.g., cannot cancel a completed job)."""


class RateLimitError(ProvenanceAPIError):
    """HTTP 429 — Rate limit or daily usage limit exceeded."""

    def __init__(self, message: str, *, retry_after: int | None = None, **kwargs) -> None:
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class ValidationError(ProvenanceAPIError):
    """HTTP 422 — Invalid request body or parameters."""


class TimeoutError(ProvenanceAPIError):
    """Transport timeout (no HTTP status code)."""

    def __init__(self, message: str = "Request timed out", **kwargs) -> None:
        super().__init__(message, **kwargs)


# Map HTTP status codes to exception classes
_STATUS_MAP: dict[int, type[ProvenanceAPIError]] = {
    401: AuthenticationError,
    403: ForbiddenError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
}


def raise_for_status(
    status_code: int,
    detail: str,
    *,
    request_id: str | None = None,
    retry_after: int | None = None,
) -> None:
    """Raise the appropriate exception for an HTTP status code.

    Does nothing for 2xx status codes.  Raises ``ProvenanceAPIError``
    for unmapped non-2xx codes.
    """
    if 200 <= status_code < 300:
        return

    exc_class = _STATUS_MAP.get(status_code, ProvenanceAPIError)
    kwargs: dict = {"status_code": status_code, "detail": detail, "request_id": request_id}
    if exc_class is RateLimitError and retry_after is not None:
        kwargs["retry_after"] = retry_after
    raise exc_class(
        f"HTTP {status_code}: {detail}",
        **kwargs,
    )
