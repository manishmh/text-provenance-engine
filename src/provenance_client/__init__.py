"""Python SDK for the Text Provenance Engine API.

Provides a typed, convenient interface for all API operations.

Example::

    from provenance_client import ProvenanceClient

    client = ProvenanceClient(
        base_url="http://localhost:8000",
        api_key="your-api-key",
    )

    result = client.analyze(text="Hello, world!", detectors=["unicode"])
    print(result.status)
"""

from provenance_client._client import ProvenanceClient
from provenance_client._exceptions import (
    ProvenanceAPIError,
    AuthenticationError,
    ForbiddenError,
    NotFoundError,
    ConflictError,
    RateLimitError,
    ValidationError,
    TimeoutError,
)

__version__ = "0.1.0"

__all__ = [
    "ProvenanceClient",
    "ProvenanceAPIError",
    "AuthenticationError",
    "ForbiddenError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ValidationError",
    "TimeoutError",
]
