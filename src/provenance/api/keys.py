"""API key management for the provenance HTTP API.

Provides:
- Key generation with random secrets
- SHA-256 hashing (raw secrets are never stored)
- Legacy PROVENANCE_API_KEY backwards compatibility
- Key validation against the repository
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from provenance.api.db import AnalysisRepository


def generate_api_key() -> str:
    """Generate a random API key (32 bytes, hex-encoded)."""
    return secrets.token_hex(32)


def hash_api_key(key: str) -> str:
    """SHA-256 hash of an API key."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_key_id() -> str:
    """Generate a UUID-based key ID."""
    return str(uuid.uuid4())


def get_legacy_key_hash() -> str | None:
    """Hash the legacy PROVENANCE_API_KEY env var, if set."""
    raw = os.environ.get("PROVENANCE_API_KEY")
    if not raw:
        return None
    return hash_api_key(raw)


def validate_api_key(
    key: str,
    repo: AnalysisRepository,
) -> tuple[bool, str | None]:
    """Validate an API key against stored keys and the legacy env var.

    Returns (is_valid, key_id). The key_id is None for legacy keys.
    """
    key_hash = hash_api_key(key)

    # Check stored keys first
    record = repo.get_api_key_by_hash(key_hash)
    if record is not None and record["status"] == "active":
        return True, record["key_id"]

    # Fallback to legacy env var
    legacy_hash = get_legacy_key_hash()
    if legacy_hash and key_hash == legacy_hash:
        return True, "_legacy"

    return False, None
