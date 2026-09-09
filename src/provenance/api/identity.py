"""Anonymous visitor identity for the public SaaS product.

Primary mechanism: a server-generated random opaque visitor ID carried in
a signed HttpOnly cookie (``pv_visitor``).  Signing uses HMAC-SHA256 with
``PROVENANCE_ANON_COOKIE_SECRET`` from the standard library only.

Secondary abuse signals (coarse, HMAC-hashed — never raw values at rest):
- coarse IP prefix (/24 for IPv4, /48 for IPv6)
- User-Agent string

No browser fingerprinting.  No raw IP persisted.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets

logger = logging.getLogger("provenance.api")

COOKIE_NAME = "pv_visitor"
_COOKIE_VERSION = "v1"

# Ephemeral fallback when no secret is configured (single-process dev/test).
# Rotating per process intentionally invalidates old cookies.
_EPHEMERAL_SECRET: bytes | None = None


def _cookie_secret() -> bytes:
    configured = os.environ.get("PROVENANCE_ANON_COOKIE_SECRET", "").strip()
    if configured:
        return configured.encode("utf-8")
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_bytes(32)
        logger.warning(
            "PROVENANCE_ANON_COOKIE_SECRET is not set; using an ephemeral "
            "per-process secret. Visitor cookies will not survive restarts. "
            "Set a stable secret in production."
        )
    return _EPHEMERAL_SECRET


def cookie_secure() -> bool:
    """Whether the visitor cookie requires HTTPS.

    Defaults to False for local development; production deployments must
    set ``PROVENANCE_ANON_COOKIE_SECURE=1``.
    """
    return os.environ.get("PROVENANCE_ANON_COOKIE_SECURE", "0").strip() in ("1", "true", "yes")


def cookie_domain() -> str | None:
    domain = os.environ.get("PROVENANCE_ANON_COOKIE_DOMAIN", "").strip()
    return domain or None


def cookie_samesite() -> str:
    raw = os.environ.get("PROVENANCE_COOKIE_SAMESITE", "lax").strip().lower()
    return raw if raw in ("lax", "strict", "none") else "lax"


def new_visitor_id() -> str:
    """Generate a fresh opaque visitor ID."""
    return f"{_COOKIE_VERSION}.{secrets.token_urlsafe(24)}"


def sign_visitor_id(visitor_id: str) -> str:
    """Return the cookie value ``<visitor_id>.<hex_signature>``."""
    sig = hmac.new(_cookie_secret(), visitor_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{visitor_id}.{sig}"


def verify_visitor_cookie(value: str | None) -> str | None:
    """Return the visitor ID if *value* is a valid signed cookie, else None."""
    if not value or value.count(".") < 2:
        return None
    visitor_id, sig = value.rsplit(".", 1)
    if not visitor_id or not sig:
        return None
    expected = hmac.new(_cookie_secret(), visitor_id.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    return visitor_id


def coarse_ip_prefix(ip: str) -> str:
    """Reduce an IP to a coarse prefix (/24 IPv4, /48-ish IPv6)."""
    ip = ip.strip()
    if ":" in ip:  # IPv6: keep first 3 hextets
        parts = ip.split(":")
        return ":".join(parts[:3])
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return ip


def abuse_signal_hash(*parts: str) -> str:
    """HMAC-hash coarse abuse signals so raw values are never persisted."""
    payload = "\x00".join(parts)
    return hmac.new(_cookie_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
