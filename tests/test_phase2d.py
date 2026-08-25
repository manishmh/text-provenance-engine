"""Phase 2D tests: rate limiting, request IDs, CORS, pagination, error handling.

Uses a temporary SQLite database for each test.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app

API_KEY = "test-phase2d-key"


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    # Reset the global rate limiter between tests
    import provenance.api.middleware as mw
    mw._limiter = None
    # Reset graceful shutdown state between tests
    import provenance.api.jobs as jobs_mod
    jobs_mod._shutting_down = False
    jobs_mod._executor = None


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


def _auth():
    return {"X-API-Key": API_KEY}


# ---------------------------------------------------------------------------
# Request IDs
# ---------------------------------------------------------------------------


def test_request_id_generated_when_absent(client):
    """X-Request-ID should be generated when not provided."""
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    rid = resp.headers["x-request-id"]
    assert len(rid) == 36  # UUID format
    assert resp.json()["status"] == "ok"


def test_request_id_propagated_when_provided(client):
    """X-Request-ID from the client should be echoed back."""
    my_rid = "custom-rid-12345"
    resp = client.get("/health", headers={"X-Request-ID": my_rid})
    assert resp.headers["X-Request-ID"] == my_rid


def test_request_id_on_v1_endpoints(client):
    """X-Request-ID should appear on /v1/* endpoints too."""
    resp = client.get("/v1/analyses", headers=_auth())
    assert "x-request-id" in resp.headers
    rid = resp.headers["x-request-id"]
    assert len(rid) == 36


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limit_allows_normal_requests(client):
    """Normal requests should pass through rate limiting."""
    resp = client.get("/v1/analyses", headers=_auth())
    assert resp.status_code == 200


def test_rate_limit_triggers_429(client, monkeypatch):
    """Exceeding the rate limit should return 429."""
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    # Reset limiter to pick up new config
    import provenance.api.middleware as mw
    mw._limiter = None

    for _ in range(3):
        resp = client.get("/v1/analyses", headers=_auth())
        assert resp.status_code == 200

    resp = client.get("/v1/analyses", headers=_auth())
    assert resp.status_code == 429
    data = resp.json()
    assert "Rate limit" in data["detail"]
    assert "retry-after" in resp.headers


def test_rate_limit_does_not_apply_to_health(client, monkeypatch):
    """/health should not be rate-limited."""
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    import provenance.api.middleware as mw
    mw._limiter = None

    for _ in range(10):
        resp = client.get("/health")
        assert resp.status_code == 200


def test_rate_limit_does_not_apply_to_ready(client, monkeypatch):
    """/ready should not be rate-limited."""
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    import provenance.api.middleware as mw
    mw._limiter = None

    for _ in range(10):
        resp = client.get("/ready")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_no_origin_by_default(client):
    """Default: no CORS headers when no CORS_ORIGINS is set."""
    resp = client.get("/health")
    # Starlette CORS middleware only adds headers for cross-origin requests
    # With no origins configured, there should be no access-control-allow-origin
    assert "access-control-allow-origin" not in resp.headers


def test_cors_with_configured_origins(tmp_path, monkeypatch):
    """When CORS_ORIGINS is set, matching origins should be allowed."""
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com,https://app.example.com")
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        resp = c.get(
            "/health",
            headers={"Origin": "https://example.com"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://example.com"


def test_cors_rejects_unconfigured_origin(tmp_path, monkeypatch):
    """Unconfigured origins should be rejected."""
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com")
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        resp = c.get(
            "/health",
            headers={"Origin": "https://evil.com"},
        )
        assert "access-control-allow-origin" not in resp.headers


# ---------------------------------------------------------------------------
# Pagination hardening
# ---------------------------------------------------------------------------


def test_pagination_max_limit_enforced(client):
    """Limit must not exceed MAX_LIST_LIMIT."""
    resp = client.get("/v1/analyses?limit=200", headers=_auth())
    assert resp.status_code == 422


def test_pagination_valid_limits_work(client):
    """Limits within bounds should work."""
    resp = client.get("/v1/analyses?limit=50&offset=0", headers=_auth())
    assert resp.status_code == 200


def test_pagination_configurable_max(tmp_path, monkeypatch):
    """MAX_LIST_LIMIT env var controls the maximum."""
    monkeypatch.setenv("MAX_LIST_LIMIT", "10")
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        # 11 should be rejected
        resp = c.get("/v1/analyses?limit=11", headers=_auth())
        assert resp.status_code == 422
        # 10 should be accepted
        resp = c.get("/v1/analyses?limit=10", headers=_auth())
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unexpected_error_returns_json(client):
    """An unexpected exception should return a clean JSON 500."""
    from provenance.api import routes

    original = routes._get_repo

    def _explode():
        raise RuntimeError("boom")

    routes._get_repo = _explode  # type: ignore[assignment]
    try:
        resp = client.get("/v1/analyses", headers=_auth())
        assert resp.status_code == 500
        data = resp.json()
        assert data["detail"] == "Internal server error"
        assert "request_id" in data
        # Must NOT expose the internal error message
        assert "boom" not in json.dumps(data)
    finally:
        routes._get_repo = original


def test_error_handler_does_not_leak_secrets(client):
    """Error responses must not contain secrets."""
    from provenance.api import routes

    original = routes._get_repo

    def _explode():
        raise RuntimeError("connection to postgresql://user:pass@host/db failed")

    routes._get_repo = _explode  # type: ignore[assignment]
    try:
        resp = client.get("/v1/analyses", headers=_auth())
        assert resp.status_code == 500
        body = json.dumps(resp.json())
        assert "pass" not in body
        assert "postgresql" not in body
    finally:
        routes._get_repo = original


def test_http_exceptions_still_work(client):
    """Existing 401/403/404 behavior must be preserved."""
    # 401 — missing API key
    resp = client.get("/v1/analyses")
    assert resp.status_code == 401

    # 403 — wrong API key
    resp = client.get("/v1/analyses", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 403

    # 404 — nonexistent analysis
    resp = client.get("/v1/analyses/nonexistent", headers=_auth())
    assert resp.status_code == 404

    # 422 — invalid body
    resp = client.post("/v1/analyze", json={}, headers=_auth())
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Health / Readiness
# ---------------------------------------------------------------------------


def test_health_unauthenticated(client):
    """/health must work without any auth."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "engine_version" in data


def test_ready_unauthenticated(client):
    """/ready must work without any auth."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert "engine_version" in data


def test_ready_reflects_backend_health(client):
    """/ready should return ready when the DB is accessible."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def test_request_logged(caplog):
    """Requests should be logged with method, path, status, duration."""
    db_path = "/tmp/test_phase2d_logging.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c, caplog.at_level(logging.INFO, logger="provenance.api"):
        c.get("/health")
        records = [r for r in caplog.records if "GET" in r.getMessage() and "/health" in r.getMessage()]
        assert len(records) >= 1
        msg = records[-1].getMessage()
        assert "200" in msg
        assert "rid=" in msg


def test_no_raw_text_in_logs(caplog, tmp_path):
    """Raw input text must never appear in log output."""
    secret = "SENSITIVE-SECRET-TEXT-12345"
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c, caplog.at_level(logging.DEBUG, logger="provenance.api"):
        c.post(
            "/v1/analyze",
            json={"text": secret},
            headers=_auth(),
        )
        all_log_text = " ".join(r.getMessage() for r in caplog.records)
        assert secret not in all_log_text


# ---------------------------------------------------------------------------
# No secret leakage in responses
# ---------------------------------------------------------------------------


def test_no_api_key_in_error_responses(client):
    """API key must not appear in any error response body."""
    resp = client.get("/v1/analyses", headers={"X-API-Key": "my-super-secret-key"})
    body = json.dumps(resp.json())
    assert "my-super-secret-key" not in body


def test_no_database_url_in_error_responses(client, monkeypatch):
    """DATABASE_URL must not leak in error responses."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://admin:s3cret@db:5432/prod")
    db_path = "/tmp/test_phase2d_leak.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        resp = c.get("/v1/analyses", headers={"X-API-Key": "wrong-key"})
        body = json.dumps(resp.json())
        assert "s3cret" not in body
        assert "postgresql" not in body
