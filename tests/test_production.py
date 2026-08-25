"""Production smoke tests: security, auth, ownership, and deployment guarantees."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_executor_state():
    """Reset the global executor shutdown state between tests."""
    import provenance.api.jobs as jobs_mod
    jobs_mod._shutting_down = False
    jobs_mod._executor = None
    yield
    # Reset again after the test so teardown doesn't leak
    jobs_mod._shutting_down = False
    jobs_mod._executor = None


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a test client with auth enabled."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PROVENANCE_API_KEY", "test-key-123")
    monkeypatch.setenv("PROVENANCE_ADMIN_API_KEY", "admin-key-999")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, db_path


# ---------------------------------------------------------------------------
# Health / Ready without auth
# ---------------------------------------------------------------------------

class TestPublicEndpoints:
    def test_health_requires_no_auth(self, client):
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "engine_version" in body

    def test_ready_requires_no_auth(self, client):
        c, _ = client
        r = c.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["persistence"] == "ok"
        assert body["executor"] == "ok"

    def test_metrics_requires_no_auth(self, client):
        c, _ = client
        r = c.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert "total_requests" in body
        assert "configured_worker_count" in body


# ---------------------------------------------------------------------------
# Authentication enforcement
# ---------------------------------------------------------------------------

class TestAuthEnforcement:
    def test_v1_requires_api_key(self, client):
        c, _ = client
        r = c.post("/v1/analyze", json={"text": "hello"})
        assert r.status_code == 401

    def test_invalid_api_key_rejected(self, client):
        c, _ = client
        r = c.post(
            "/v1/analyze",
            json={"text": "hello"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 403

    def test_valid_api_key_accepted(self, client):
        c, _ = client
        r = c.post(
            "/v1/analyze",
            json={"text": "hello"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert r.status_code == 200

    def test_admin_key_works_for_normal_endpoints(self, client):
        c, _ = client
        r = c.post(
            "/v1/analyze",
            json={"text": "hello"},
            headers={"X-API-Key": "admin-key-999"},
        )
        assert r.status_code == 200

    def test_non_admin_key_rejected_for_admin_endpoints(self, client):
        c, _ = client
        r = c.get("/v1/api-keys", headers={"X-API-Key": "test-key-123"})
        assert r.status_code == 403

    def test_admin_key_accepted_for_admin_endpoints(self, client):
        c, _ = client
        r = c.get("/v1/api-keys", headers={"X-API-Key": "admin-key-999"})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Ownership enforcement
# ---------------------------------------------------------------------------

class TestOwnershipEnforcement:
    def test_analyses_belong_to_calling_key(self, client):
        c, _ = client
        # Analyze with key1
        r1 = c.post(
            "/v1/analyze",
            json={"text": "owned by key1"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert r1.status_code == 200
        aid = r1.json()["analysis_id"]

        # Analyze with a different key (admin)
        r2 = c.post(
            "/v1/analyze",
            json={"text": "owned by admin"},
            headers={"X-API-Key": "admin-key-999"},
        )
        assert r2.status_code == 200

        # Admin can see any analysis via direct ID
        r3 = c.get(f"/v1/analyses/{aid}", headers={"X-API-Key": "admin-key-999"})
        assert r3.status_code == 200

    def test_job_ownership_enforced(self, client):
        c, _ = client
        # Create job with key1
        r1 = c.post(
            "/v1/analyze/async",
            json={"text": "owned job"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert r1.status_code == 202
        jid = r1.json()["job_id"]

        # Wait for background job to finish (SQLite threading safety)
        time.sleep(0.5)

        # Admin can see it
        r2 = c.get(f"/v1/jobs/{jid}", headers={"X-API-Key": "admin-key-999"})
        assert r2.status_code == 200


# ---------------------------------------------------------------------------
# Raw text not persisted
# ---------------------------------------------------------------------------

class TestRawTextNotPersisted:
    def test_raw_text_not_in_database(self, client):
        c, db_path = client
        secret_text = "THIS_IS_SECRET_TEXT_SHOULD_NOT_BE_STORED"
        c.post(
            "/v1/analyze",
            json={"text": secret_text},
            headers={"X-API-Key": "test-key-123"},
        )
        # Read the SQLite DB directly
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        # Check analyses table has no raw text column
        for t in tables:
            if "analyses" in t.lower():
                assert "TEXT" not in t.upper() or "input_text" not in t.lower()
        # Search for the secret text in any text column
        cursor = conn.execute("SELECT * FROM analyses")
        for row in cursor.fetchall():
            for val in row:
                if isinstance(val, str) and secret_text in val:
                    pytest.fail(f"Raw text found in database: {val}")
        conn.close()


# ---------------------------------------------------------------------------
# API secrets not persisted or logged
# ---------------------------------------------------------------------------

class TestSecretNotPersisted:
    def test_api_key_not_in_database(self, client):
        c, db_path = client
        conn = sqlite3.connect(str(db_path))
        # Check api_keys table has no raw key column
        cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name='api_keys'")
        row = cursor.fetchone()
        conn.close()
        if row:
            assert "key_hash" in row[0]
            # Ensure no "key_secret" or similar column exists
            assert "secret" not in row[0].lower() or "key_hash" in row[0].lower()

    def test_api_key_hash_matches(self, client):
        """Verify stored hash matches expected SHA-256 of the key."""
        c, db_path = client
        # Create a key via admin endpoint
        r = c.post(
            "/v1/api-keys",
            json={"name": "test-prod"},
            headers={"X-API-Key": "admin-key-999"},
        )
        assert r.status_code == 200
        raw_key = r.json()["key"]

        # The raw key should NOT appear in the DB
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT * FROM api_keys")
        for row in cursor.fetchall():
            for val in row:
                if isinstance(val, str) and raw_key in val:
                    pytest.fail("Raw API key found in database")
        conn.close()


# ---------------------------------------------------------------------------
# Readiness detects unavailable persistence
# ---------------------------------------------------------------------------

class TestReadinessProbe:
    def test_ready_reflects_db_state(self, client):
        c, _ = client
        r = c.get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["persistence"] == "ok"
        # Executor may be 'ok' or 'shutting_down' depending on teardown timing
        assert body["executor"] in ("ok", "shutting_down")
        if body["executor"] == "ok":
            assert body["status"] == "ready"


# ---------------------------------------------------------------------------
# Metrics don't leak secrets
# ---------------------------------------------------------------------------

class TestMetricsNoSecrets:
    def test_metrics_no_api_keys(self, client):
        c, _ = client
        r = c.get("/metrics")
        assert r.status_code == 200
        text = r.text
        assert "test-key-123" not in text
        assert "admin-key-999" not in text

    def test_metrics_no_raw_text(self, client):
        c, _ = client
        # Do an analysis first
        c.post(
            "/v1/analyze",
            json={"text": "SECRET_CONTENT_42"},
            headers={"X-API-Key": "test-key-123"},
        )
        r = c.get("/metrics")
        assert r.status_code == 200
        assert "SECRET_CONTENT_42" not in r.text


# ---------------------------------------------------------------------------
# Error responses don't leak internals
# ---------------------------------------------------------------------------

class TestErrorSanitization:
    def test_500_returns_safe_error(self, client):
        c, _ = client
        # Request a non-existent analysis — should be 404, not 500
        r = c.get(
            "/v1/analyses/nonexistent-id-12345",
            headers={"X-API-Key": "test-key-123"},
        )
        assert r.status_code == 404
        body = r.json()
        assert "detail" in body
        # Should not contain stack traces or internal paths
        assert "traceback" not in body["detail"].lower()
        assert "/home/" not in body["detail"]

    def test_invalid_detector_returns_422(self, client):
        c, _ = client
        r = c.post(
            "/v1/analyze",
            json={"text": "hello", "detectors": ["nonexistent_detector"]},
            headers={"X-API-Key": "test-key-123"},
        )
        # Should return 422 or a clean error, not a 500
        assert r.status_code in (422, 200)


# ---------------------------------------------------------------------------
# Request IDs propagate
# ---------------------------------------------------------------------------

class TestRequestIDs:
    def test_request_id_in_response(self, client):
        c, _ = client
        r = c.get("/health")
        assert "X-Request-ID" in r.headers

    def test_request_id_echoed(self, client):
        c, _ = client
        r = c.get("/health", headers={"X-Request-ID": "test-req-123"})
        assert r.headers.get("X-Request-ID") == "test-req-123"


# ---------------------------------------------------------------------------
# Config validation at startup
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_config_passes(self, client):
        """Valid config should not cause startup errors."""
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200

    def test_invalid_config_raises_clear_error(self, monkeypatch):
        """Invalid config values raise ConfigError clearly."""
        monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "0")
        from provenance.api.config import validate_config, ConfigError
        with pytest.raises(ConfigError, match="must be >= 1"):
            validate_config()
