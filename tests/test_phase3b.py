"""Phase 3B tests: API key provisioning, admin auth, usage limits, headers.

Uses a temporary SQLite database for each test.
"""

from __future__ import annotations

import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import SqliteRepository
from provenance.api.keys import hash_api_key

API_KEY = "test-phase3b-normal-key"
ADMIN_KEY = "test-phase3b-admin-key"


@pytest.fixture(autouse=True)
def _set_keys(monkeypatch):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    monkeypatch.setenv("PROVENANCE_ADMIN_API_KEY", ADMIN_KEY)
    import provenance.api.middleware as mw
    mw._limiter = None


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "test.db")
    yield r
    r.close()


def _auth(key: str = API_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


def _admin_auth() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}


# ---------------------------------------------------------------------------
# API Key Provisioning — POST /v1/api-keys
# ---------------------------------------------------------------------------


def test_create_api_key_requires_admin(client):
    """Normal key cannot create API keys."""
    resp = client.post("/v1/api-keys", json={"name": "Test"}, headers=_auth())
    assert resp.status_code == 403


def test_create_api_key_requires_admin_key(client):
    """Missing key gets 401."""
    resp = client.post("/v1/api-keys", json={"name": "Test"})
    assert resp.status_code == 401


def test_create_api_key_wrong_admin_key(client):
    """Wrong admin key gets 403."""
    resp = client.post("/v1/api-keys", json={"name": "Test"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 403


def test_create_api_key_works(client):
    """Admin can create a key and gets the raw secret back."""
    resp = client.post("/v1/api-keys", json={"name": "My Key"}, headers=_admin_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "My Key"
    assert data["status"] == "active"
    assert data["key_id"]
    assert data["created_at"]
    # The raw key must be present in creation response
    assert data["key"]
    assert len(data["key"]) == 64  # 32 bytes hex


def test_create_api_key_raw_not_persisted(client, tmp_path):
    """The raw API key must never appear in the database."""
    resp = client.post("/v1/api-keys", json={"name": "Persist Test"}, headers=_admin_auth())
    raw_key = resp.json()["key"]

    # Check DB directly
    db_path = tmp_path / "test.db"
    conn = __import__("sqlite3").connect(str(db_path))
    rows = conn.execute("SELECT * FROM api_keys").fetchall()
    conn.close()
    for row in rows:
        serialized = json.dumps(row)
        assert raw_key not in serialized
        # Also check prefix
        assert raw_key[:8] not in serialized


def test_create_api_key_empty_name_rejected(client):
    resp = client.post("/v1/api-keys", json={"name": ""}, headers=_admin_auth())
    assert resp.status_code == 422


def test_create_api_key_no_name_rejected(client):
    resp = client.post("/v1/api-keys", json={}, headers=_admin_auth())
    assert resp.status_code == 422


def test_created_key_authenticates(client):
    """A key created via POST /v1/api-keys should work for /v1/analyze."""
    resp = client.post("/v1/api-keys", json={"name": "Auth Test"}, headers=_admin_auth())
    raw_key = resp.json()["key"]

    # Use the new key
    resp2 = client.post("/v1/analyze", json={"text": "key auth test"}, headers=_auth(raw_key))
    assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# API Key Listing — GET /v1/api-keys
# ---------------------------------------------------------------------------


def test_list_api_keys_requires_admin(client):
    resp = client.get("/v1/api-keys", headers=_auth())
    assert resp.status_code == 403


def test_list_api_keys_empty(client):
    resp = client.get("/v1/api-keys", headers=_admin_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["keys"] == []
    assert data["total"] == 0


def test_list_api_keys_returns_created(client):
    client.post("/v1/api-keys", json={"name": "Key A"}, headers=_admin_auth())
    client.post("/v1/api-keys", json={"name": "Key B"}, headers=_admin_auth())

    resp = client.get("/v1/api-keys", headers=_admin_auth())
    data = resp.json()
    assert data["total"] == 2
    names = {k["name"] for k in data["keys"]}
    assert names == {"Key A", "Key B"}


def test_list_api_keys_no_secrets(client):
    """Listing must never return key_hash or raw key."""
    client.post("/v1/api-keys", json={"name": "No Secrets"}, headers=_admin_auth())
    resp = client.get("/v1/api-keys", headers=_admin_auth())
    body = json.dumps(resp.json())
    assert "key_hash" not in body
    # Verify no 64-char hex string (raw key) appears in key summaries
    for k in resp.json()["keys"]:
        assert "key" not in k  # no raw key field


# ---------------------------------------------------------------------------
# API Key Revocation — DELETE /v1/api-keys/{key_id}
# ---------------------------------------------------------------------------


def test_revoke_api_key_requires_admin(client):
    # First create a key
    resp = client.post("/v1/api-keys", json={"name": "Revoke Test"}, headers=_admin_auth())
    key_id = resp.json()["key_id"]

    resp2 = client.delete(f"/v1/api-keys/{key_id}", headers=_auth())
    assert resp2.status_code == 403


def test_revoke_api_key_works(client):
    resp = client.post("/v1/api-keys", json={"name": "Revoke Me"}, headers=_admin_auth())
    key_id = resp.json()["key_id"]
    raw_key = resp.json()["key"]

    # Revoke
    resp2 = client.delete(f"/v1/api-keys/{key_id}", headers=_admin_auth())
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "revoked"

    # Revoked key should not authenticate
    resp3 = client.post("/v1/analyze", json={"text": "revoked"}, headers=_auth(raw_key))
    assert resp3.status_code == 403


def test_revoke_nonexistent_key_returns_404(client):
    resp = client.delete("/v1/api-keys/nonexistent-id", headers=_admin_auth())
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Authentication — stored vs legacy vs admin
# ---------------------------------------------------------------------------


def test_legacy_key_still_works(client):
    """Legacy PROVENANCE_API_KEY should still authenticate."""
    resp = client.post("/v1/analyze", json={"text": "legacy"}, headers=_auth(API_KEY))
    assert resp.status_code == 200


def test_admin_key_works_for_normal_endpoints(client):
    """Admin key can also access normal /v1/* endpoints."""
    resp = client.post("/v1/analyze", json={"text": "admin as normal"}, headers=_auth(ADMIN_KEY))
    assert resp.status_code == 200


def test_invalid_key_rejected(client):
    resp = client.post("/v1/analyze", json={"text": "x"}, headers=_auth("bad-key"))
    assert resp.status_code == 403


def test_no_key_when_auth_configured(client):
    resp = client.post("/v1/analyze", json={"text": "x"})
    assert resp.status_code == 401


def test_secrets_never_in_logs(client, caplog):
    """Raw API keys must never appear in log output."""
    import logging
    with caplog.at_level(logging.DEBUG, logger="provenance.api"):
        client.post("/v1/analyze", json={"text": "log test"}, headers=_auth())
    all_log_text = " ".join(r.getMessage() for r in caplog.records)
    assert API_KEY not in all_log_text
    assert ADMIN_KEY not in all_log_text


def test_secrets_never_in_error_responses(client):
    resp = client.post("/v1/analyze", json={"text": "x"}, headers=_auth("my-secret-123"))
    body = json.dumps(resp.json())
    assert "my-secret-123" not in body


# ---------------------------------------------------------------------------
# Database-backed Daily Limits
# ---------------------------------------------------------------------------


def test_no_limits_by_default(client, monkeypatch):
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)
    resp = client.post("/v1/analyze", json={"text": "no limits"}, headers=_auth())
    assert resp.status_code == 200


def test_request_limit_enforced(client, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "2")
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)

    resp1 = client.post("/v1/analyze", json={"text": "r1"}, headers=_auth())
    assert resp1.status_code == 200
    resp2 = client.post("/v1/analyze", json={"text": "r2"}, headers=_auth())
    assert resp2.status_code == 200
    resp3 = client.post("/v1/analyze", json={"text": "r3"}, headers=_auth())
    assert resp3.status_code == 429
    assert "Daily request limit" in resp3.json()["detail"]


def test_character_limit_enforced(client, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DAILY_CHARACTER_LIMIT", "10")
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)

    resp1 = client.post("/v1/analyze", json={"text": "short"}, headers=_auth())
    assert resp1.status_code == 200
    resp2 = client.post("/v1/analyze", json={"text": "a" * 20}, headers=_auth())
    assert resp2.status_code == 429
    assert "Daily character limit" in resp2.json()["detail"]


def test_limits_survive_restart(client, tmp_path, monkeypatch):
    """Limits are derived from DB, so they persist across app restarts."""
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "1")

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    app1 = create_app(db_url=db_url)
    with TestClient(app1) as c1:
        resp = c1.post("/v1/analyze", json={"text": "first"}, headers=_auth())
        assert resp.status_code == 200

    # Second app instance with same DB
    app2 = create_app(db_url=db_url)
    with TestClient(app2) as c2:
        resp = c2.post("/v1/analyze", json={"text": "second"}, headers=_auth())
        assert resp.status_code == 429  # limit already reached


def test_different_keys_separate_limits(client, monkeypatch):
    """Daily limits are per-key, not global."""
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "1")
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)

    # Create a second key
    resp = client.post("/v1/api-keys", json={"name": "Key 2"}, headers=_admin_auth())
    key2 = resp.json()["key"]

    # Key 1 uses its limit
    resp1 = client.post("/v1/analyze", json={"text": "k1 first"}, headers=_auth())
    assert resp1.status_code == 200
    resp1b = client.post("/v1/analyze", json={"text": "k1 second"}, headers=_auth())
    assert resp1b.status_code == 429

    # Key 2 still has its limit
    resp2 = client.post("/v1/analyze", json={"text": "k2 first"}, headers=_auth(key2))
    assert resp2.status_code == 200


# ---------------------------------------------------------------------------
# Usage Headers
# ---------------------------------------------------------------------------


def test_usage_headers_present_with_limits(client, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "100")
    monkeypatch.setenv("PROVENANCE_DAILY_CHARACTER_LIMIT", "50000")

    resp = client.post("/v1/analyze", json={"text": "header test"}, headers=_auth())
    assert resp.status_code == 200
    assert "x-usage-request-limit" in resp.headers
    assert resp.headers["x-usage-request-limit"] == "100"
    assert "x-usage-requests-remaining" in resp.headers
    assert int(resp.headers["x-usage-requests-remaining"]) >= 0
    assert "x-usage-character-limit" in resp.headers
    assert resp.headers["x-usage-character-limit"] == "50000"


def test_no_usage_headers_without_limits(client, monkeypatch):
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)

    resp = client.post("/v1/analyze", json={"text": "no headers"}, headers=_auth())
    assert "x-usage-request-limit" not in resp.headers


def test_usage_headers_on_analyze_and_get(client, monkeypatch):
    """Usage headers appear on both POST /v1/analyze and GET /v1/analyses/{id}."""
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "50")

    resp = client.post("/v1/analyze", json={"text": "h test"}, headers=_auth())
    aid = resp.json()["analysis_id"]
    assert "x-usage-request-limit" in resp.headers

    resp2 = client.get(f"/v1/analyses/{aid}", headers=_auth())
    assert "x-usage-request-limit" in resp2.headers


# ---------------------------------------------------------------------------
# Persistence — SQLite CRUD
# ---------------------------------------------------------------------------


def test_api_key_lifecycle_sqlite(repo):
    """Full CRUD cycle on the repository layer."""
    key_hash = hash_api_key("lifecycle-key")
    repo.create_api_key(key_id="lc1", key_hash=key_hash, name="Lifecycle")

    record = repo.get_api_key_by_hash(key_hash)
    assert record is not None
    assert record["key_id"] == "lc1"
    assert record["status"] == "active"

    # get by id
    record2 = repo.get_api_key_by_id("lc1")
    assert record2 is not None
    assert record2["name"] == "Lifecycle"

    # update last_used
    repo.update_api_key_last_used("lc1")
    record3 = repo.get_api_key_by_id("lc1")
    assert record3["last_used_at"] is not None

    # revoke
    repo.revoke_api_key("lc1")
    record4 = repo.get_api_key_by_hash(key_hash)
    assert record4["status"] == "revoked"

    # list
    keys = repo.list_api_keys()
    assert len(keys) == 1
    assert keys[0]["key_id"] == "lc1"


def test_get_nonexistent_api_key_by_id(repo):
    assert repo.get_api_key_by_id("nope") is None


# ---------------------------------------------------------------------------
# Mocked PostgreSQL API key methods
# ---------------------------------------------------------------------------


def test_pg_api_key_methods_mocked():
    from tests.test_api import _make_mock_pg
    from provenance.api.db import PostgresRepository

    mock_pg, mock_conn, mock_cursor = _make_mock_pg()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")
        key_hash = hash_api_key("pg-key")
        repo.create_api_key(key_id="pg-k1", key_hash=key_hash, name="PG Key")
        assert mock_cursor.execute.called

        repo.get_api_key_by_id("pg-k1")
        repo.update_api_key_last_used("pg-k1")
        repo.revoke_api_key("pg-k1")
        repo.close()


# ---------------------------------------------------------------------------
# Admin endpoint not configured
# ---------------------------------------------------------------------------


def test_admin_endpoints_disabled_without_env(tmp_path, monkeypatch):
    """When PROVENANCE_ADMIN_API_KEY is not set, admin endpoints return 403."""
    monkeypatch.delenv("PROVENANCE_ADMIN_API_KEY", raising=False)
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        resp = c.post("/v1/api-keys", json={"name": "X"}, headers=_auth())
        assert resp.status_code == 403
        assert "not configured" in resp.json()["detail"]
