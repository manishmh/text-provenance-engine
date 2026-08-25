"""Phase 3A tests: API key management, usage tracking, limits, response metadata.

Uses a temporary SQLite database for each test.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import SqliteRepository, _text_hash
from provenance.api.keys import generate_api_key, hash_api_key

API_KEY = "test-phase3a-secret-key"


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    # Reset global rate limiter between tests
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
    db_path = tmp_path / "test.db"
    r = SqliteRepository(db_path)
    yield r
    r.close()


def _auth(key: str = API_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------


def test_legacy_api_key_works(client):
    """Legacy PROVENANCE_API_KEY env var should still work for auth."""
    resp = client.post("/v1/analyze", json={"text": "legacy test"}, headers=_auth())
    assert resp.status_code == 200


def test_invalid_api_key_rejected(client):
    """A wrong API key should get 403."""
    resp = client.post("/v1/analyze", json={"text": "x"}, headers=_auth("wrong-key"))
    assert resp.status_code == 403


def test_missing_api_key_rejected(client):
    """Missing API key should get 401."""
    resp = client.post("/v1/analyze", json={"text": "x"})
    assert resp.status_code == 401


def test_stored_api_key_works(client, repo):
    """A key stored in the database should be accepted."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    key_id = "test-key-001"
    repo.create_api_key(key_id=key_id, key_hash=key_hash, name="Test Key")

    resp = client.post("/v1/analyze", json={"text": "stored key test"}, headers=_auth(key))
    assert resp.status_code == 200


def test_revoked_api_key_rejected(client, repo):
    """A revoked key should get 403."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    key_id = "revoked-key-001"
    repo.create_api_key(key_id=key_id, key_hash=key_hash, name="Revoked Key")
    repo.revoke_api_key(key_id)

    resp = client.post("/v1/analyze", json={"text": "x"}, headers=_auth(key))
    assert resp.status_code == 403


def test_api_key_secret_never_stored(client, repo):
    """The raw API key secret must never appear in the database."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    repo.create_api_key(key_id="secret-test", key_hash=key_hash, name="Secret Test")

    # Query DB directly
    import sqlite3
    db_path = repo._db_path
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT * FROM api_keys").fetchall()
    conn.close()
    for row in rows:
        serialized = json.dumps(row)
        assert key not in serialized
        assert key[:8] not in serialized  # no prefix leakage


def test_api_key_secret_never_in_response(client, repo):
    """Raw API key must never appear in any API response."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    repo.create_api_key(key_id="resp-test", key_hash=key_hash, name="Resp Test")

    resp = client.post("/v1/analyze", json={"text": "response test"}, headers=_auth(key))
    body = json.dumps(resp.json())
    assert key not in body
    assert key[:8] not in body


def test_list_api_keys(repo):
    """list_api_keys should return all keys."""
    key_hash = hash_api_key("test-key-1")
    repo.create_api_key(key_id="k1", key_hash=key_hash, name="Key 1")
    key_hash2 = hash_api_key("test-key-2")
    repo.create_api_key(key_id="k2", key_hash=key_hash2, name="Key 2")

    keys = repo.list_api_keys()
    assert len(keys) == 2
    names = {k["name"] for k in keys}
    assert names == {"Key 1", "Key 2"}


def test_auth_disabled_without_env_var(tmp_path, monkeypatch):
    """When PROVENANCE_API_KEY is unset, /v1/* should work without a key."""
    monkeypatch.delenv("PROVENANCE_API_KEY", raising=False)
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        resp = c.post("/v1/analyze", json={"text": "no auth test"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Usage Tracking
# ---------------------------------------------------------------------------


def test_usage_recorded_after_analyze(client, repo):
    """Usage should be recorded after a successful /v1/analyze call."""
    resp = client.post("/v1/analyze", json={"text": "usage test"}, headers=_auth())
    assert resp.status_code == 200

    # Check usage was recorded
    summaries = repo.get_usage_summary()
    assert summaries["total_requests"] >= 1
    assert summaries["successful_requests"] >= 1


def test_usage_by_endpoint(client, repo):
    """Usage should be broken down by endpoint."""
    client.post("/v1/analyze", json={"text": "ep test"}, headers=_auth())
    client.get("/v1/analyses", headers=_auth())

    summary = repo.get_usage_summary()
    assert "/v1/analyze" in summary["by_endpoint"]
    assert "/v1/analyses" in summary["by_endpoint"]


def test_usage_by_detector(client, repo):
    """Usage should track the detector used."""
    client.post("/v1/analyze", json={"text": "det test", "detectors": ["unicode"]}, headers=_auth())

    summary = repo.get_usage_summary()
    assert "unicode" in summary["by_detector"]


def test_usage_character_counting(client, repo):
    """Usage should track total characters analyzed."""
    client.post("/v1/analyze", json={"text": "abc"}, headers=_auth())  # 3 chars
    client.post("/v1/analyze", json={"text": "hello world"}, headers=_auth())  # 11 chars

    summary = repo.get_usage_summary()
    assert summary["total_characters"] >= 14


def test_usage_endpoint(client, repo):
    """GET /v1/usage should return usage summary."""
    client.post("/v1/analyze", json={"text": "endpoint test"}, headers=_auth())

    resp = client.get("/v1/usage", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "total_characters" in data
    assert "by_endpoint" in data
    assert "by_detector" in data
    assert data["total_requests"] >= 1


def test_usage_requires_auth(client):
    """GET /v1/usage should require authentication."""
    resp = client.get("/v1/usage")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Daily Limits
# ---------------------------------------------------------------------------


def test_no_limits_by_default(client, monkeypatch):
    """Without PROVENANCE_DAILY_REQUEST_LIMIT, requests should pass."""
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)
    resp = client.post("/v1/analyze", json={"text": "no limits"}, headers=_auth())
    assert resp.status_code == 200


def test_request_limit_enforced(client, monkeypatch):
    """PROVENANCE_DAILY_REQUEST_LIMIT should block requests when exceeded."""
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
    """PROVENANCE_DAILY_CHARACTER_LIMIT should block requests when exceeded."""
    monkeypatch.setenv("PROVENANCE_DAILY_CHARACTER_LIMIT", "10")
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)

    resp1 = client.post("/v1/analyze", json={"text": "short"}, headers=_auth())
    assert resp1.status_code == 200
    resp2 = client.post("/v1/analyze", json={"text": "a" * 20}, headers=_auth())
    assert resp2.status_code == 429
    assert "Daily character limit" in resp2.json()["detail"]


# ---------------------------------------------------------------------------
# Response Metadata
# ---------------------------------------------------------------------------


def test_analyze_returns_duration(client):
    """POST /v1/analyze should include duration_ms in the response."""
    resp = client.post("/v1/analyze", json={"text": "duration test"}, headers=_auth())
    data = resp.json()
    assert "duration_ms" in data
    assert data["duration_ms"] is not None
    assert data["duration_ms"] >= 0


def test_analyze_response_schema_extended(client):
    """AnalyzeResponse should contain all expected fields."""
    resp = client.post("/v1/analyze", json={"text": "schema test"}, headers=_auth())
    data = resp.json()
    for key in [
        "analysis_id", "engine_version", "status", "text_stats",
        "results", "limitations", "metadata", "duration_ms",
    ]:
        assert key in data, f"missing key: {key}"


# ---------------------------------------------------------------------------
# Privacy / Security
# ---------------------------------------------------------------------------


def test_raw_text_not_in_usage(client, repo):
    """Raw input text must never appear in usage records."""
    secret = "SENSITIVE-TEXT-FOR-USAGE-12345"
    client.post("/v1/analyze", json={"text": secret}, headers=_auth())

    import sqlite3
    conn = sqlite3.connect(str(repo._db_path))
    rows = conn.execute("SELECT * FROM usage_records").fetchall()
    conn.close()
    for row in rows:
        serialized = json.dumps(row)
        assert secret not in serialized


def test_api_key_not_in_usage(client, repo):
    """API key must never appear in usage records."""
    client.post("/v1/analyze", json={"text": "key in usage test"}, headers=_auth())

    import sqlite3
    conn = sqlite3.connect(str(repo._db_path))
    rows = conn.execute("SELECT * FROM usage_records").fetchall()
    conn.close()
    for row in rows:
        serialized = json.dumps(row)
        assert API_KEY not in serialized


def test_watermark_keys_not_in_usage(client, repo):
    """Watermark keys must never appear in usage records."""
    client.post("/v1/analyze", json={"text": "wm usage test"}, headers=_auth())

    import sqlite3
    conn = sqlite3.connect(str(repo._db_path))
    rows = conn.execute("SELECT * FROM usage_records").fetchall()
    conn.close()
    for row in rows:
        serialized = json.dumps(row)
        assert "secret_key" not in serialized.lower()
        assert "private_key" not in serialized.lower()


# ---------------------------------------------------------------------------
# Database Schema / CRUD
# ---------------------------------------------------------------------------


def test_api_keys_table_exists(repo):
    """The api_keys table should be created on init."""
    import sqlite3
    conn = sqlite3.connect(str(repo._db_path))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "api_keys" in tables


def test_usage_records_table_exists(repo):
    """The usage_records table should be created on init."""
    import sqlite3
    conn = sqlite3.connect(str(repo._db_path))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "usage_records" in tables


def test_create_and_retrieve_api_key(repo):
    """API key CRUD should work."""
    key_hash = hash_api_key("my-test-key")
    repo.create_api_key(key_id="k1", key_hash=key_hash, name="Test")

    record = repo.get_api_key_by_hash(key_hash)
    assert record is not None
    assert record["key_id"] == "k1"
    assert record["name"] == "Test"
    assert record["status"] == "active"


def test_revoke_api_key(repo):
    """Revoking a key should set status to revoked."""
    key_hash = hash_api_key("revoke-test")
    repo.create_api_key(key_id="rk1", key_hash=key_hash, name="Revoke Test")
    repo.revoke_api_key("rk1")

    record = repo.get_api_key_by_hash(key_hash)
    assert record["status"] == "revoked"


def test_get_nonexistent_api_key(repo):
    """Querying a non-existent key hash should return None."""
    assert repo.get_api_key_by_hash("nonexistent-hash") is None


def test_usage_today(repo):
    """get_usage_today should return today's request and character counts."""
    repo.record_usage(
        key_id="today-test",
        endpoint="/v1/analyze",
        status_code=200,
        duration_ms=10.0,
        character_count=42,
        success=True,
    )
    reqs, chars = repo.get_usage_today("today-test")
    assert reqs == 1
    assert chars == 42


def test_usage_summary_empty(repo):
    """Empty usage should return zeroed summary."""
    summary = repo.get_usage_summary()
    assert summary["total_requests"] == 0
    assert summary["total_characters"] == 0
    assert summary["by_endpoint"] == {}
    assert summary["by_detector"] == {}


# ---------------------------------------------------------------------------
# Mocked PostgreSQL API key/usage tests
# ---------------------------------------------------------------------------


def test_pg_api_key_and_usage_mocked():
    """PostgresRepository API key and usage methods work (mocked)."""
    from tests.test_api import _make_mock_pg
    from provenance.api.db import PostgresRepository

    mock_pg, mock_conn, mock_cursor = _make_mock_pg()

    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")

        # create key
        key_hash = hash_api_key("pg-key")
        repo.create_api_key(key_id="pg-k1", key_hash=key_hash, name="PG Key")
        # verify mock was called
        assert mock_cursor.execute.called

        # record usage
        repo.record_usage(
            key_id="pg-k1",
            endpoint="/v1/analyze",
            status_code=200,
            duration_ms=5.0,
            character_count=10,
            success=True,
            detector="unicode",
        )
        assert mock_cursor.execute.called

        repo.close()
