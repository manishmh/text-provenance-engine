"""Tests for the provenance HTTP API.

Uses a temporary SQLite database for each test.  Does not require a real
Hugging Face model -- the Unicode detector is sufficient for basic API tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import sys
import uuid
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import (
    AnalysisRepository,
    PostgresRepository,
    SqliteRepository,
    _text_hash,
    create_repository,
)
from provenance.engine import ENGINE_VERSION

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "test-secret-key-123"


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    """Ensure PROVENANCE_API_KEY is set for every test in this module."""
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)


@pytest.fixture()
def client(tmp_path):
    """Create a test client with auth enabled and a temporary SQLite database."""
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_auth(tmp_path, monkeypatch):
    """Create a test client with authentication DISABLED."""
    monkeypatch.delenv("PROVENANCE_API_KEY", raising=False)
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def repo(tmp_path):
    """Create a real repository for direct DB tests."""
    db_path = tmp_path / "test.db"
    r = SqliteRepository(db_path)
    yield r
    r.close()


def _auth_headers(api_key: str = API_KEY) -> dict[str, str]:
    return {"X-API-Key": api_key}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_health_no_auth_required(client):
    """Health endpoint must be accessible without any API key."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_v1_requires_api_key(client):
    """All /v1/* endpoints must reject requests without an API key."""
    for method, path in [
        ("POST", "/v1/analyze"),
        ("GET", "/v1/analyses"),
        ("GET", "/v1/analyses/fake-id"),
    ]:
        if method == "POST":
            resp = client.post(path, json={"text": "x"})
        else:
            resp = client.get(path)
        if path == "/v1/analyses/fake-id":
            assert resp.status_code in (401, 404), f"{method} {path} should be 401 or 404"
        else:
            assert resp.status_code == 401, f"{method} {path} should be 401"


def test_v1_rejects_invalid_api_key(client):
    """Requests with a wrong API key must get 403."""
    resp = client.post(
        "/v1/analyze",
        json={"text": "test"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_v1_accepts_valid_api_key(client):
    """Requests with the correct API key must succeed."""
    resp = client.post(
        "/v1/analyze",
        json={"text": "test"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200


def test_v1_analyses_requires_auth(client):
    resp = client.get("/v1/analyses")
    assert resp.status_code == 401


def test_v1_analyses_requires_auth_get_by_id(client):
    resp = client.get("/v1/analyses/some-id")
    assert resp.status_code == 401


def test_auth_disabled_without_env_var(client_no_auth):
    """When PROVENANCE_API_KEY is unset, /v1/* should work without a key."""
    resp = client_no_auth.post("/v1/analyze", json={"text": "no-auth test"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/analyze
# ---------------------------------------------------------------------------


def test_analyze_unicode_only(client):
    resp = client.post("/v1/analyze", json={"text": "Hello, world!"}, headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_version"] == ENGINE_VERSION
    assert data["status"] == "ok"
    assert data["analysis_id"]
    assert data["text_stats"]["character_count"] == 13
    assert len(data["results"]) >= 1
    assert data["results"][0]["detector"] == "unicode"


def test_analyze_explicit_detectors(client):
    resp = client.post(
        "/v1/analyze",
        json={"text": "Test text", "detectors": ["unicode"]},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["detector"] == "unicode"


def test_analyze_empty_text_rejected(client):
    resp = client.post("/v1/analyze", json={"text": ""}, headers=_auth_headers())
    assert resp.status_code == 422


def test_analyze_missing_text_rejected(client):
    resp = client.post("/v1/analyze", json={}, headers=_auth_headers())
    assert resp.status_code == 422


def test_analyze_oversized_text_rejected(client):
    from provenance.api.models import MAX_TEXT_LENGTH
    resp = client.post(
        "/v1/analyze",
        json={"text": "x" * (MAX_TEXT_LENGTH + 1)},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


def test_analyze_unknown_detector_rejected(client):
    resp = client.post(
        "/v1/analyze",
        json={"text": "test", "detectors": ["nonexistent"]},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


def test_analyze_watermark_detector_requires_config(client):
    resp = client.post(
        "/v1/analyze",
        json={"text": "test", "detectors": ["unicode", "kgw"]},
        headers=_auth_headers(),
    )
    assert resp.status_code == 422


def test_analyze_persists_result(client):
    resp = client.post("/v1/analyze", json={"text": "Persist test"}, headers=_auth_headers())
    analysis_id = resp.json()["analysis_id"]
    resp2 = client.get(f"/v1/analyses/{analysis_id}", headers=_auth_headers())
    assert resp2.status_code == 200
    assert resp2.json()["analysis_id"] == analysis_id


def test_analyze_response_schema(client):
    resp = client.post("/v1/analyze", json={"text": "Schema check"}, headers=_auth_headers())
    data = resp.json()
    for key in [
        "analysis_id", "engine_version", "status", "text_stats",
        "results", "limitations", "metadata",
    ]:
        assert key in data, f"missing key: {key}"
    if data["results"]:
        item = data["results"][0]
        for key in ["detector", "detector_version", "status", "detected",
                     "implementation_kind", "compatibility", "score",
                     "threshold", "confidence", "evidence", "limitations"]:
            assert key in item, f"missing result key: {key}"


# ---------------------------------------------------------------------------
# GET /v1/analyses/{analysis_id}
# ---------------------------------------------------------------------------


def test_get_analysis(client):
    resp = client.post("/v1/analyze", json={"text": "Get test"}, headers=_auth_headers())
    aid = resp.json()["analysis_id"]
    resp2 = client.get(f"/v1/analyses/{aid}", headers=_auth_headers())
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["analysis_id"] == aid
    assert "result" in data
    assert "text_hash" in data


def test_get_analysis_not_found(client):
    resp = client.get("/v1/analyses/nonexistent-id", headers=_auth_headers())
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/analyses
# ---------------------------------------------------------------------------


def test_list_analyses_empty(client):
    resp = client.get("/v1/analyses", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyses"] == []
    assert data["total"] == 0


def test_list_analyses_with_results(client):
    for i in range(3):
        client.post("/v1/analyze", json={"text": f"List test {i}"}, headers=_auth_headers())
    resp = client.get("/v1/analyses", headers=_auth_headers())
    data = resp.json()
    assert data["total"] == 3
    assert len(data["analyses"]) == 3


def test_list_analyses_pagination(client):
    for i in range(5):
        client.post("/v1/analyze", json={"text": f"Page test {i}"}, headers=_auth_headers())
    resp = client.get("/v1/analyses?limit=2&offset=0", headers=_auth_headers())
    data = resp.json()
    assert data["total"] == 5
    assert len(data["analyses"]) == 2
    assert data["limit"] == 2
    assert data["offset"] == 0

    resp2 = client.get("/v1/analyses?limit=2&offset=2", headers=_auth_headers())
    data2 = resp2.json()
    assert len(data2["analyses"]) == 2


def test_list_limit_validation(client):
    resp = client.get("/v1/analyses?limit=0", headers=_auth_headers())
    assert resp.status_code == 422
    resp2 = client.get("/v1/analyses?limit=101", headers=_auth_headers())
    assert resp2.status_code == 422


# ---------------------------------------------------------------------------
# Input hashing and privacy
# ---------------------------------------------------------------------------


def test_raw_text_not_stored(client, tmp_path):
    """The raw input text should not appear in the database."""
    secret_text = "This is sensitive content that must not be stored"
    resp = client.post("/v1/analyze", json={"text": secret_text}, headers=_auth_headers())
    aid = resp.json()["analysis_id"]
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT result_json FROM analyses WHERE analysis_id = ?", (aid,)).fetchone()
    conn.close()
    assert row is not None
    assert secret_text not in row[0]


def test_text_hash_is_sha256(client):
    text = "Hash verification test"
    resp = client.post("/v1/analyze", json={"text": text}, headers=_auth_headers())
    aid = resp.json()["analysis_id"]
    resp2 = client.get(f"/v1/analyses/{aid}", headers=_auth_headers())
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert resp2.json()["text_hash"] == expected_hash


def test_same_text_same_hash(client):
    text = "Deterministic hash test"
    r1 = client.post("/v1/analyze", json={"text": text}, headers=_auth_headers())
    r2 = client.post("/v1/analyze", json={"text": text}, headers=_auth_headers())
    h1 = client.get(f"/v1/analyses/{r1.json()['analysis_id']}", headers=_auth_headers()).json()["text_hash"]
    h2 = client.get(f"/v1/analyses/{r2.json()['analysis_id']}", headers=_auth_headers()).json()["text_hash"]
    assert h1 == h2


# ---------------------------------------------------------------------------
# No secret key leakage
# ---------------------------------------------------------------------------


def test_no_raw_watermark_key_in_responses(client):
    """Raw watermark keys should never appear in API responses."""
    resp = client.post("/v1/analyze", json={"text": "Key leakage test"}, headers=_auth_headers())
    resp_json = json.dumps(resp.json())
    assert API_KEY not in resp_json
    for pattern in ["secret_key", "private_key"]:
        assert pattern.lower() not in resp_json.lower()


# ---------------------------------------------------------------------------
# Direct SQLite repository tests
# ---------------------------------------------------------------------------


def test_repo_save_and_get(repo):
    text = "Direct repo test"
    analysis_id = repo.save(
        text=text,
        engine_version="0.1.0",
        detectors=["unicode"],
        result_dict={"status": "ok", "text_stats": {"character_count": 14}},
    )
    record = repo.get(analysis_id)
    assert record is not None
    assert record["analysis_id"] == analysis_id
    assert record["text_hash"] == _text_hash(text)


def test_repo_list(repo):
    for i in range(3):
        repo.save(
            text=f"List {i}",
            engine_version="0.1.0",
            detectors=["unicode"],
            result_dict={"status": "ok", "text_stats": {"character_count": 6}},
        )
    summaries, total = repo.list_analyses(limit=10, offset=0)
    assert total == 3
    assert len(summaries) == 3


def test_repo_get_nonexistent(repo):
    assert repo.get("nonexistent") is None


def test_text_hash_function():
    h = _text_hash("hello")
    assert len(h) == 64
    assert h == hashlib.sha256(b"hello").hexdigest()


# ---------------------------------------------------------------------------
# Database factory / configuration tests
# ---------------------------------------------------------------------------


def test_factory_default_sqlite(tmp_path):
    """No DATABASE_URL → SQLite at data/provenance.db."""
    with patch.dict("os.environ", {}, clear=True):
        repo = create_repository()
        assert isinstance(repo, SqliteRepository)
        repo.close()


def test_factory_explicit_sqlite_url(tmp_path):
    """sqlite:///path → SqliteRepository."""
    db_path = tmp_path / "explicit.db"
    repo = create_repository(f"sqlite:///{db_path}")
    assert isinstance(repo, SqliteRepository)
    # Verify it works
    aid = repo.save(text="x", engine_version="0.1.0", detectors=[], result_dict={"status": "ok", "text_stats": {}})
    assert repo.get(aid) is not None
    repo.close()


def test_factory_explicit_file_path(tmp_path):
    """Plain path → SqliteRepository."""
    db_path = tmp_path / "plain.db"
    repo = create_repository(str(db_path))
    assert isinstance(repo, SqliteRepository)
    repo.close()


def _make_mock_pg():
    """Create mock psycopg2 module + connection for testing."""
    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_psycopg2.connect.return_value = mock_conn
    return mock_psycopg2, mock_conn, mock_cursor


def test_postgres_schema_matches_sqlite_required_tables():
    """Both backends bootstrap the same application table set."""
    from provenance.api.db import (
        _SCHEMA_BENCHMARK_RUNS,
        _SCHEMA_IDENTITY,
        _SCHEMA_PG,
        _SCHEMA_SQLITE,
    )

    def tables(*blocks):
        return {
            match.group(1)
            for block in blocks
            for match in re.finditer(
                r"CREATE TABLE IF NOT EXISTS\s+([a-z_]+)", block, re.IGNORECASE
            )
        }

    shared = (_SCHEMA_BENCHMARK_RUNS, _SCHEMA_IDENTITY)
    expected = {
        "analyses", "api_keys", "usage_records", "jobs", "benchmark_runs",
        "app_users", "anonymous_visitors", "usage_events",
    }
    assert tables(_SCHEMA_SQLITE, *shared) == expected
    assert tables(_SCHEMA_PG, *shared) == expected


def test_postgres_schema_migration_is_transaction_safe_and_idempotent():
    from provenance.api.db import _MIGRATE_PG

    assert _MIGRATE_PG.count("ADD COLUMN IF NOT EXISTS") == 2
    mock_pg, mock_conn, mock_cursor = _make_mock_pg()
    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")
    statements = [call.args[0] for call in mock_cursor.execute.call_args_list]
    assert statements[-1] == _MIGRATE_PG
    mock_conn.commit.assert_called_once()
    mock_conn.rollback.assert_not_called()
    repo.close()


def test_postgres_schema_failure_rolls_back_and_logs_no_credentials(caplog):
    mock_pg, mock_conn, mock_cursor = _make_mock_pg()
    mock_cursor.execute.side_effect = [None, RuntimeError("contains-super-secret")]
    dsn = "postgresql://app:contains-super-secret@example.invalid/database"

    with caplog.at_level(logging.ERROR, logger="provenance.api"):
        with patch.dict("sys.modules", {"psycopg2": mock_pg}):
            with pytest.raises(RuntimeError, match="contains-super-secret"):
                PostgresRepository(dsn)

    mock_conn.rollback.assert_called_once()
    mock_conn.commit.assert_not_called()
    mock_conn.close.assert_called_once()
    assert "stage=benchmark_tables" in caplog.text
    assert "contains-super-secret" not in caplog.text
    assert dsn not in caplog.text


def test_factory_pg_url():
    """postgresql:// URL → PostgresRepository (mocked)."""
    mock_pg, mock_conn, _ = _make_mock_pg()
    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = create_repository("postgresql://user:pass@localhost/provenance")
        assert isinstance(repo, PostgresRepository)
        repo.close()


def test_factory_postgres_url():
    """postgres:// URL → PostgresRepository (mocked)."""
    mock_pg, mock_conn, _ = _make_mock_pg()
    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = create_repository("postgres://user:pass@localhost/provenance")
        assert isinstance(repo, PostgresRepository)
        repo.close()


def test_postgres_repository_save_and_get():
    """PostgresRepository save/get round-trip (mocked)."""
    mock_pg, mock_conn, mock_cursor = _make_mock_pg()

    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")

        # save
        aid = repo.save(
            text="pg test",
            engine_version="0.1.0",
            detectors=["unicode"],
            result_dict={"status": "ok", "text_stats": {"character_count": 7, "token_count": 1}},
        )
        assert aid  # UUID string

        # get — simulate a row
        mock_cursor.fetchone.return_value = (
            aid, "2025-01-01T00:00:00+00:00", "0.1.0", _text_hash("pg test"),
            7, 1, "ok", '["unicode"]', json.dumps({"status": "ok"}),
        )
        mock_cursor.description = [
            ("analysis_id",), ("timestamp",), ("engine_version",), ("text_hash",),
            ("character_count",), ("token_count",), ("status",), ("detectors",),
            ("result_json",),
        ]
        record = repo.get(aid)
        assert record is not None
        assert record["analysis_id"] == aid
        assert record["detectors"] == ["unicode"]
        assert record["result"] == {"status": "ok"}

        # list
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = [
            ("id-1", "2025-01-01T00:00:00+00:00", "0.1.0", "abc123hash", 5, 1, "ok", '["unicode"]'),
        ]
        summaries, total = repo.list_analyses(limit=10, offset=0)
        assert total == 1
        assert len(summaries) == 1

        repo.close()


# ---------------------------------------------------------------------------
# Engine still works
# ---------------------------------------------------------------------------


def test_engine_unchanged():
    from provenance.engine import ProvenanceEngine
    engine = ProvenanceEngine()
    result = engine.analyze("Quick engine test")
    assert result.status == "ok"
    assert result.engine_version == ENGINE_VERSION


# ---------------------------------------------------------------------------
# Live PostgreSQL integration tests
# ---------------------------------------------------------------------------
#
# These tests run only when a PostgreSQL server is reachable via the
# PROVENANCE_TEST_PG_DSN environment variable.  If the variable is unset
# the tests are silently skipped — no infrastructure provisioning is done.
#
# Example:
#   export PROVENANCE_TEST_PG_DSN='postgresql://edith@/countzero?options=-c search_path%3Dprovenance_test'
#   pytest tests/test_api.py -k pg_live -q
# ---------------------------------------------------------------------------

import os

_pg_dsn = os.environ.get("PROVENANCE_TEST_PG_DSN")
_pg_reason = "PROVENANCE_TEST_PG_DSN not set — skipping live PostgreSQL tests"


@pytest.fixture()
def pg_repo():
    """Create a real PostgresRepository against an available server.

    Uses a dedicated schema for test isolation.  Falls back to skip
    when no DSN is configured.
    """
    if _pg_dsn is None:
        pytest.skip(_pg_reason)

    import psycopg2

    # Create a unique schema for test isolation
    schema = f"test_{uuid.uuid4().hex[:8]}"

    # Connect to base DB, create schema, then set search_path on
    # the PostgresRepository's connection after it initializes.
    base_conn = psycopg2.connect(_pg_dsn)
    base_conn.autocommit = True
    cur = base_conn.cursor()
    cur.execute(f"CREATE SCHEMA {schema}")
    base_conn.close()

    # libpq accepts session settings through the URI's `options` parameter.
    # Rebuild the query with standard percent encoding so DSNs that already
    # contain sslmode or pooler options remain valid.
    parts = urlsplit(_pg_dsn)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != "options"]
    query.append(("options", f"-c search_path={schema}"))
    dsn_with_schema = urlunsplit(parts._replace(
        query=urlencode(query, quote_via=quote)
    ))

    r = PostgresRepository(dsn_with_schema)
    yield r

    # Cleanup: drop the test schema
    r.close()
    base_conn = psycopg2.connect(_pg_dsn)
    base_conn.autocommit = True
    cur = base_conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
    base_conn.close()


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_schema_init(pg_repo):
    """Schema is initialized automatically on PostgresRepository creation."""
    # If we got here, schema was created in the fixture
    assert pg_repo is not None


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_save_and_get(pg_repo):
    text = "Live PG save/get test"
    aid = pg_repo.save(
        text=text,
        engine_version="0.2.0",
        detectors=["unicode"],
        result_dict={"status": "ok", "text_stats": {"character_count": len(text), "token_count": 5}},
    )
    record = pg_repo.get(aid)
    assert record is not None
    assert record["analysis_id"] == aid
    assert record["text_hash"] == _text_hash(text)
    assert record["detectors"] == ["unicode"]
    assert record["result"]["status"] == "ok"


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_list_and_pagination(pg_repo):
    for i in range(5):
        pg_repo.save(
            text=f"List item {i}",
            engine_version="0.2.0",
            detectors=["unicode"],
            result_dict={"status": "ok", "text_stats": {"character_count": 10, "token_count": 2}},
        )
    summaries, total = pg_repo.list_analyses(limit=2, offset=0)
    assert total == 5
    assert len(summaries) == 2
    summaries2, total2 = pg_repo.list_analyses(limit=2, offset=4)
    assert len(summaries2) == 1


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_get_nonexistent(pg_repo):
    assert pg_repo.get("nonexistent-uuid") is None


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_text_hash_is_sha256(pg_repo):
    text = "PG hash check"
    aid = pg_repo.save(
        text=text,
        engine_version="0.2.0",
        detectors=[],
        result_dict={"status": "ok", "text_stats": {}},
    )
    record = pg_repo.get(aid)
    assert record["text_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_raw_text_not_stored(pg_repo):
    secret = "This sensitive text must not be in the database"
    aid = pg_repo.save(
        text=secret,
        engine_version="0.2.0",
        detectors=[],
        result_dict={"status": "ok", "text_stats": {}},
    )
    record = pg_repo.get(aid)
    serialized = json.dumps(record)
    assert secret not in serialized
    # Also verify via direct DB query
    cur = pg_repo._conn.cursor()
    cur.execute("SELECT result_json FROM analyses WHERE analysis_id = %s", (aid,))
    row = cur.fetchone()
    assert row is not None
    assert secret not in row[0]


@pytest.mark.skipif(_pg_dsn is None, reason=_pg_reason)
def test_pg_live_secrets_not_leaked(pg_repo):
    """API key and internal secrets must not appear in stored records."""
    aid = pg_repo.save(
        text="Secret leakage check",
        engine_version="0.2.0",
        detectors=[],
        result_dict={"status": "ok", "text_stats": {}, "metadata": {}},
    )
    record = pg_repo.get(aid)
    serialized = json.dumps(record)
    assert "X-API-Key" not in serialized
    assert "PROVENANCE_API_KEY" not in serialized
