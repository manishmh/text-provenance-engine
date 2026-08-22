"""Tests for the provenance HTTP API.

Uses a temporary SQLite database for each test.  Does not require a real
Hugging Face model -- the Unicode detector is sufficient for basic API tests.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import AnalysisRepository, _text_hash
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
    app = create_app(db_path=str(db_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def client_no_auth(tmp_path, monkeypatch):
    """Create a test client with authentication DISABLED."""
    monkeypatch.delenv("PROVENANCE_API_KEY", raising=False)
    db_path = tmp_path / "test.db"
    app = create_app(db_path=str(db_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def repo(tmp_path):
    """Create a real repository for direct DB tests."""
    db_path = tmp_path / "test.db"
    r = AnalysisRepository(db_path)
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
        # GET /v1/analyses/fake-id returns 401 (auth runs first) or 404
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
    # Auth runs before the route, so missing key gets 401
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
    """Text exceeding MAX_TEXT_LENGTH must be rejected."""
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


def test_analyze_persists_result(client, tmp_path):
    resp = client.post("/v1/analyze", json={"text": "Persist test"}, headers=_auth_headers())
    analysis_id = resp.json()["analysis_id"]

    # Verify it's stored (with auth)
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

    # Check the DB directly
    db_path = tmp_path / "test.db"
    import sqlite3
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
# Direct DB tests
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
    assert len(h) == 64  # SHA-256 hex
    assert h == hashlib.sha256(b"hello").hexdigest()


# ---------------------------------------------------------------------------
# Engine still works
# ---------------------------------------------------------------------------


def test_engine_unchanged():
    from provenance.engine import ProvenanceEngine
    engine = ProvenanceEngine()
    result = engine.analyze("Quick engine test")
    assert result.status == "ok"
    assert result.engine_version == ENGINE_VERSION
