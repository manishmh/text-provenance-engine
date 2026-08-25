"""Phase 3C tests: AnalysisService, async jobs, job lifecycle, concurrency.

Uses a temporary SQLite database for each test.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import SqliteRepository, _text_hash
from provenance.api.jobs import cleanup_old_jobs

API_KEY = "test-phase3c-key"
ADMIN_KEY = "test-phase3c-admin"


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
# AnalysisService
# ---------------------------------------------------------------------------


def test_analysis_service_basic():
    """AnalysisService should produce valid results."""
    from provenance.api.service import run_analysis
    result = run_analysis("Hello world", ["unicode"], None)
    assert result["status"] == "ok"
    assert "text_stats" in result
    assert "results" in result
    assert result["results"][0]["detector"] == "unicode"


def test_analysis_service_validate_detectors():
    from provenance.api.service import validate_detectors
    names, inc_unicode, needs = validate_detectors(["unicode"])
    assert names == ["unicode"]
    assert inc_unicode is True
    assert needs == set()

    names2, inc2, needs2 = validate_detectors(["unicode", "kgw"])
    assert "kgw" in needs2


def test_analysis_service_invalid_detector():
    from provenance.api.service import validate_detectors
    with pytest.raises(ValueError, match="Unknown detector"):
        validate_detectors(["nonexistent"])


def test_analysis_service_privacy():
    """Raw text should not appear in the result dict."""
    from provenance.api.service import run_analysis
    secret = "SECRET-TEXT-12345"
    result = run_analysis(secret, ["unicode"], None)
    serialized = json.dumps(result)
    assert secret not in serialized


# ---------------------------------------------------------------------------
# Async Analysis — POST /v1/analyze/async
# ---------------------------------------------------------------------------


def test_async_analyze_returns_202(client):
    resp = client.post("/v1/analyze/async", json={"text": "async test"}, headers=_auth())
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["job_id"]
    assert len(data["job_id"]) == 36  # UUID


def test_async_analyze_requires_auth(client):
    resp = client.post("/v1/analyze/async", json={"text": "x"})
    assert resp.status_code == 401


def test_async_analyze_rejects_invalid_detectors(client):
    resp = client.post(
        "/v1/analyze/async",
        json={"text": "x", "detectors": ["bad"]},
        headers=_auth(),
    )
    assert resp.status_code == 422


def test_async_analyze_enforces_daily_limits(client, monkeypatch):
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "1")
    resp1 = client.post("/v1/analyze/async", json={"text": "r1"}, headers=_auth())
    assert resp1.status_code == 202
    # Wait for background job to complete so usage is recorded
    time.sleep(2)
    resp2 = client.post("/v1/analyze/async", json={"text": "r2"}, headers=_auth())
    assert resp2.status_code == 429


# ---------------------------------------------------------------------------
# Job Lifecycle
# ---------------------------------------------------------------------------


def test_job_completes_and_result_available(client):
    """Async job should complete and result should be fetchable."""
    resp = client.post("/v1/analyze/async", json={"text": "lifecycle test"}, headers=_auth())
    job_id = resp.json()["job_id"]

    # Wait for background execution
    time.sleep(2)

    resp2 = client.get(f"/v1/jobs/{job_id}", headers=_auth())
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["status"] == "completed"
    assert data["result"] is not None
    assert data["result"]["status"] == "ok"
    assert data["completed_at"] is not None
    assert data["duration_ms"] is not None


def test_job_started_at_recorded(client):
    resp = client.post("/v1/analyze/async", json={"text": "started_at test"}, headers=_auth())
    job_id = resp.json()["job_id"]
    time.sleep(2)

    resp2 = client.get(f"/v1/jobs/{job_id}", headers=_auth())
    data = resp2.json()
    assert data["started_at"] is not None
    assert data["completed_at"] is not None


def test_job_not_found(client):
    resp = client.get("/v1/jobs/nonexistent", headers=_auth())
    assert resp.status_code == 404


def test_job_ownership_enforced(client, repo):
    """A key should only see its own jobs."""
    # Create a job with the normal key
    resp = client.post("/v1/analyze/async", json={"text": "ownership test"}, headers=_auth())
    job_id = resp.json()["job_id"]
    time.sleep(1)

    # Create a second key and try to access the job
    resp2 = client.post("/v1/api-keys", json={"name": "Other Key"}, headers=_admin_auth())
    other_key = resp2.json()["key"]

    resp3 = client.get(f"/v1/jobs/{job_id}", headers=_auth(other_key))
    assert resp3.status_code == 404  # Not found for this key


def test_admin_can_access_any_job(client):
    """Admin key can access any job."""
    resp = client.post("/v1/analyze/async", json={"text": "admin access test"}, headers=_auth())
    job_id = resp.json()["job_id"]
    time.sleep(1)

    resp2 = client.get(f"/v1/jobs/{job_id}", headers=_admin_auth())
    assert resp2.status_code == 200


def test_job_list_requires_key(client):
    """Job listing requires an actual API key, not _none or _admin."""
    resp = client.get("/v1/jobs", headers=_auth())
    assert resp.status_code == 200  # _legacy key works
    resp2 = client.get("/v1/jobs", headers=_admin_auth())
    assert resp2.status_code == 400  # admin identity rejected


def test_job_list_returns_jobs(client):
    client.post("/v1/analyze/async", json={"text": "list test 1"}, headers=_auth())
    client.post("/v1/analyze/async", json={"text": "list test 2"}, headers=_auth())
    time.sleep(1)

    resp = client.get("/v1/jobs", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 2
    assert len(data["jobs"]) >= 2


# ---------------------------------------------------------------------------
# Job Persistence — raw text never stored
# ---------------------------------------------------------------------------


def test_job_raw_text_not_persisted(client, tmp_path):
    secret = "SECRET-INPUT-FOR-JOB-12345"
    resp = client.post("/v1/analyze/async", json={"text": secret}, headers=_auth())
    job_id = resp.json()["job_id"]
    time.sleep(1)

    # Check DB directly
    import sqlite3
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    assert row is not None
    # Serialize all columns and check
    serialized = json.dumps(row)
    assert secret not in serialized


def test_job_input_hash_matches(client):
    text = "Hash verification for job"
    resp = client.post("/v1/analyze/async", json={"text": text}, headers=_auth())
    job_id = resp.json()["job_id"]
    time.sleep(1)

    resp2 = client.get(f"/v1/jobs/{job_id}", headers=_auth())
    # input_hash is in the DB but not exposed in response; check via DB
    from provenance.api.state import get_repo
    repo = get_repo()
    job = repo.get_job(job_id)
    assert job["input_hash"] == _text_hash(text)


# ---------------------------------------------------------------------------
# Job Retention Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_removes_old_completed_jobs(repo):
    """cleanup_jobs should remove completed jobs older than the cutoff."""
    from datetime import datetime, timedelta, timezone

    # Create a job
    repo.create_job(
        job_id="old-job-1",
        key_id="test-key",
        detectors=["unicode"],
        config_path=None,
        input_hash="abc",
        character_count=5,
    )
    # Complete it with an old timestamp
    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    repo._conn.execute(
        "UPDATE jobs SET status = 'completed', completed_at = ?, result_json = '{}' WHERE job_id = 'old-job-1'",
        (old_time,),
    )
    repo._conn.commit()

    # Create a recent completed job
    repo.create_job(
        job_id="recent-job-1",
        key_id="test-key",
        detectors=["unicode"],
        config_path=None,
        input_hash="def",
        character_count=5,
    )
    repo.complete_job(job_id="recent-job-1", result_json="{}", duration_ms=10.0)

    # Cleanup with 24h retention
    cleaned = cleanup_old_jobs(repo)
    assert cleaned == 1

    # Old job should be gone, recent should remain
    assert repo.get_job("old-job-1") is None
    assert repo.get_job("recent-job-1") is not None


def test_cleanup_keeps_queued_and_running_jobs(repo):
    """Only completed/failed jobs should be cleaned up."""
    from datetime import datetime, timedelta, timezone

    repo.create_job(
        job_id="running-job",
        key_id="test-key",
        detectors=["unicode"],
        config_path=None,
        input_hash="abc",
        character_count=5,
    )
    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    repo.update_job_status(job_id="running-job", status="running", started_at=old_time)

    cleaned = cleanup_old_jobs(repo)
    assert cleaned == 0
    assert repo.get_job("running-job") is not None


# ---------------------------------------------------------------------------
# Concurrency Limit
# ---------------------------------------------------------------------------


def test_max_background_jobs_respected(monkeypatch):
    """PROVENANCE_MAX_BACKGROUND_JOBS should limit the executor."""
    import provenance.api.jobs as jobs_mod
    monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "1")

    # Reset executor to pick up new config
    jobs_mod._executor = None

    executor = jobs_mod.get_executor()
    assert executor._max_workers == 1

    # Cleanup
    jobs_mod.shutdown_executor()


# ---------------------------------------------------------------------------
# Sync API still works
# ---------------------------------------------------------------------------


def test_sync_analyze_still_works(client):
    resp = client.post("/v1/analyze", json={"text": "sync still works"}, headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["analysis_id"]


def test_sync_analyze_uses_service(client):
    """Sync endpoint should produce identical results to direct service call."""
    from provenance.api.service import run_analysis

    text = "Service comparison test"
    result_service = run_analysis(text, ["unicode"], None)

    resp = client.post("/v1/analyze", json={"text": text}, headers=_auth())
    result_api = resp.json()

    # Same detector results
    assert result_api["results"][0]["detector"] == result_service["results"][0]["detector"]
    assert result_api["status"] == result_service["status"]


# ---------------------------------------------------------------------------
# Mocked PostgreSQL job methods
# ---------------------------------------------------------------------------


def test_pg_job_methods_mocked():
    from tests.test_api import _make_mock_pg
    from provenance.api.db import PostgresRepository

    mock_pg, mock_conn, mock_cursor = _make_mock_pg()
    with __import__("unittest.mock", fromlist=["patch"]).patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")
        repo.create_job(
            job_id="pg-job-1",
            key_id="pg-key",
            detectors=["unicode"],
            config_path=None,
            input_hash="abc123",
            character_count=10,
        )
        assert mock_cursor.execute.called

        # Mock fetchone to return a row-like object for get_job
        mock_cursor.fetchone.return_value = (
            "pg-job-1", "pg-key", "queued", "2025-01-01T00:00:00+00:00",
            None, None, "abc123", 10, '["unicode"]', None, None, None, None,
        )
        mock_cursor.description = [
            ("job_id",), ("key_id",), ("status",), ("created_at",),
            ("started_at",), ("completed_at",), ("input_hash",),
            ("character_count",), ("detectors",), ("config_path",),
            ("result_json",), ("error_message",), ("duration_ms",),
        ]
        job = repo.get_job("pg-job-1")
        assert job is not None
        assert job["job_id"] == "pg-job-1"
        assert job["detectors"] == ["unicode"]

        repo.update_job_status(job_id="pg-job-1", status="running")
        repo.complete_job(job_id="pg-job-1", result_json="{}", duration_ms=5.0)

        # Mock for list_jobs_by_key
        mock_cursor.fetchone.return_value = (1,)
        mock_cursor.fetchall.return_value = []
        repo.list_jobs_by_key(key_id="pg-key")

        repo.cleanup_jobs(older_than="2020-01-01T00:00:00+00:00")
        repo.close()
