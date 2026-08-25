"""Phase 3D tests: Job cancellation, retry, recovery, state transitions.

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

API_KEY = "test-phase3d-key"
ADMIN_KEY = "test-phase3d-admin"

# When using legacy PROVENANCE_API_KEY, auth returns key_id="_legacy"
LEGACY_KEY_ID = "_legacy"


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
# Job Cancellation — DELETE /v1/jobs/{job_id}
# ---------------------------------------------------------------------------


def test_cancel_queued_job_via_repo(client):
    """Cancelling a queued job atomically transitions to cancelled."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-q1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    assert r.get_job("cancel-q1")["status"] == "queued"

    resp = client.delete("/v1/jobs/cancel-q1", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"

    resp3 = client.get("/v1/jobs/cancel-q1", headers=_auth())
    assert resp3.json()["status"] == "cancelled"
    assert resp3.json()["error_message"] == "Job cancelled by user"


def test_cancel_race_queued_to_running(client):
    """If a job raced from queued to running, cancellation_requested is returned."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-race1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    r.update_job_status(job_id="cancel-race1", status="running", started_at="2025-01-01T00:00:00+00:00")

    resp = client.delete("/v1/jobs/cancel-race1", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancellation_requested"


def test_cancel_requires_ownership(client):
    """A key cannot cancel another key's job."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-owner1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    resp2 = client.post("/v1/api-keys", json={"name": "Other"}, headers=_admin_auth())
    other_key = resp2.json()["key"]

    resp3 = client.delete("/v1/jobs/cancel-owner1", headers=_auth(other_key))
    assert resp3.status_code == 404


def test_admin_can_cancel_any_job(client):
    """Admin key can cancel any job."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-admin1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    resp2 = client.delete("/v1/jobs/cancel-admin1", headers=_admin_auth())
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "cancelled"


def test_cancel_nonexistent_returns_404(client):
    resp = client.delete("/v1/jobs/nonexistent", headers=_auth())
    assert resp.status_code == 404


def test_cancel_completed_job_returns_409(client):
    """Cannot cancel a completed job."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-comp1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    r.complete_job(job_id="cancel-comp1", result_json="{}", duration_ms=10.0)

    resp = client.delete("/v1/jobs/cancel-comp1", headers=_auth())
    assert resp.status_code == 409
    assert "Cannot cancel" in resp.json()["detail"]


def test_cancel_failed_job_returns_409(client, repo):
    """Cannot cancel a failed job."""
    repo.create_job(
        job_id="cancel-fail1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.fail_job(job_id="cancel-fail1", error_message="test error", duration_ms=10.0)

    resp = client.delete("/v1/jobs/cancel-fail1", headers=_auth())
    assert resp.status_code == 409


def test_cancel_already_cancelled_returns_ok(client, repo):
    """Cancelling an already-cancelled job returns 200."""
    repo.create_job(
        job_id="cancel-already", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.cancel_job_if_status("cancel-already", expected_status="queued", new_status="cancelled")

    resp = client.delete("/v1/jobs/cancel-already", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_cancel_running_job_returns_cancellation_requested(client):
    """Cancelling a running job should return cancellation_requested."""
    from provenance.api.state import get_repo
    r = get_repo()
    r.create_job(
        job_id="cancel-run1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    r.update_job_status(job_id="cancel-run1", status="running", started_at="2025-01-01T00:00:00+00:00")

    resp = client.delete("/v1/jobs/cancel-run1", headers=_auth())
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancellation_requested"

    job = r.get_job("cancel-run1")
    assert job["status"] == "cancellation_requested"


def test_cancel_requires_auth(client):
    resp = client.delete("/v1/jobs/some-id")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Job Retry — POST /v1/jobs/{job_id}/retry
# ---------------------------------------------------------------------------


def test_retry_failed_job(client, repo):
    """Retrying a failed job should create a new job."""
    repo.create_job(
        job_id="retry-test-1", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.fail_job(job_id="retry-test-1", error_message="test failure", duration_ms=10.0)

    resp = client.post(
        "/v1/jobs/retry-test-1/retry",
        json={"text": "retry text"},
        headers=_auth(),
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data["job_id"] != "retry-test-1"
    assert data["status"] == "queued"

    new_job = repo.get_job(data["job_id"])
    assert new_job is not None
    assert new_job["parent_job_id"] == "retry-test-1"
    assert new_job["retry_count"] == 1

    time.sleep(2)
    resp2 = client.get(f"/v1/jobs/{data['job_id']}", headers=_auth())
    assert resp2.json()["status"] == "completed"


def test_retry_requires_ownership(client, repo):
    """A key cannot retry another key's job."""
    repo.create_job(
        job_id="retry-ownership", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.fail_job(job_id="retry-ownership", error_message="err", duration_ms=10.0)

    resp2 = client.post("/v1/api-keys", json={"name": "Other"}, headers=_admin_auth())
    other_key = resp2.json()["key"]

    resp = client.post(
        "/v1/jobs/retry-ownership/retry",
        json={"text": "retry text"},
        headers=_auth(other_key),
    )
    assert resp.status_code == 404


def test_retry_nonexistent_returns_404(client):
    resp = client.post(
        "/v1/jobs/nonexistent/retry",
        json={"text": "x"},
        headers=_auth(),
    )
    assert resp.status_code == 404


def test_retry_completed_job_returns_409(client, repo):
    """Cannot retry a completed job."""
    repo.create_job(
        job_id="retry-completed", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.complete_job(job_id="retry-completed", result_json="{}", duration_ms=10.0)

    resp = client.post(
        "/v1/jobs/retry-completed/retry",
        json={"text": "x"},
        headers=_auth(),
    )
    assert resp.status_code == 409
    assert "Only failed jobs" in resp.json()["detail"]


def test_retry_queued_job_returns_409(client, repo):
    """Cannot retry a queued job."""
    repo.create_job(
        job_id="retry-queued", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )

    resp = client.post(
        "/v1/jobs/retry-queued/retry",
        json={"text": "x"},
        headers=_auth(),
    )
    assert resp.status_code == 409


def test_retry_requires_auth(client):
    resp = client.post("/v1/jobs/some-id/retry", json={"text": "x"})
    assert resp.status_code == 401


def test_retry_uses_request_detectors(client, repo):
    """Retry with different detectors should use the new detectors."""
    repo.create_job(
        job_id="retry-dets", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.fail_job(job_id="retry-dets", error_message="err", duration_ms=10.0)

    resp = client.post(
        "/v1/jobs/retry-dets/retry",
        json={"text": "x", "detectors": ["unicode"]},
        headers=_auth(),
    )
    assert resp.status_code == 202
    new_job_id = resp.json()["job_id"]
    new_job = repo.get_job(new_job_id)
    assert new_job["detectors"] == ["unicode"]


def test_retry_incremental_retry_count(client, repo):
    """Multiple retries should increment retry_count."""
    repo.create_job(
        job_id="retry-multi", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.fail_job(job_id="retry-multi", error_message="err", duration_ms=10.0)

    resp1 = client.post("/v1/jobs/retry-multi/retry", json={"text": "r1"}, headers=_auth())
    assert resp1.status_code == 202
    job1_id = resp1.json()["job_id"]
    job1 = repo.get_job(job1_id)
    assert job1["retry_count"] == 1

    # Wait for background job to complete, then mark it as failed
    time.sleep(2)
    repo.fail_job(job_id=job1_id, error_message="err2", duration_ms=5.0)

    resp2 = client.post(f"/v1/jobs/{job1_id}/retry", json={"text": "r2"}, headers=_auth())
    assert resp2.status_code == 202
    job2_id = resp2.json()["job_id"]
    job2 = repo.get_job(job2_id)
    assert job2["retry_count"] == 2
    assert job2["parent_job_id"] == job1_id


def test_retry_running_job_returns_409(client, repo):
    """Cannot retry a running job."""
    repo.create_job(
        job_id="retry-running", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="abc", character_count=5,
    )
    repo.update_job_status(job_id="retry-running", status="running", started_at="2025-01-01T00:00:00+00:00")

    resp = client.post(
        "/v1/jobs/retry-running/retry",
        json={"text": "x"},
        headers=_auth(),
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Job State Model — Valid Transitions
# ---------------------------------------------------------------------------


def test_valid_transitions(repo):
    """Verify valid state transitions via direct repo calls."""
    # queued → running
    repo.create_job(job_id="t1", key_id="k", detectors=["unicode"], config_path=None, input_hash="a", character_count=1)
    repo.update_job_status(job_id="t1", status="running", started_at="2025-01-01T00:00:00+00:00")
    assert repo.get_job("t1")["status"] == "running"

    # running → completed
    repo.complete_job(job_id="t1", result_json="{}", duration_ms=5.0)
    assert repo.get_job("t1")["status"] == "completed"

    # queued → cancelled (atomic)
    repo.create_job(job_id="t2", key_id="k", detectors=["unicode"], config_path=None, input_hash="b", character_count=1)
    repo.cancel_job_if_status("t2", expected_status="queued", new_status="cancelled")
    assert repo.get_job("t2")["status"] == "cancelled"

    # queued → failed (via fail_job)
    repo.create_job(job_id="t3", key_id="k", detectors=["unicode"], config_path=None, input_hash="c", character_count=1)
    repo.fail_job(job_id="t3", error_message="err", duration_ms=10.0)
    assert repo.get_job("t3")["status"] == "failed"


def test_cancel_job_if_status_prevents_races(repo):
    """cancel_job_if_status should only succeed if status matches."""
    repo.create_job(
        job_id="race1", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.update_job_status(job_id="race1", status="running", started_at="2025-01-01T00:00:00+00:00")

    # Try to cancel expecting queued — should fail (returns False)
    result = repo.cancel_job_if_status("race1", expected_status="queued", new_status="cancelled")
    assert result is False
    assert repo.get_job("race1")["status"] == "running"

    # Now cancel expecting running — should succeed
    result = repo.cancel_job_if_status("race1", expected_status="running", new_status="cancelled")
    assert result is True
    assert repo.get_job("race1")["status"] == "cancelled"


def test_retry_count_persisted(repo):
    """retry_count should be stored and retrievable."""
    repo.create_job(
        job_id="rc1", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
        retry_count=3, parent_job_id="original-job",
    )
    job = repo.get_job("rc1")
    assert job["retry_count"] == 3
    assert job["parent_job_id"] == "original-job"


def test_retry_count_default_zero(repo):
    """retry_count defaults to 0 for new jobs."""
    repo.create_job(
        job_id="rc2", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    job = repo.get_job("rc2")
    assert job["retry_count"] == 0
    assert job["parent_job_id"] is None


# ---------------------------------------------------------------------------
# Recovery on Restart
# ---------------------------------------------------------------------------


def test_recover_stale_running_jobs(repo):
    """Running jobs should be marked failed on recovery."""
    repo.create_job(
        job_id="stale-run", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.update_job_status(job_id="stale-run", status="running", started_at="2025-01-01T00:00:00+00:00")

    recovered = repo.recover_stale_jobs()
    assert recovered == 1

    job = repo.get_job("stale-run")
    assert job["status"] == "failed"
    assert "interrupted by application restart" in job["error_message"]
    assert job["completed_at"] is not None
    assert job["duration_ms"] == 0


def test_recover_stale_queued_jobs(repo):
    """Queued jobs should be marked failed on recovery."""
    repo.create_job(
        job_id="stale-q", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )

    recovered = repo.recover_stale_jobs()
    assert recovered == 1

    job = repo.get_job("stale-q")
    assert job["status"] == "failed"
    assert "interrupted by application restart" in job["error_message"]


def test_recover_does_not_touch_completed_jobs(repo):
    """Completed jobs should not be affected by recovery."""
    repo.create_job(
        job_id="done", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.complete_job(job_id="done", result_json="{}", duration_ms=5.0)

    recovered = repo.recover_stale_jobs()
    assert recovered == 0

    job = repo.get_job("done")
    assert job["status"] == "completed"


def test_recover_does_not_touch_failed_jobs(repo):
    """Already-failed jobs should not be affected by recovery."""
    repo.create_job(
        job_id="already-fail", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.fail_job(job_id="already-fail", error_message="original error", duration_ms=10.0)

    recovered = repo.recover_stale_jobs()
    assert recovered == 0

    job = repo.get_job("already-fail")
    assert job["error_message"] == "original error"


def test_recover_multiple_stale_jobs(repo):
    """Multiple stale jobs should all be recovered."""
    for i in range(3):
        repo.create_job(
            job_id=f"stale-{i}", key_id="k", detectors=["unicode"],
            config_path=None, input_hash=f"h{i}", character_count=1,
        )
    repo.update_job_status(job_id="stale-0", status="running", started_at="2025-01-01T00:00:00+00:00")

    recovered = repo.recover_stale_jobs()
    assert recovered == 3

    for i in range(3):
        job = repo.get_job(f"stale-{i}")
        assert job["status"] == "failed"
        assert "interrupted by application restart" in job["error_message"]


def test_recover_does_not_touch_cancelled_jobs(repo):
    """Cancelled jobs should not be affected by recovery."""
    repo.create_job(
        job_id="cancelled-job", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.cancel_job_if_status("cancelled-job", expected_status="queued", new_status="cancelled")

    recovered = repo.recover_stale_jobs()
    assert recovered == 0

    job = repo.get_job("cancelled-job")
    assert job["status"] == "cancelled"


def test_startup_recovery_via_app(tmp_path):
    """App startup should recover stale jobs automatically."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"

    repo = SqliteRepository(tmp_path / "test.db")
    repo.create_job(
        job_id="pre-start", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="x", character_count=1,
    )
    repo.update_job_status(job_id="pre-start", status="running", started_at="2025-01-01T00:00:00+00:00")
    repo.close()

    app = create_app(db_url=db_url)
    with TestClient(app):
        pass

    repo2 = SqliteRepository(tmp_path / "test.db")
    job = repo2.get_job("pre-start")
    assert job["status"] == "failed"
    assert "interrupted by application restart" in job["error_message"]
    repo2.close()


# ---------------------------------------------------------------------------
# Cleanup includes cancelled
# ---------------------------------------------------------------------------


def test_cleanup_removes_cancelled_jobs(repo):
    """Cancelled jobs should be cleaned up like completed/failed."""
    from datetime import datetime, timedelta, timezone

    repo.create_job(
        job_id="old-cancel", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    repo.cancel_job_if_status("old-cancel", expected_status="queued", new_status="cancelled")

    old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    repo._conn.execute(
        "UPDATE jobs SET completed_at = ? WHERE job_id = 'old-cancel'",
        (old_time,),
    )
    repo._conn.commit()

    from provenance.api.jobs import cleanup_old_jobs
    cleaned = cleanup_old_jobs(repo)
    assert cleaned == 1
    assert repo.get_job("old-cancel") is None


# ---------------------------------------------------------------------------
# Persistence — new fields
# ---------------------------------------------------------------------------


def test_new_job_fields_in_list_response(client, repo):
    """Job listing should include retry_count and parent_job_id."""
    repo.create_job(
        job_id="list-fields", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=5,
        retry_count=2, parent_job_id="parent-123",
    )

    resp = client.get("/v1/jobs", headers=_auth())
    assert resp.status_code == 200
    jobs = resp.json()["jobs"]
    matching = [j for j in jobs if j["job_id"] == "list-fields"]
    assert len(matching) == 1
    assert matching[0]["retry_count"] == 2
    assert matching[0]["parent_job_id"] == "parent-123"


def test_new_job_fields_in_get_response(client, repo):
    """GET /v1/jobs/{id} should include retry_count and parent_job_id."""
    repo.create_job(
        job_id="get-fields", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=5,
        retry_count=1, parent_job_id="orig",
    )

    resp = client.get("/v1/jobs/get-fields", headers=_auth())
    assert resp.status_code == 200
    data = resp.json()
    assert data["retry_count"] == 1
    assert data["parent_job_id"] == "orig"


# ---------------------------------------------------------------------------
# Cancellation race: worker checks cancellation
# ---------------------------------------------------------------------------


def test_cancel_before_worker_picks_up(client, repo):
    """Cancel a queued job before the worker processes it — using repo directly."""
    # Create and cancel a queued job entirely via repo, bypassing the worker
    repo.create_job(
        job_id="cancel-no-worker", key_id=LEGACY_KEY_ID, detectors=["unicode"],
        config_path=None, input_hash="a", character_count=1,
    )
    assert repo.get_job("cancel-no-worker")["status"] == "queued"

    # Use the API's repo (same file) to cancel
    from provenance.api.state import get_repo
    r = get_repo()
    r.cancel_job_if_status("cancel-no-worker", expected_status="queued", new_status="cancelled")
    assert repo.get_job("cancel-no-worker")["status"] == "cancelled"
    assert repo.get_job("cancel-no-worker")["error_message"] == "Job cancelled by user"


# ---------------------------------------------------------------------------
# Mocked PostgreSQL methods
# ---------------------------------------------------------------------------


def test_pg_new_job_methods_mocked():
    """PostgresRepository job methods with retry_count/parent_job_id work (mocked)."""
    from unittest.mock import MagicMock, patch
    from provenance.api.db import PostgresRepository

    mock_psycopg2 = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (0,)
    mock_cursor.fetchall.return_value = []
    mock_cursor.__enter__ = lambda s: s
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value = mock_cursor
    mock_psycopg2.connect.return_value = mock_conn

    with patch.dict("sys.modules", {"psycopg2": mock_psycopg2}):
        repo = PostgresRepository("postgresql://test")

        repo.create_job(
            job_id="pg-j1", key_id="pg-k", detectors=["unicode"],
            config_path=None, input_hash="abc", character_count=10,
            retry_count=1, parent_job_id="pg-orig",
        )
        assert mock_cursor.execute.called

        mock_cursor.fetchone.return_value = (
            "pg-j1", "pg-k", "queued", "2025-01-01T00:00:00+00:00",
            None, None, "abc", 10, '["unicode"]', None, None, None, None, 1, "pg-orig",
        )
        mock_cursor.description = [
            ("job_id",), ("key_id",), ("status",), ("created_at",),
            ("started_at",), ("completed_at",), ("input_hash",), ("character_count",),
            ("detectors",), ("config_path",), ("result_json",), ("error_message",),
            ("duration_ms",), ("retry_count",), ("parent_job_id",),
        ]
        job = repo.get_job("pg-j1")
        assert job is not None
        assert job["retry_count"] == 1
        assert job["parent_job_id"] == "pg-orig"

        mock_cursor.fetchone.return_value = (0,)
        mock_cursor.fetchall.return_value = []
        repo.recover_stale_jobs()

        mock_cursor.rowcount = 1
        repo.cancel_job_if_status("pg-j1", expected_status="queued", new_status="cancelled")
        assert mock_cursor.execute.called

        repo.cleanup_jobs(older_than="2020-01-01T00:00:00+00:00")
        repo.close()
