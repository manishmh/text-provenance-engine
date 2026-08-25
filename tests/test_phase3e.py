"""Phase 3E tests: Metrics, readiness, queue observability, config validation,
graceful shutdown, and privacy protection.

Uses a temporary SQLite database for each test.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app
from provenance.api.db import SqliteRepository, _text_hash

API_KEY = "test-phase3e-key"
ADMIN_KEY = "test-phase3e-admin"


@pytest.fixture(autouse=True)
def _set_keys(monkeypatch):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    monkeypatch.setenv("PROVENANCE_ADMIN_API_KEY", ADMIN_KEY)
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


@pytest.fixture()
def repo(tmp_path):
    r = SqliteRepository(tmp_path / "test.db")
    yield r
    r.close()


def _auth(key: str = API_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


# ---------------------------------------------------------------------------
# Metrics Endpoint — GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_returns_200(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "engine_version" in data
    assert "total_requests" in data
    assert "successful_requests" in data
    assert "failed_requests" in data
    assert "total_characters" in data
    assert "by_endpoint" in data
    assert "by_detector" in data
    assert "by_status_code" in data


def test_metrics_no_auth_required(client):
    """Metrics endpoint is public (no auth needed)."""
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_metrics_job_counts(client):
    resp = client.get("/metrics")
    data = resp.json()
    assert "jobs_queued" in data
    assert "jobs_running" in data
    assert "jobs_completed" in data
    assert "jobs_failed" in data
    assert "jobs_cancelled" in data
    assert "jobs_cancellation_requested" in data
    assert isinstance(data["jobs_queued"], int)
    assert isinstance(data["jobs_running"], int)


def test_metrics_worker_count(client):
    resp = client.get("/metrics")
    data = resp.json()
    assert "configured_worker_count" in data
    assert isinstance(data["configured_worker_count"], int)
    assert data["configured_worker_count"] >= 1


def test_metrics_reflects_analyze(client):
    """After an analyze call, metrics should reflect the request."""
    client.post("/v1/analyze", json={"text": "metrics test"}, headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    assert data["total_requests"] >= 1
    assert data["successful_requests"] >= 1
    assert data["total_characters"] >= 1


def test_metrics_reflects_usage_by_endpoint(client):
    """Metrics should break down by endpoint."""
    client.post("/v1/analyze", json={"text": "ep test"}, headers=_auth())
    client.get("/v1/analyses", headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    assert "/v1/analyze" in data["by_endpoint"]
    assert "/v1/analyses" in data["by_endpoint"]


def test_metrics_reflects_usage_by_detector(client):
    """Metrics should break down by detector."""
    client.post("/v1/analyze", json={"text": "det test", "detectors": ["unicode"]}, headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    assert "unicode" in data["by_detector"]


def test_metrics_reflects_usage_by_status_code(client):
    """Metrics should break down by HTTP status code."""
    client.post("/v1/analyze", json={"text": "sc test"}, headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    assert "200" in data["by_status_code"]


def test_metrics_avg_duration(client):
    """Average duration should be a float when requests exist."""
    client.post("/v1/analyze", json={"text": "dur test"}, headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    assert data["avg_duration_ms"] is not None
    assert isinstance(data["avg_duration_ms"], float)
    assert data["avg_duration_ms"] >= 0


def test_metrics_empty_state(client):
    """Fresh database should return zero counts."""
    resp = client.get("/metrics")
    data = resp.json()
    assert data["total_requests"] == 0
    assert data["total_characters"] == 0
    assert data["avg_duration_ms"] is None
    assert data["by_endpoint"] == {}
    assert data["by_detector"] == {}


# ---------------------------------------------------------------------------
# Job Queue Metrics in /metrics
# ---------------------------------------------------------------------------


def test_metrics_job_counts_after_async(client):
    """Job counts should update after async job submission."""
    resp = client.post("/v1/analyze/async", json={"text": "qm test"}, headers=_auth())
    assert resp.status_code == 202
    # Wait for completion
    time.sleep(2)
    resp2 = client.get("/metrics")
    data = resp2.json()
    assert data["jobs_completed"] >= 1


# ---------------------------------------------------------------------------
# Readiness Endpoint — GET /ready
# ---------------------------------------------------------------------------


def test_ready_returns_200(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["engine_version"]
    assert data["persistence"] == "ok"
    assert data["executor"] == "ok"


def test_ready_no_auth_required(client):
    resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_reports_persistence(client):
    resp = client.get("/ready")
    data = resp.json()
    assert "persistence" in data
    assert data["persistence"] in ("ok", "error")


def test_ready_reports_executor(client):
    resp = client.get("/ready")
    data = resp.json()
    assert "executor" in data
    assert data["executor"] in ("ok", "shutting_down", "error")


# ---------------------------------------------------------------------------
# Graceful Shutdown
# ---------------------------------------------------------------------------


def test_shutdown_sets_flag():
    """shutdown_executor should set the shutting_down flag."""
    import provenance.api.jobs as jobs_mod
    # Reset state
    jobs_mod._executor = None
    jobs_mod._shutting_down = False

    executor = jobs_mod.get_executor()
    assert not jobs_mod.is_shutting_down()

    jobs_mod.shutdown_executor(wait=True)
    assert jobs_mod.is_shutting_down()

    # Cleanup
    jobs_mod._executor = None
    jobs_mod._shutting_down = False


def test_executor_none_after_shutdown():
    """Executor should be None after shutdown."""
    import provenance.api.jobs as jobs_mod
    jobs_mod._executor = None
    jobs_mod._shutting_down = False

    executor = jobs_mod.get_executor()
    assert executor is not None

    jobs_mod.shutdown_executor(wait=True)
    assert jobs_mod._executor is None

    # Cleanup
    jobs_mod._shutting_down = False


def test_ready_during_shutdown(tmp_path, monkeypatch):
    """Ready endpoint should report executor as shutting_down."""
    import provenance.api.jobs as jobs_mod
    jobs_mod._shutting_down = False
    jobs_mod._executor = None

    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")

    # Simulate shutdown
    jobs_mod._shutting_down = True

    with TestClient(app) as c:
        resp = c.get("/ready")
        data = resp.json()
        assert data["executor"] == "shutting_down"
        assert data["status"] == "not_ready"

    # Cleanup
    jobs_mod._shutting_down = False
    jobs_mod._executor = None


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------


def test_validate_config_valid(monkeypatch):
    """Valid config should not raise."""
    monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "2")
    monkeypatch.setenv("PROVENANCE_JOB_RETENTION_HOURS", "24")
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "60")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    monkeypatch.setenv("MAX_LIST_LIMIT", "100")

    from provenance.api.config import validate_config
    validate_config()  # Should not raise


def test_validate_config_invalid_max_workers(monkeypatch):
    """Invalid PROVENANCE_MAX_BACKGROUND_JOBS should raise."""
    monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "0")
    from provenance.api.config import validate_config, ConfigError
    with pytest.raises(ConfigError, match="PROVENANCE_MAX_BACKGROUND_JOBS"):
        validate_config()


def test_validate_config_negative_retention(monkeypatch):
    """Negative PROVENANCE_JOB_RETENTION_HOURS should raise."""
    monkeypatch.setenv("PROVENANCE_JOB_RETENTION_HOURS", "-1")
    from provenance.api.config import validate_config, ConfigError
    with pytest.raises(ConfigError, match="PROVENANCE_JOB_RETENTION_HOURS"):
        validate_config()


def test_validate_config_non_numeric(monkeypatch):
    """Non-numeric PROVENANCE_MAX_BACKGROUND_JOBS should raise."""
    monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "abc")
    from provenance.api.config import validate_config, ConfigError
    with pytest.raises(ConfigError, match="must be a positive integer"):
        validate_config()


def test_validate_config_optional_not_set(monkeypatch):
    """Optional limits (not set) should not raise."""
    monkeypatch.delenv("PROVENANCE_DAILY_REQUEST_LIMIT", raising=False)
    monkeypatch.delenv("PROVENANCE_DAILY_CHARACTER_LIMIT", raising=False)
    from provenance.api.config import validate_config
    validate_config()  # Should not raise


def test_validate_config_optional_valid(monkeypatch):
    """Valid optional limits should not raise."""
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "1000")
    monkeypatch.setenv("PROVENANCE_DAILY_CHARACTER_LIMIT", "500000")
    from provenance.api.config import validate_config
    validate_config()  # Should not raise


def test_validate_config_optional_invalid(monkeypatch):
    """Invalid optional limits should raise."""
    monkeypatch.setenv("PROVENANCE_DAILY_REQUEST_LIMIT", "0")
    from provenance.api.config import validate_config, ConfigError
    with pytest.raises(ConfigError, match="PROVENANCE_DAILY_REQUEST_LIMIT"):
        validate_config()


def test_app_logs_config_warning(tmp_path, monkeypatch, caplog):
    """Invalid config should log a warning, not crash the app."""
    import logging
    monkeypatch.setenv("PROVENANCE_MAX_BACKGROUND_JOBS", "0")

    with caplog.at_level(logging.WARNING, logger="provenance.api"):
        app = create_app(db_url=f"sqlite:///{tmp_path / 'test.db'}")
        with TestClient(app):
            pass  # App starts despite invalid config

    assert any("Configuration validation failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Repository: Job Counts
# ---------------------------------------------------------------------------


def test_get_job_counts_empty(repo):
    """Empty database should return all zeros."""
    counts = repo.get_job_counts()
    assert counts["queued"] == 0
    assert counts["running"] == 0
    assert counts["completed"] == 0
    assert counts["failed"] == 0
    assert counts["cancelled"] == 0
    assert counts["cancellation_requested"] == 0


def test_get_job_counts_mixed(repo):
    """Counts should reflect actual job statuses."""
    for i in range(3):
        repo.create_job(
            job_id=f"q{i}", key_id="k", detectors=["unicode"],
            config_path=None, input_hash=f"h{i}", character_count=1,
        )
    repo.create_job(
        job_id="r1", key_id="k", detectors=["unicode"],
        config_path=None, input_hash="hr", character_count=1,
    )
    repo.update_job_status(job_id="r1", status="running", started_at="2025-01-01T00:00:00+00:00")

    repo.complete_job(job_id="q0", result_json="{}", duration_ms=5.0)
    repo.fail_job(job_id="q1", error_message="err", duration_ms=5.0)

    counts = repo.get_job_counts()
    assert counts["queued"] == 1  # q2
    assert counts["running"] == 1  # r1
    assert counts["completed"] == 1  # q0
    assert counts["failed"] == 1  # q1


# ---------------------------------------------------------------------------
# Repository: Average Duration
# ---------------------------------------------------------------------------


def test_get_avg_duration_empty(repo):
    """Empty usage records should return None."""
    assert repo.get_avg_duration() is None


def test_get_avg_duration_with_records(repo):
    """Average duration should be computed correctly."""
    repo.record_usage(
        key_id="k", endpoint="/v1/analyze", status_code=200,
        duration_ms=100.0, character_count=10, success=True,
    )
    repo.record_usage(
        key_id="k", endpoint="/v1/analyze", status_code=200,
        duration_ms=200.0, character_count=10, success=True,
    )
    avg = repo.get_avg_duration()
    assert avg == 150.0


# ---------------------------------------------------------------------------
# Repository: Usage By Status
# ---------------------------------------------------------------------------


def test_get_usage_by_status_empty(repo):
    """Empty usage records should return empty dict."""
    assert repo.get_usage_by_status() == {}


def test_get_usage_by_status_mixed(repo):
    """Status breakdown should reflect actual status codes."""
    repo.record_usage(
        key_id="k", endpoint="/v1/analyze", status_code=200,
        duration_ms=10.0, character_count=5, success=True,
    )
    repo.record_usage(
        key_id="k", endpoint="/v1/analyze", status_code=200,
        duration_ms=10.0, character_count=5, success=True,
    )
    repo.record_usage(
        key_id="k", endpoint="/v1/analyze", status_code=404,
        duration_ms=10.0, character_count=0, success=False,
    )
    by_status = repo.get_usage_by_status()
    assert by_status["200"] == 2
    assert by_status["404"] == 1


# ---------------------------------------------------------------------------
# Privacy: /metrics does not leak secrets
# ---------------------------------------------------------------------------


def test_metrics_no_api_keys_leaked(client):
    """Metrics response must not contain any API keys."""
    resp = client.get("/metrics")
    body = json.dumps(resp.json())
    assert API_KEY not in body
    assert ADMIN_KEY not in body


def test_metrics_no_text_hashes(client):
    """Metrics should not expose individual text hashes."""
    client.post("/v1/analyze", json={"text": "secret text"}, headers=_auth())
    resp = client.get("/metrics")
    data = resp.json()
    # Ensure no hash-like values are in top-level fields (64-char hex)
    for key, value in data.items():
        if isinstance(value, str):
            assert len(value) != 64 or not all(c in "0123456789abcdef" for c in value), \
                f"Potential hash leaked in {key}"


def test_metrics_no_job_ids_exposed(client):
    """Metrics should not expose individual job IDs."""
    client.post("/v1/analyze/async", json={"text": "jid test"}, headers=_auth())
    time.sleep(1)
    resp = client.get("/metrics")
    body = json.dumps(resp.json())
    # No UUID-like strings in metrics
    import re
    uuids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', body)
    assert len(uuids) == 0, f"UUIDs found in metrics: {uuids}"


def test_ready_no_secrets(client):
    """Ready response must not contain secrets."""
    resp = client.get("/ready")
    body = json.dumps(resp.json())
    assert API_KEY not in body
    assert ADMIN_KEY not in body


def test_health_still_lightweight(client):
    """Health endpoint should remain extremely lightweight."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "engine_version" in data
    # Should NOT have job counts or usage data
    assert "total_requests" not in data
    assert "jobs_queued" not in data


# ---------------------------------------------------------------------------
# Mocked PostgreSQL methods for new repo methods
# ---------------------------------------------------------------------------


def test_pg_job_counts_mocked():
    """PostgresRepository.get_job_counts works (mocked)."""
    from tests.test_api import _make_mock_pg
    from provenance.api.db import PostgresRepository

    mock_pg, mock_conn, mock_cursor = _make_mock_pg()
    with patch.dict("sys.modules", {"psycopg2": mock_pg}):
        repo = PostgresRepository("postgresql://test")

        # Mock get_job_counts
        mock_cursor.fetchall.return_value = [("queued", 2), ("running", 1), ("completed", 5)]
        counts = repo.get_job_counts()
        assert counts["queued"] == 2
        assert counts["running"] == 1
        assert counts["completed"] == 5
        assert counts["failed"] == 0

        # Mock get_avg_duration
        mock_cursor.fetchone.return_value = (150.5,)
        avg = repo.get_avg_duration()
        assert avg == 150.5

        # Mock get_usage_by_status
        mock_cursor.fetchall.return_value = [("200", 10), ("404", 2)]
        by_status = repo.get_usage_by_status()
        assert by_status["200"] == 10
        assert by_status["404"] == 2

        repo.close()


# ---------------------------------------------------------------------------
# Usage aggregation respects ownership
# ---------------------------------------------------------------------------


def test_usage_ownership_preserved(client):
    """GET /v1/usage should only return stats for the calling key."""
    # Key 1 makes a request
    client.post("/v1/analyze", json={"text": "key1 text"}, headers=_auth())
    time.sleep(1)

    # Create key 2 and make a request
    resp = client.post("/v1/api-keys", json={"name": "Key2"}, headers=_admin_auth())
    key2 = resp.json()["key"]
    client.post("/v1/analyze", json={"text": "key2 text"}, headers=_auth(key2))
    time.sleep(1)

    # Key 1's usage should only show 1 request
    resp1 = client.get("/v1/usage", headers=_auth())
    assert resp1.json()["total_requests"] == 1

    # Key 2's usage should only show 1 request
    resp2 = client.get("/v1/usage", headers=_auth(key2))
    assert resp2.json()["total_requests"] == 1


def _admin_auth() -> dict[str, str]:
    return {"X-API-Key": ADMIN_KEY}
