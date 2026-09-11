"""Vercel adapter and serverless-runtime regressions.

These tests execute the same ASGI entry shape that Vercel detects at
``api/index.py``.  They intentionally avoid HF/Torch and external services.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.index import ApiPrefixAdapter
from provenance.api.app import create_app


def _vercel_client(tmp_path, monkeypatch, *, serverless: bool = False) -> TestClient:
    monkeypatch.setenv("PROVENANCE_ENVIRONMENT", "development")
    monkeypatch.setenv("PROVENANCE_ANON_COOKIE_SECRET", "vercel-test-cookie-secret")
    monkeypatch.setenv("PROVENANCE_PUBLIC_DAILY_LIMIT", "2")
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "10000")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    if serverless:
        monkeypatch.setenv("VERCEL", "1")
    else:
        monkeypatch.delenv("VERCEL", raising=False)
        # The durable executor is process-local.  A prior TestClient lifespan
        # may have completed its graceful shutdown in this same pytest
        # process; a new test app is not that shutting-down application.
        import provenance.api.jobs as jobs_mod
        jobs_mod._shutting_down = False
        jobs_mod._executor = None
    app = ApiPrefixAdapter(create_app(db_url=f"sqlite:///{tmp_path / 'vercel.db'}"))
    return TestClient(app)


def test_vercel_api_adapter_preserves_existing_routes_and_cookie_quota(tmp_path, monkeypatch):
    with _vercel_client(tmp_path, monkeypatch) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/ready").json()["status"] == "ready"

        normal = client.post("/api/v1/public/analyze", json={"text": "Visible text."})
        assert normal.status_code == 200
        assert normal.json()["signals_detected"] == []

        hidden = client.post("/api/v1/public/analyze", json={"text": "hello\u200bworld"})
        assert hidden.status_code == 200
        assert hidden.json()["signals_detected"] == ["unicode"]

        exhausted = client.post("/api/v1/public/analyze", json={"text": "third request"})
        assert exhausted.status_code == 429


def test_vercel_runtime_keeps_sync_unicode_but_gates_non_durable_work(tmp_path, monkeypatch):
    with _vercel_client(tmp_path, monkeypatch, serverless=True) as client:
        ready = client.get("/api/ready").json()
        assert ready["status"] == "ready"
        assert ready["executor"] == "not_applicable"
        assert client.get("/api/v1/public/config").json()["durable_worker_available"] is False

        unicode = client.post("/api/v1/analyze", json={"text": "visible", "detectors": ["unicode"]})
        assert unicode.status_code == 200

        configured = client.post("/api/v1/analyze", json={
            "text": "visible", "detectors": ["kgw"], "config_path": "/not-used-on-vercel.json",
        })
        assert configured.status_code == 503
        assert "durable worker" in configured.json()["detail"].lower()

        queued = client.post("/api/v1/analyze/async", json={"text": "visible"})
        assert queued.status_code == 503

        # Mutating a persisted job would otherwise promise cancellation/retry
        # semantics that no durable serverless worker can honor.
        assert client.delete("/api/v1/jobs/not-a-serverless-job").status_code == 503

        robustness = client.get("/api/v1/robustness/results")
        assert robustness.status_code == 503
