"""Tests for the Python SDK (provenance_client).

Uses mocked HTTP responses via a custom urllib handler.  No live server required.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Callable

import pytest

from provenance_client import (
    AuthenticationError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ProvenanceAPIError,
    ProvenanceClient,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from provenance_client._models import (
    AnalyzeResult,
    AsyncAnalyzeResult,
    CancelResult,
    JobResult,
    JobSummary,
    UsageResult,
)


# -----------------------------------------------------------------------
# Test transport (lightweight HTTP server)
# -----------------------------------------------------------------------

# Module-level mutable dict that the handler reads from.
# Tests set routes here before making requests.
_routes: dict[str, Callable] = {}


class _MockHandler(BaseHTTPRequestHandler):
    """Minimal handler that returns canned responses based on path/method."""

    def log_message(self, format, *args):  # noqa: A002
        pass  # Suppress request logging in tests

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")

    def _handle(self, method: str):
        path = self.path.split("?")[0]
        key = f"{method} {path}"
        handler = _routes.get(key)
        if handler is None:
            self._respond(404, {"detail": "Not found"})
            return
        body = None
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = {}
        status, data = handler(body, dict(self.headers))
        self._respond(status, data)

    def _respond(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Request-ID", "test-req-id")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(autouse=True)
def _clear_routes():
    """Ensure routes are cleared between tests."""
    _routes.clear()
    yield
    _routes.clear()


@pytest.fixture()
def server():
    """Start a mock HTTP server and return the base URL."""
    httpd = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = httpd.server_address[1]
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


@pytest.fixture()
def client(server) -> ProvenanceClient:
    return ProvenanceClient(base_url=server, api_key="test-key")


# -----------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------


def test_auth_header_sent(client):
    """X-API-Key header must be included in every request."""
    captured_headers = {}

    def _handler(body, headers):
        captured_headers.update(headers)
        return 200, {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_characters": 0, "by_endpoint": {}, "by_detector": {},
        }

    _routes["GET /v1/usage"] = _handler
    client.get_usage()
    # Check case-insensitively (HTTP message may lowercase keys)
    lower = {k.lower(): v for k, v in captured_headers.items()}
    assert lower.get("x-api-key") == "test-key"


def test_auth_error_401(client):
    """HTTP 401 should raise AuthenticationError."""
    _routes["GET /v1/usage"] = lambda body, headers: (401, {"detail": "Missing X-API-Key header"})
    with pytest.raises(AuthenticationError) as exc_info:
        client.get_usage()
    assert exc_info.value.status_code == 401
    assert "Missing" in exc_info.value.detail


def test_auth_error_403(client):
    """HTTP 403 should raise ForbiddenError."""
    _routes["GET /v1/usage"] = lambda body, headers: (403, {"detail": "Invalid API key"})
    with pytest.raises(ForbiddenError) as exc_info:
        client.get_usage()
    assert exc_info.value.status_code == 403


def test_no_api_key_no_header(client):
    """When api_key is empty, X-API-Key header should not be sent."""
    client._api_key = ""
    captured_headers = {}

    def _handler(body, headers):
        captured_headers.update(headers)
        return 200, {
            "total_requests": 0, "successful_requests": 0, "failed_requests": 0,
            "total_characters": 0, "by_endpoint": {}, "by_detector": {},
        }

    _routes["GET /v1/usage"] = _handler
    client.get_usage()
    lower = {k.lower(): v for k, v in captured_headers.items()}
    assert "x-api-key" not in lower


# -----------------------------------------------------------------------
# analyze()
# -----------------------------------------------------------------------


def test_analyze_basic(client):
    _routes["POST /v1/analyze"] = lambda body, headers: (200, {
        "analysis_id": "abc-123",
        "engine_version": "0.1.0",
        "status": "ok",
        "text_stats": {"character_count": 13},
        "results": [{"detector": "unicode", "detector_version": "0.1.0", "status": "ok",
                     "detected": False, "confidence": "none", "evidence": {},
                     "limitations": [], "metadata": {},
                     "implementation_kind": "deterministic", "compatibility": "ok",
                     "text_requirements": {}}],
        "limitations": [],
        "metadata": {},
        "duration_ms": 12.5,
    })
    result = client.analyze(text="Hello, world!")
    assert isinstance(result, AnalyzeResult)
    assert result.analysis_id == "abc-123"
    assert result.status == "ok"
    assert result.duration_ms == 12.5
    assert len(result.results) == 1
    assert result.results[0].detector == "unicode"


def test_analyze_with_detectors(client):
    _routes["POST /v1/analyze"] = lambda body, headers: (200, {
        "analysis_id": "def-456",
        "engine_version": "0.1.0",
        "status": "ok",
        "text_stats": {"character_count": 4},
        "results": [],
        "limitations": [],
        "metadata": {},
    })
    result = client.analyze(text="Test", detectors=["unicode", "kgw"], config_path="/tmp/c.json")
    assert result.analysis_id == "def-456"


def test_analyze_validation_error(client):
    _routes["POST /v1/analyze"] = lambda body, headers: (422, {"detail": "Unknown detector(s): foo"})
    with pytest.raises(ValidationError) as exc_info:
        client.analyze(text="Test", detectors=["foo"])
    assert exc_info.value.status_code == 422
    assert "Unknown detector" in exc_info.value.detail


# -----------------------------------------------------------------------
# analyze_async()
# -----------------------------------------------------------------------


def test_analyze_async(client):
    _routes["POST /v1/analyze/async"] = lambda body, headers: (202, {
        "job_id": "job-789",
        "status": "queued",
        "message": "Analysis job submitted",
    })
    result = client.analyze_async(text="Async test")
    assert isinstance(result, AsyncAnalyzeResult)
    assert result.job_id == "job-789"
    assert result.status == "queued"


# -----------------------------------------------------------------------
# get_job()
# -----------------------------------------------------------------------


def test_get_job_completed(client):
    _routes["GET /v1/jobs/job-789"] = lambda body, headers: (200, {
        "job_id": "job-789",
        "status": "completed",
        "created_at": "2025-01-01T00:00:00+00:00",
        "started_at": "2025-01-01T00:00:01+00:00",
        "completed_at": "2025-01-01T00:00:02+00:00",
        "character_count": 10,
        "detectors": ["unicode"],
        "result": {"status": "ok", "results": []},
        "retry_count": 0,
        "parent_job_id": None,
    })
    job = client.get_job("job-789")
    assert isinstance(job, JobResult)
    assert job.status == "completed"
    assert job.is_completed
    assert job.is_terminal
    assert job.result is not None


def test_get_job_failed(client):
    _routes["GET /v1/jobs/job-fail"] = lambda body, headers: (200, {
        "job_id": "job-fail",
        "status": "failed",
        "created_at": "2025-01-01T00:00:00+00:00",
        "error_message": "ValueError: bad input",
        "retry_count": 0,
    })
    job = client.get_job("job-fail")
    assert job.is_failed
    assert job.error_message == "ValueError: bad input"


def test_get_job_not_found(client):
    _routes["GET /v1/jobs/missing"] = lambda body, headers: (404, {"detail": "Job missing not found"})
    with pytest.raises(NotFoundError):
        client.get_job("missing")


# -----------------------------------------------------------------------
# list_jobs()
# -----------------------------------------------------------------------


def test_list_jobs(client):
    _routes["GET /v1/jobs"] = lambda body, headers: (200, {
        "jobs": [
            {"job_id": "j1", "status": "completed", "created_at": "2025-01-01T00:00:00+00:00",
             "character_count": 5, "detectors": ["unicode"], "retry_count": 0},
            {"job_id": "j2", "status": "queued", "created_at": "2025-01-01T00:01:00+00:00",
             "character_count": 3, "detectors": ["unicode"], "retry_count": 0},
        ],
        "total": 2,
        "limit": 20,
        "offset": 0,
    })
    jobs, total = client.list_jobs()
    assert total == 2
    assert len(jobs) == 2
    assert isinstance(jobs[0], JobSummary)
    assert jobs[0].job_id == "j1"


# -----------------------------------------------------------------------
# cancel_job()
# -----------------------------------------------------------------------


def test_cancel_queued_job(client):
    _routes["DELETE /v1/jobs/job-q"] = lambda body, headers: (200, {
        "job_id": "job-q",
        "status": "cancelled",
        "message": "Queued job cancelled",
    })
    result = client.cancel_job("job-q")
    assert isinstance(result, CancelResult)
    assert result.status == "cancelled"


def test_cancel_conflict(client):
    _routes["DELETE /v1/jobs/job-done"] = lambda body, headers: (409, {
        "detail": "Cannot cancel job in 'completed' status",
    })
    with pytest.raises(ConflictError) as exc_info:
        client.cancel_job("job-done")
    assert exc_info.value.status_code == 409


# -----------------------------------------------------------------------
# retry_job()
# -----------------------------------------------------------------------


def test_retry_job(client):
    _routes["POST /v1/jobs/job-fail/retry"] = lambda body, headers: (202, {
        "job_id": "job-retry-1",
        "status": "queued",
        "message": "Retry submitted (attempt 1)",
    })
    result = client.retry_job("job-fail", text="Re-analyze this")
    assert isinstance(result, AsyncAnalyzeResult)
    assert result.job_id == "job-retry-1"


def test_retry_non_retryable(client):
    _routes["POST /v1/jobs/job-ok/retry"] = lambda body, headers: (409, {
        "detail": "Only failed jobs can be retried (current status: completed)",
    })
    with pytest.raises(ConflictError):
        client.retry_job("job-ok", text="text")


# -----------------------------------------------------------------------
# get_usage()
# -----------------------------------------------------------------------


def test_get_usage(client):
    _routes["GET /v1/usage"] = lambda body, headers: (200, {
        "total_requests": 42,
        "successful_requests": 40,
        "failed_requests": 2,
        "total_characters": 12345,
        "by_endpoint": {"/v1/analyze": 40},
        "by_detector": {"unicode": 38},
    })
    usage = client.get_usage()
    assert isinstance(usage, UsageResult)
    assert usage.total_requests == 42
    assert usage.successful_requests == 40
    assert usage.failed_requests == 2
    assert usage.total_characters == 12345
    assert usage.by_endpoint["/v1/analyze"] == 40
    assert usage.by_detector["unicode"] == 38


# -----------------------------------------------------------------------
# wait_for_job()
# -----------------------------------------------------------------------


def test_wait_for_job_polls_until_completed(client):
    """wait_for_job should poll and return when job completes."""
    call_count = 0

    def _handler(body, headers):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return 200, {
                "job_id": "job-poll",
                "status": "running",
                "created_at": "2025-01-01T00:00:00+00:00",
                "character_count": 5,
                "detectors": ["unicode"],
            }
        return 200, {
            "job_id": "job-poll",
            "status": "completed",
            "created_at": "2025-01-01T00:00:00+00:00",
            "completed_at": "2025-01-01T00:00:02+00:00",
            "character_count": 5,
            "detectors": ["unicode"],
            "result": {"status": "ok"},
        }

    _routes["GET /v1/jobs/job-poll"] = _handler
    job = client.wait_for_job("job-poll", poll_interval=0.1, timeout=5.0)
    assert job.status == "completed"
    assert call_count == 3


def test_wait_for_job_timeout(client):
    """wait_for_job should raise TimeoutError if job never completes."""
    _routes["GET /v1/jobs/job-stuck"] = lambda body, headers: (200, {
        "job_id": "job-stuck",
        "status": "running",
        "created_at": "2025-01-01T00:00:00+00:00",
        "character_count": 5,
        "detectors": ["unicode"],
    })
    with pytest.raises(TimeoutError):
        client.wait_for_job("job-stuck", poll_interval=0.1, timeout=0.5)


def test_wait_for_job_immediately_terminal(client):
    """wait_for_job should return immediately if job is already completed."""
    _routes["GET /v1/jobs/job-done"] = lambda body, headers: (200, {
        "job_id": "job-done",
        "status": "completed",
        "created_at": "2025-01-01T00:00:00+00:00",
        "completed_at": "2025-01-01T00:00:01+00:00",
        "character_count": 5,
        "detectors": ["unicode"],
        "result": {"status": "ok"},
    })
    job = client.wait_for_job("job-done")
    assert job.status == "completed"


def test_wait_for_job_failed(client):
    """wait_for_job should return (not raise) when job fails."""
    _routes["GET /v1/jobs/job-f"] = lambda body, headers: (200, {
        "job_id": "job-f",
        "status": "failed",
        "created_at": "2025-01-01T00:00:00+00:00",
        "error_message": "oops",
    })
    job = client.wait_for_job("job-f")
    assert job.is_failed


# -----------------------------------------------------------------------
# HTTP error mapping
# -----------------------------------------------------------------------


def test_rate_limit_error(client):
    _routes["GET /v1/usage"] = lambda body, headers: (429, {"detail": "Rate limit exceeded"})
    with pytest.raises(RateLimitError) as exc_info:
        client.get_usage()
    assert exc_info.value.status_code == 429


def test_unknown_status_raises_base_error(client):
    _routes["GET /v1/usage"] = lambda body, headers: (502, {"detail": "Bad gateway"})
    with pytest.raises(ProvenanceAPIError) as exc_info:
        client.get_usage()
    assert exc_info.value.status_code == 502


# -----------------------------------------------------------------------
# API key non-leakage
# -----------------------------------------------------------------------


def test_api_key_not_in_exception_message(client):
    """API key must never appear in exception messages or str representation."""
    client._api_key = "super-secret-key-abc123"
    _routes["GET /v1/usage"] = lambda body, headers: (401, {"detail": "Unauthorized"})
    with pytest.raises(AuthenticationError) as exc_info:
        client.get_usage()
    assert "super-secret-key-abc123" not in str(exc_info.value)


def test_api_key_not_in_models():
    """API key should not be stored in model objects."""
    result = AnalyzeResult(
        analysis_id="test",
        engine_version="0.1.0",
        status="ok",
    )
    assert "api_key" not in str(result)
    assert "secret" not in str(result)


# -----------------------------------------------------------------------
# Connection error handling
# -----------------------------------------------------------------------


def test_connection_error():
    """Connection to a non-existent server should raise ProvenanceAPIError."""
    client = ProvenanceClient(base_url="http://127.0.0.1:1", api_key="key", timeout=1.0)
    with pytest.raises(ProvenanceAPIError) as exc_info:
        client.get_usage()
    assert exc_info.value.status_code == 0


# -----------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------


def test_analyze_result_from_dict():
    d = {
        "analysis_id": "x",
        "engine_version": "0.1.0",
        "status": "ok",
        "text_stats": {"character_count": 5},
        "results": [
            {"detector": "unicode", "detector_version": "0.1.0", "status": "ok",
             "detected": False, "confidence": "none", "evidence": {},
             "limitations": [], "metadata": {},
             "implementation_kind": "d", "compatibility": "c",
             "text_requirements": {}}
        ],
    }
    r = AnalyzeResult.from_dict(d)
    assert r.analysis_id == "x"
    assert len(r.results) == 1
    assert r.results[0].detector == "unicode"


def test_job_result_properties():
    completed = JobResult(job_id="j", status="completed", created_at="t")
    assert completed.is_terminal
    assert completed.is_completed
    assert not completed.is_failed

    failed = JobResult(job_id="j", status="failed", created_at="t")
    assert failed.is_terminal
    assert failed.is_failed

    running = JobResult(job_id="j", status="running", created_at="t")
    assert not running.is_terminal
    assert not running.is_completed
    assert not running.is_failed


def test_cancel_result_from_dict():
    r = CancelResult.from_dict({"job_id": "j", "status": "cancelled", "message": "done"})
    assert r.job_id == "j"
    assert r.status == "cancelled"
    assert r.message == "done"


# -----------------------------------------------------------------------
# Detector discovery (Phase 5A: GET /v1/detectors)
# -----------------------------------------------------------------------


def test_list_detectors(client):
    _routes["GET /v1/detectors"] = lambda body, headers: (200, {"detectors": [
        {"name": "unicode", "display_name": "Unicode Artifact Detection",
         "implementation_kind": "unicode", "compatibility": "any",
         "requires_config": False, "supports_generation": False,
         "supports_benchmarking": True, "tokenizer_requirements": None,
         "known_limitations": [], "description": "d"},
        {"name": "kgw", "display_name": "KGW Watermark Detection",
         "implementation_kind": "watermark-kgw", "compatibility": "gpt-2",
         "requires_config": True, "supports_generation": True,
         "supports_benchmarking": True, "tokenizer_requirements": "huggingface",
         "known_limitations": [], "description": "d"},
    ]})
    dets = client.list_detectors()
    assert len(dets) == 2
    assert dets[0]["name"] == "unicode"
    assert dets[1]["requires_config"] is True


def test_list_detectors_empty(client):
    _routes["GET /v1/detectors"] = lambda body, headers: (200, {"detectors": []})
    assert client.list_detectors() == []


def test_list_detectors_auth_error(client):
    _routes["GET /v1/detectors"] = lambda body, headers: (401, {"detail": "Missing X-API-Key header"})
    with pytest.raises(AuthenticationError):
        client.list_detectors()


# -----------------------------------------------------------------------
# Robustness benchmark artifacts (Phase 6B)
# -----------------------------------------------------------------------


def test_robustness_query_builder():
    assert ProvenanceClient._robustness_query() == ""
    assert ProvenanceClient._robustness_query(detector="kgw") == "?detector=kgw"
    q = ProvenanceClient._robustness_query(
        detector="kgw", config="a b", transform="lowercase", text_length=50)
    assert "detector=kgw" in q
    assert "config=a+b" in q or "config=a%20b" in q
    assert "transform=lowercase" in q
    assert "text_length=50" in q


def test_get_robustness_results(client):
    payload = {
        "schema_version": "provenance-robustness-v1",
        "total_results": 2,
        "detectors": ["kgw"],
        "transforms": ["identity"],
        "results": [],
        "aggregated": [],
        "matrix": None,
        "limitations": [],
        "summary": {"total_files_scanned": 2},
        "warnings": [],
    }
    _routes["GET /v1/robustness/results"] = lambda body, headers: (200, payload)
    data = client.get_robustness_results(detector="kgw", text_length=50)
    assert data["total_results"] == 2
    assert data["summary"]["total_files_scanned"] == 2


def test_get_robustness_comparison(client):
    payload = {
        "schema_version": "provenance-benchmark-report-v1",
        "run_id": "api",
        "benchmark_name": "robustness-comparison",
        "rows": [],
        "by_model": [{"model_config": "a", "robustness_rate": 0.8}],
        "by_transform": [],
        "by_category": [],
        "by_length": [],
        "limitations": [],
        "warnings": [],
    }
    _routes["GET /v1/robustness/comparison"] = lambda body, headers: (200, payload)
    data = client.get_robustness_comparison()
    assert data["by_model"][0]["model_config"] == "a"


def test_robustness_auth_error(client):
    _routes["GET /v1/robustness/results"] = lambda body, headers: (401, {"detail": "Missing X-API-Key header"})
    with pytest.raises(AuthenticationError):
        client.get_robustness_results()
    _routes["GET /v1/robustness/comparison"] = lambda body, headers: (403, {"detail": "denied"})
    with pytest.raises(ForbiddenError):
        client.get_robustness_comparison()


def test_robustness_connection_error(server):
    bad = ProvenanceClient(base_url="http://127.0.0.1:1", api_key="k", timeout=1.0)
    with pytest.raises(ProvenanceAPIError):
        bad.get_robustness_results()
