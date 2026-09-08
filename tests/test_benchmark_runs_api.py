"""Tests for Phase 6C: dashboard-triggered benchmark runs.

Lifecycle tests use an injected fake experiment function — no Hugging
Face downloads. The fake writes real Phase 5 result files so artifact
association is exercised end to end.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app

API_KEY = "test-phase6c-key"
OTHER_KEY = "test-phase6c-other"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)
    monkeypatch.setenv("PROVENANCE_ROBUSTNESS_DIR", str(tmp_path / "robustness"))
    # Polling loops make many requests: lift rate limits for this module and
    # reset the process-global limiter so counts start fresh.
    monkeypatch.setenv("RATE_LIMIT_MAX_REQUESTS", "100000")
    import provenance.api.middleware as middleware_mod
    monkeypatch.setattr(middleware_mod, "_limiter", None)
    import provenance.api.benchmarks as bench_mod
    monkeypatch.setattr(bench_mod, "EXPERIMENT_FN", None)
    monkeypatch.setattr(bench_mod, "_semaphore", None)


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_headers():
    return {"X-API-Key": API_KEY}


@pytest.fixture()
def hf_config(tmp_path):
    """Structurally valid (but unloadable) HF experiment config."""
    p = tmp_path / "model.json"
    p.write_text(json.dumps({
        "model": {"model_identifier": "fake-model"},
        "tokenizer": {"type": "simple", "vocabulary": ["a", "b"]},
    }), encoding="utf-8")
    return str(p)


def _ok_experiment_fn(spec, exp_out_dir):
    from provenance.robustness.benchmark import (
        build_benchmark_result,
        write_benchmark_results,
    )
    recs = [{
        "original_detected": True, "transformed_detected": True,
        "original_score": 1.0, "transformed_score": 1.0,
        "score_delta": 0.0, "detection_changed": False,
    }]
    result = build_benchmark_result(
        detector_name=spec.detector,
        config_identifier=spec.config,
        transform_name="identity",
        text_length=spec.lengths[0] if spec.lengths else None,
        records=recs, seed=spec.seed,
    )
    write_benchmark_results([result], Path(exp_out_dir) / "benchmark_results.jsonl")
    return [result.to_dict()]


def _use_fake(monkeypatch=None, fn=None):
    import provenance.api.benchmarks as bench_mod
    bench_mod.EXPERIMENT_FN = fn or _ok_experiment_fn


def _valid_payload(hf_config, **overrides):
    payload = {
        "detector": "kgw-reference",
        "config": hf_config,
        "profile": "all_safe",
        "lengths": [50],
        "samples": 2,
        "seed": 42,
    }
    payload.update(overrides)
    return payload


def _wait_for(client, headers, run_id, statuses=("completed", "failed", "cancelled"), timeout=15.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/v1/benchmark-runs/{run_id}", headers=headers)
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in statuses:
            return last
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach {statuses} in time (last={last})")


# ---------------------------------------------------------------------------
# Options schema
# ---------------------------------------------------------------------------

class TestOptions:
    def test_options_shape(self, client, auth_headers):
        data = client.get("/v1/benchmark-runs/options", headers=auth_headers).json()
        names = [d["name"] for d in data["detectors"]]
        # Only reference detectors can complete generation runs
        assert sorted(names) == ["kgw-reference", "synthid-reference"]
        assert len(data["profiles"]) > 0
        assert len(data["transforms"]) > 0
        assert data["constraints"]["max_text_length"] == 500
        assert data["constraints"]["max_samples"] == 50

    def test_options_requires_auth(self, client):
        assert client.get("/v1/benchmark-runs/options").status_code == 401


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _create(self, client, auth_headers, payload):
        return client.post("/v1/benchmark-runs", headers=auth_headers, json=payload)

    def test_unknown_detector(self, client, auth_headers, hf_config):
        resp = self._create(client, auth_headers, _valid_payload(hf_config, detector="nope"))
        assert resp.status_code == 422
        assert any(e["field"] == "detector" for e in resp.json()["errors"])

    def test_unicode_rejected_with_guidance(self, client, auth_headers):
        payload = {"detector": "unicode", "lengths": [50], "samples": 1, "seed": 1}
        resp = self._create(client, auth_headers, payload)
        assert resp.status_code == 422
        assert "unicode" in resp.json()["errors"][0]["message"]

    def test_simulation_detectors_rejected_with_guidance(self, client, auth_headers, hf_config):
        for detector in ("kgw", "synthid"):
            resp = self._create(client, auth_headers, _valid_payload(hf_config, detector=detector))
            assert resp.status_code == 422, detector
            assert "reference" in resp.json()["errors"][0]["message"], detector

    def test_missing_config_for_watermark(self, client, auth_headers):
        payload = _valid_payload(None)
        del payload["config"]
        resp = self._create(client, auth_headers, payload)
        assert resp.status_code == 422
        assert any(e["field"] == "config" for e in resp.json()["errors"])

    def test_missing_config_file(self, client, auth_headers):
        resp = self._create(client, auth_headers, _valid_payload("/nonexistent/x.json"))
        assert resp.status_code == 422
        assert any(e["field"] == "config" for e in resp.json()["errors"])

    def test_non_json_config(self, client, auth_headers, tmp_path):
        p = tmp_path / "c.txt"
        p.write_text("{}", encoding="utf-8")
        resp = self._create(client, auth_headers, _valid_payload(str(p)))
        assert resp.status_code == 422

    def test_simulation_config_rejected(self, client, auth_headers, tmp_path):
        p = tmp_path / "sim.json"
        p.write_text(json.dumps({"tokenizer": "simple"}), encoding="utf-8")
        resp = self._create(client, auth_headers, _valid_payload(str(p)))
        assert resp.status_code == 422
        assert "simulation" in resp.json()["errors"][0]["message"].lower()

    def test_unknown_profile(self, client, auth_headers, hf_config):
        resp = self._create(client, auth_headers, _valid_payload(hf_config, profile="nope"))
        assert resp.status_code == 422

    def test_unknown_transform(self, client, auth_headers, hf_config):
        payload = _valid_payload(hf_config, transforms=["nope"])
        del payload["profile"]
        resp = self._create(client, auth_headers, payload)
        assert resp.status_code == 422

    def test_bad_lengths(self, client, auth_headers, hf_config):
        for bad in ([], [0], [501], list(range(11))):
            resp = self._create(client, auth_headers, _valid_payload(hf_config, lengths=bad))
            assert resp.status_code == 422, bad

    def test_bad_samples_seed(self, client, auth_headers, hf_config):
        resp = self._create(client, auth_headers, _valid_payload(hf_config, samples=0))
        assert resp.status_code == 422
        resp = self._create(client, auth_headers, _valid_payload(hf_config, seed="x"))
        assert resp.status_code == 422

    def test_unknown_field(self, client, auth_headers, hf_config):
        resp = self._create(client, auth_headers, _valid_payload(hf_config, bogus=1))
        assert resp.status_code == 422

    def test_requires_auth(self, client, hf_config):
        assert client.post("/v1/benchmark-runs", json=_valid_payload(hf_config)).status_code == 401


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_create_and_complete(self, client, auth_headers, hf_config):
        _use_fake()
        resp = client.post(
            "/v1/benchmark-runs", headers=auth_headers,
            json=_valid_payload(hf_config),
        )
        assert resp.status_code == 201
        run_id = resp.json()["run_id"]
        assert resp.json()["status"] in ("queued", "running", "completed")
        done = _wait_for(client, auth_headers, run_id, ("completed",))
        assert done["progress"]["experiments_total"] == 1
        assert done["progress"]["experiments_completed"] == 1
        assert done["result"]["experiments_failed"] == 0
        assert done["result"]["has_results"] is True
        assert done["duration_ms"] is not None
        assert done["retry_count"] == 0

    def test_partial_failure_is_failure(self, client, auth_headers, hf_config):
        import provenance.api.benchmarks as bench_mod

        def _boom(spec, out_dir):
            raise RuntimeError("generation exploded")

        bench_mod.EXPERIMENT_FN = _boom
        resp = client.post(
            "/v1/benchmark-runs", headers=auth_headers,
            json=_valid_payload(hf_config),
        )
        run_id = resp.json()["run_id"]
        done = _wait_for(client, auth_headers, run_id, ("failed",))
        assert "generation exploded" in done["error_message"]
        assert done["status"] == "failed"  # never success-looking

    def test_cancel_queued(self, client, auth_headers, hf_config):
        import threading
        import provenance.api.benchmarks as bench_mod

        gate = threading.Event()
        calls = {"first": True}

        def _gated(spec, out_dir):
            if calls["first"]:
                calls["first"] = False
                assert gate.wait(timeout=15)
            return _ok_experiment_fn(spec, out_dir)

        bench_mod.EXPERIMENT_FN = _gated
        r1 = client.post("/v1/benchmark-runs", headers=auth_headers,
                         json=_valid_payload(hf_config)).json()["run_id"]
        # Wait until run 1 holds the execution slot
        deadline = time.time() + 10
        while client.get(f"/v1/benchmark-runs/{r1}", headers=auth_headers).json()["status"] != "running":
            assert time.time() < deadline
            time.sleep(0.05)
        r2 = client.post("/v1/benchmark-runs", headers=auth_headers,
                         json=_valid_payload(hf_config)).json()["run_id"]
        # Concurrency cap: run 2 waits behind run 1
        time.sleep(0.3)
        assert client.get(f"/v1/benchmark-runs/{r2}", headers=auth_headers).json()["status"] == "queued"
        cancel = client.post(f"/v1/benchmark-runs/{r2}/cancel", headers=auth_headers)
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"
        # Idempotent
        again = client.post(f"/v1/benchmark-runs/{r2}/cancel", headers=auth_headers)
        assert again.status_code == 200
        gate.set()
        _wait_for(client, auth_headers, r1, ("completed",))

    def test_cancel_terminal_conflicts(self, client, auth_headers, hf_config):
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        _wait_for(client, auth_headers, run_id, ("completed",))
        assert client.post(f"/v1/benchmark-runs/{run_id}/cancel", headers=auth_headers).status_code == 409
        assert client.post(f"/v1/benchmark-runs/{run_id}/retry", headers=auth_headers).status_code == 409

    def test_retry_failed_to_success(self, client, auth_headers, hf_config):
        import provenance.api.benchmarks as bench_mod

        def _flaky(spec, out_dir):
            if _flaky.calls == 0:
                _flaky.calls += 1
                raise RuntimeError("transient")
            return _ok_experiment_fn(spec, out_dir)
        _flaky.calls = 0
        bench_mod.EXPERIMENT_FN = _flaky

        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        _wait_for(client, auth_headers, run_id, ("failed",))
        retry = client.post(f"/v1/benchmark-runs/{run_id}/retry", headers=auth_headers)
        assert retry.status_code == 200
        assert retry.json()["status"] == "queued"
        assert retry.json()["retry_count"] == 1
        assert retry.json()["completed_at"] is None  # stale terminal state cleared
        done = _wait_for(client, auth_headers, run_id, ("completed",))
        assert done["result"]["has_results"] is True

    def test_retry_invalid_status(self, client, auth_headers, hf_config):
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        # Queued/running runs cannot be retried
        resp = client.post(f"/v1/benchmark-runs/{run_id}/retry", headers=auth_headers)
        assert resp.status_code in (200, 409)  # race-tolerant: queued or already completed
        _wait_for(client, auth_headers, run_id, ("completed",))


# ---------------------------------------------------------------------------
# Listing, ownership, recovery, artifacts, privacy
# ---------------------------------------------------------------------------

class TestRunManagement:
    def test_list_and_status_filter(self, client, auth_headers, hf_config):
        _use_fake()
        ids = [
            client.post("/v1/benchmark-runs", headers=auth_headers,
                        json=_valid_payload(hf_config)).json()["run_id"]
            for _ in range(2)
        ]
        for run_id in ids:
            _wait_for(client, auth_headers, run_id, ("completed",))
        data = client.get("/v1/benchmark-runs", headers=auth_headers).json()
        assert data["total"] == 2
        assert len(data["runs"]) == 2
        filtered = client.get("/v1/benchmark-runs", headers=auth_headers,
                              params={"status": "completed"}).json()
        assert filtered["total"] == 2
        assert client.get("/v1/benchmark-runs", headers=auth_headers,
                          params={"status": "queued"}).json()["total"] == 0

    def test_list_requires_key(self, client):
        assert client.get("/v1/benchmark-runs").status_code == 401

    def test_auth_boundaries(self, client, auth_headers, hf_config):
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        assert client.get(f"/v1/benchmark-runs/{run_id}").status_code == 401
        assert client.get(f"/v1/benchmark-runs/{run_id}",
                          headers={"X-API-Key": "wrong-key"}).status_code == 403

    def test_foreign_owner_404_and_admin_access(self, client, auth_headers, hf_config, monkeypatch):
        from provenance.api.keys import generate_api_key, generate_key_id, hash_api_key
        from provenance.api.state import get_repo

        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        # A different managed key sees 404 (not 403): no existence leak.
        other = generate_api_key()
        get_repo().create_api_key(
            key_id=generate_key_id(), key_hash=hash_api_key(other), name="other",
        )
        resp = client.get(f"/v1/benchmark-runs/{run_id}", headers={"X-API-Key": other})
        assert resp.status_code == 404
        # Admin bypasses ownership.
        monkeypatch.setenv("PROVENANCE_ADMIN_API_KEY", "adminkey")
        resp = client.get(f"/v1/benchmark-runs/{run_id}", headers={"X-API-Key": "adminkey"})
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id

    def test_recovery_marks_interrupted_failed(self, client, auth_headers, hf_config, tmp_path):
        from provenance.api.state import get_repo
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        _wait_for(client, auth_headers, run_id, ("completed",))
        # Simulate a crash: flip back to running with an artifact present
        repo = get_repo()
        run = repo.get_benchmark_run(run_id)
        out = Path(run["out_dir"])
        sentinel = out / "keep.me"
        sentinel.write_text("artifact", encoding="utf-8")
        repo.update_benchmark_run(run_id=run_id, status="running")
        recovered = repo.recover_stale_benchmark_runs()
        assert recovered >= 1
        after = repo.get_benchmark_run(run_id)
        assert after["status"] == "failed"
        assert "restart" in after["error_message"]
        assert sentinel.exists()  # artifacts preserved
        # Retry permitted after recovery
        retry = client.post(f"/v1/benchmark-runs/{run_id}/retry", headers=auth_headers)
        assert retry.status_code == 200

    def test_artifact_association_via_run_id(self, client, auth_headers, hf_config):
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        _wait_for(client, auth_headers, run_id, ("completed",))
        data = client.get("/v1/robustness/results", headers=auth_headers,
                          params={"run_id": run_id}).json()
        assert data["total_results"] >= 1
        assert data["summary"]["filters"]["run_id"] == run_id
        # Unrelated run id matches nothing
        empty = client.get("/v1/robustness/results", headers=auth_headers,
                           params={"run_id": "brun-doesnotexist"}).json()
        assert empty["total_results"] == 0

    def test_no_secrets_in_responses(self, client, auth_headers, hf_config):
        _use_fake()
        run_id = client.post("/v1/benchmark-runs", headers=auth_headers,
                             json=_valid_payload(hf_config)).json()["run_id"]
        done = _wait_for(client, auth_headers, run_id, ("completed",))

        def _keys(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k
                    yield from _keys(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from _keys(v)

        forbidden = {"hash_key", "api_key", "watermark_key", "password",
                     "private_key", "credentials", "raw_text", "text"}
        found = {str(k).lower() for k in _keys(done)} & forbidden
        assert not found, f"forbidden fields: {found}"

    def test_get_unknown_run_404(self, client, auth_headers):
        assert client.get("/v1/benchmark-runs/brun-nope", headers=auth_headers).status_code == 404


class TestExecutionCompatibility:
    def test_sim_detector_with_hf_config_fails_fast(self, tmp_path):
        """kgw/synthid + HF config raises guidance before any model load."""
        import pytest

        from provenance.robustness.execution import run_plan_experiment
        from provenance.robustness.orchestration import BenchmarkSpec

        cfg = tmp_path / "hf.json"
        cfg.write_text(json.dumps({
            "model": {"model_identifier": "x"},
            "tokenizer": {"type": "simple", "vocabulary": ["a"]},
        }), encoding="utf-8")
        spec = BenchmarkSpec(
            detector="kgw", config=str(cfg), lengths=(8,),
            samples=1, seed=1,
        )
        with pytest.raises(ValueError, match="reference"):
            run_plan_experiment(spec, tmp_path / "out")

    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        """Regression: threads sharing one repository must not corrupt it."""
        import threading

        from provenance.api.db import SqliteRepository
        repo = SqliteRepository(str(tmp_path / "race.db"))
        errors = []

        def _writer(n):
            try:
                for i in range(25):
                    repo.create_benchmark_run(
                        run_id=f"race-{n}-{i}", key_id="k",
                        config_json="{}", out_dir="/tmp/x",
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        _, total = repo.list_benchmark_runs(key_id="k")
        assert total == 100
