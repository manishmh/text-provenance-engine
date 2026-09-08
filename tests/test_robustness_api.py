"""Tests for Phase 6A: robustness benchmark API endpoints.

Covers detector discovery (already covered in test_phase5a.py), robustness
result retrieval, filtering, comparison rendering data, malformed/empty
responses, API failures, and privacy guarantees.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app

API_KEY = "test-phase6a-key"


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    monkeypatch.setenv("PROVENANCE_API_KEY", API_KEY)


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    app = create_app(db_url=f"sqlite:///{db_path}")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_headers():
    return {"X-API-Key": API_KEY}


def _records(n=5, detected=True):
    return [
        {
            "original_detected": detected,
            "transformed_detected": detected,
            "original_score": 10.0 + i,
            "transformed_score": 10.0 + i,
            "score_delta": 0.0,
            "detection_changed": False,
        }
        for i in range(n)
    ]


@pytest.fixture()
def results_dir(tmp_path, monkeypatch):
    """Directory with two benchmark result files; env pointed at it."""
    from provenance.robustness.benchmark import (
        build_benchmark_result,
        write_benchmark_results,
    )
    d = tmp_path / "robustness"
    d.mkdir()
    write_benchmark_results([
        build_benchmark_result(
            detector_name="kgw", config_identifier="model-a",
            transform_name="identity", text_length=50,
            records=_records(5), seed=42,
        ),
        build_benchmark_result(
            detector_name="kgw", config_identifier="model-a",
            transform_name="lowercase", text_length=50,
            records=_records(5), seed=42,
        ),
        build_benchmark_result(
            detector_name="synthid", config_identifier="model-b",
            transform_name="identity", text_length=100,
            records=_records(4), seed=7,
        ),
    ], d / "run1.jsonl")
    monkeypatch.setenv("PROVENANCE_ROBUSTNESS_DIR", str(d))
    return d


@pytest.fixture()
def empty_results_dir(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setenv("PROVENANCE_ROBUSTNESS_DIR", str(d))
    return d


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestRobustnessAuth:
    def test_results_requires_auth(self, client):
        assert client.get("/v1/robustness/results").status_code == 401

    def test_comparison_requires_auth(self, client):
        assert client.get("/v1/robustness/comparison").status_code == 401


# ---------------------------------------------------------------------------
# Results retrieval
# ---------------------------------------------------------------------------

class TestRobustnessResults:
    def test_empty_dir(self, client, auth_headers, empty_results_dir):
        resp = client.get("/v1/robustness/results", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 0
        assert data["results"] == []
        assert data["aggregated"] == []
        assert data["summary"]["detectors"] == []
        assert data["warnings"] == []

    def test_missing_dir(self, client, auth_headers, tmp_path, monkeypatch):
        monkeypatch.setenv("PROVENANCE_ROBUSTNESS_DIR", str(tmp_path / "nope"))
        resp = client.get("/v1/robustness/results", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["total_results"] == 0

    def test_with_results(self, client, auth_headers, results_dir):
        resp = client.get("/v1/robustness/results", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 3
        assert data["schema_version"] == "provenance-robustness-v1"
        assert data["matrix"] is not None
        assert "kgw" in data["matrix"]["detectors"]
        assert "limitations" in data
        assert data["summary"]["detectors"] == ["kgw", "synthid"]
        assert data["summary"]["configs"] == ["model-a", "model-b"]
        assert "category_aggregation" in data
        assert data["category_map"]["lowercase"] == "casing"
        assert data["category_map"]["identity"] == "baseline"

    def test_matrix_cells_have_rates_and_cis(self, client, auth_headers, results_dir):
        data = client.get("/v1/robustness/results", headers=auth_headers).json()
        cells = data["matrix"]["cells"]
        assert len(cells) > 0
        for cell in cells.values():
            assert 0.0 <= cell["robustness_rate"] <= 1.0
            assert 0.0 <= cell["robustness_ci_low"] <= cell["robustness_ci_high"] <= 1.0
            assert "baseline_rate" in cell
            assert "transformed_rate" in cell

    def test_no_raw_text_or_secrets(self, client, auth_headers, results_dir):
        payload = client.get("/v1/robustness/results", headers=auth_headers).json()

        def _keys(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    yield k
                    yield from _keys(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from _keys(v)

        # Secret-like *field names* must never appear. (Substring matching on
        # the raw dump is avoided: tmp paths and disclaimer copy can contain
        # words like "secret".)
        forbidden = {"hash_key", "api_key", "watermark_key", "password",
                     "private_key", "credentials", "raw_text", "text"}
        found = {k.lower() for k in _keys(payload)} & forbidden
        assert not found, f"forbidden fields in payload: {found}"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestRobustnessFiltering:
    def test_filter_detector(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/results", headers=auth_headers,
            params={"detector": "synthid"},
        ).json()
        assert data["total_results"] == 1
        assert data["summary"]["filters"]["detector"] == "synthid"

    def test_filter_config(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/results", headers=auth_headers,
            params={"config": "model-a"},
        ).json()
        assert data["total_results"] == 2

    def test_filter_transform(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/results", headers=auth_headers,
            params={"transform": "lowercase"},
        ).json()
        assert data["total_results"] == 1

    def test_filter_length(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/results", headers=auth_headers,
            params={"text_length": 100},
        ).json()
        assert data["total_results"] == 1

    def test_filter_no_match(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/results", headers=auth_headers,
            params={"detector": "nonexistent"},
        ).json()
        assert data["total_results"] == 0
        # Summary still describes the full corpus for selector population
        assert data["summary"]["detectors"] == ["kgw", "synthid"]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestRobustnessComparison:
    def test_comparison_structure(self, client, auth_headers, results_dir):
        resp = client.get("/v1/robustness/comparison", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["schema_version"] == "provenance-benchmark-report-v1"
        assert len(data["rows"]) == 3
        assert len(data["by_model"]) == 2
        assert len(data["by_transform"]) == 2
        assert len(data["by_length"]) == 2
        assert "limitations" in data

    def test_comparison_empty(self, client, auth_headers, empty_results_dir):
        data = client.get("/v1/robustness/comparison", headers=auth_headers).json()
        assert data["rows"] == []
        assert data["by_model"] == []

    def test_comparison_filter(self, client, auth_headers, results_dir):
        data = client.get(
            "/v1/robustness/comparison", headers=auth_headers,
            params={"detector": "kgw"},
        ).json()
        assert len(data["rows"]) == 2
        assert len(data["by_model"]) == 1


# ---------------------------------------------------------------------------
# Malformed data
# ---------------------------------------------------------------------------

class TestMalformedData:
    def test_malformed_file_warns_not_fails(self, client, auth_headers, results_dir):
        (results_dir / "bad.jsonl").write_text(
            "not json at all\n", encoding="utf-8")
        resp = client.get("/v1/robustness/results", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 3  # good file still loaded
        assert len(data["warnings"]) == 1
        assert "bad.jsonl" in data["warnings"][0]["file"]

    def test_comparison_malformed_warns(self, client, auth_headers, results_dir):
        (results_dir / "bad.jsonl").write_text("{oops", encoding="utf-8")
        resp = client.get("/v1/robustness/comparison", headers=auth_headers)
        assert resp.status_code == 200
        assert len(resp.json()["warnings"]) == 1


# ---------------------------------------------------------------------------
# Category map
# ---------------------------------------------------------------------------

class TestCategoryMap:
    def test_advanced_categories(self):
        from provenance.api.service import get_transform_category_map
        m = get_transform_category_map()
        assert m["lowercase"] == "casing"
        assert m["unicode_nfkd"] == "unicode"
        assert m["conservative_synonym_substitution"] == "lexical"

    def test_baseline_category(self):
        from provenance.api.service import get_transform_category_map
        m = get_transform_category_map()
        assert m["identity"] == "baseline"
        assert m["whitespace_normalization"] == "baseline"
