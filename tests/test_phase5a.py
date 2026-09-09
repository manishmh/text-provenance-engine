"""Tests for Phase 5A: Detection Engine Expansion & Robustness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from provenance.api.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "test-phase5a-key"


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


# ---------------------------------------------------------------------------
# Detector Registry
# ---------------------------------------------------------------------------


class TestDetectorRegistry:
    def test_registry_singleton(self):
        from provenance.detectors.registry import get_registry
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_registry_contains_unicode(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "unicode" in reg.names()
        entry = reg.get("unicode")
        assert entry is not None
        assert entry.capability.name == "unicode"

    def test_registry_contains_kgw(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "kgw" in reg.names()
        assert reg.get("kgw").capability.requires_config is True

    def test_registry_contains_synthid(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "synthid" in reg.names()
        assert "synthid-reference" in reg.names()

    def test_registry_create_unicode(self):
        from provenance.detectors.registry import get_registry
        detector = get_registry().create("unicode")
        assert detector.name == "unicode"

    def test_registry_create_unknown_raises(self):
        from provenance.detectors.registry import get_registry
        with pytest.raises(ValueError, match="Unknown detector"):
            get_registry().create("nonexistent_detector")

    def test_registry_get_unknown_returns_none(self):
        from provenance.detectors.registry import get_registry
        assert get_registry().get("nonexistent") is None

    def test_all_capability_dicts(self):
        from provenance.detectors.registry import get_registry
        caps = get_registry().all_capability_dicts()
        assert isinstance(caps, list)
        assert len(caps) >= 3
        for cap in caps:
            assert "name" in cap
            assert "display_name" in cap
            assert "implementation_kind" in cap
            assert "compatibility" in cap
            assert "requires_config" in cap
            assert "supports_generation" in cap
            assert "supports_benchmarking" in cap
            assert "known_limitations" in cap
            assert "description" in cap


class TestDetectorCapabilities:
    def test_unicode_capability(self):
        from provenance.detectors.registry import get_registry
        cap = get_registry().capability_dict("unicode")
        assert cap is not None
        assert cap["name"] == "unicode"
        assert cap["requires_config"] is False
        assert cap["supports_generation"] is False
        assert cap["tokenizer_requirements"] is None

    def test_kgw_capability(self):
        from provenance.detectors.registry import get_registry
        cap = get_registry().capability_dict("kgw")
        assert cap is not None
        assert cap["requires_config"] is True
        assert cap["compatibility"] == "gpt-2"

    def test_synthid_capability(self):
        from provenance.detectors.registry import get_registry
        cap = get_registry().capability_dict("synthid")
        assert cap is not None
        assert cap["requires_config"] is True
        assert cap["compatibility"] == "gemini"

    def test_generation_flag_means_generation_pipeline_accepts_name(self):
        # supports_generation = accepted by the reference generation pipeline.
        # Full benchmark runs additionally require evaluation compatibility;
        # runnable names are enumerated in benchmarks.RUNNABLE_DETECTORS.
        from provenance.api import benchmarks
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        for name in ("kgw", "kgw-reference", "synthid", "synthid-reference"):
            assert reg.capability_dict(name)["supports_generation"] is True
        assert set(benchmarks.RUNNABLE_DETECTORS) <= set(reg.names())
        assert reg.capability_dict("unicode")["supports_generation"] is False

    def test_capability_deterministic(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert reg.all_capability_dicts() == reg.all_capability_dicts()


# ---------------------------------------------------------------------------
# /v1/detectors endpoint
# ---------------------------------------------------------------------------


class TestDetectorsEndpoint:
    def test_requires_auth(self, client):
        resp = client.get("/v1/detectors")
        assert resp.status_code == 401

    def test_returns_list(self, client, auth_headers):
        resp = client.get("/v1/detectors", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "detectors" in data
        assert isinstance(data["detectors"], list)

    def test_contain_unicode(self, client, auth_headers):
        resp = client.get("/v1/detectors", headers=auth_headers)
        names = [d["name"] for d in resp.json()["detectors"]]
        assert "unicode" in names

    def test_no_secrets(self, client, auth_headers):
        text = json.dumps(client.get("/v1/detectors", headers=auth_headers).json())
        assert "hash_key" not in text.lower()
        assert "api_key" not in text.lower()
        assert "database_url" not in text.lower()

    def test_deterministic(self, client, auth_headers):
        r1 = client.get("/v1/detectors", headers=auth_headers).json()
        r2 = client.get("/v1/detectors", headers=auth_headers).json()
        assert r1 == r2


# ---------------------------------------------------------------------------
# Robustness Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_identity(self):
        from provenance.robustness.transforms import get_transform
        result = get_transform("identity").apply("Hello World")
        assert result.text == "Hello World"
        assert result.transform_name == "identity"

    def test_whitespace_normalization(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("Hello   World\n\n\n\nEnd", "whitespace_normalization")
        assert "  " not in result.text
        assert "\n\n\n" not in result.text

    def test_unicode_nfc(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("caf\u00e9", "unicode_normalization_nfc")
        assert result.text == "caf\u00e9"

    def test_unicode_nfd(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("caf\u00e9", "unicode_normalization_nfd")
        # NFD decomposes é into e + combining accent
        assert len(result.text) >= len("cafe")

    def test_lowercase(self):
        from provenance.robustness.transforms import apply_transform
        assert apply_transform("HELLO", "lowercase").text == "hello"

    def test_uppercase(self):
        from provenance.robustness.transforms import apply_transform
        assert apply_transform("hello", "uppercase").text == "HELLO"

    def test_punctuation_normalization(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("Hello\u2014World\u2019s", "punctuation_normalization")
        assert "-" in result.text
        assert "'" in result.text

    def test_strip_blank_lines(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("Line 1\n\n\nLine 2", "strip_blank_lines")
        assert "\n\n" not in result.text

    def test_add_leading_trailing_whitespace(self):
        from provenance.robustness.transforms import apply_transform
        result = apply_transform("text", "add_leading_trailing_whitespace")
        assert result.text.startswith("  ")
        assert result.text.endswith("  ")

    def test_double_spaces(self):
        from provenance.robustness.transforms import apply_transform
        assert "  " in apply_transform("a b c", "double_spaces").text

    def test_all_deterministic(self):
        from provenance.robustness.transforms import get_all_transforms
        text = "The quick brown fox jumps over the lazy dog."
        for t in get_all_transforms():
            r1, r2 = t.apply(text), t.apply(text)
            assert r1.text == r2.text, f"{t.name} not deterministic"

    def test_no_input_mutation(self):
        from provenance.robustness.transforms import get_all_transforms
        original = "Hello World"
        for t in get_all_transforms():
            t.apply(original)
            assert original == "Hello World"

    def test_all_available(self):
        from provenance.robustness.transforms import get_all_transforms
        names = [t.name for t in get_all_transforms()]
        assert "identity" in names
        assert "whitespace_normalization" in names
        assert "lowercase" in names

    def test_unknown_raises(self):
        from provenance.robustness.transforms import get_transform
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform("nonexistent")

    def test_transform_metadata(self):
        from provenance.robustness.transforms import apply_transform
        r = apply_transform("Hello World", "whitespace_normalization")
        assert "original_length" in r.metadata


# ---------------------------------------------------------------------------
# Robustness Evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_evaluate_single(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import evaluate_single
        from provenance.robustness.transforms import get_transform

        detector = get_registry().create("unicode")
        record = evaluate_single("Hello World with \u00e9", detector, get_transform("identity"))
        assert record.detector_name == "unicode"
        assert record.original_hash is not None
        assert record.duration_ms >= 0

    def test_evaluate_robustness(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import evaluate_robustness

        records = evaluate_robustness("Hello \u00e9", get_registry().create("unicode"))
        assert len(records) >= 5
        for r in records:
            assert r.detector_name == "unicode"

    def test_compute_statistics(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import compute_statistics, evaluate_robustness

        records = evaluate_robustness("Hello \u00e9", get_registry().create("unicode"))
        stats = compute_statistics(records)
        assert "total_transforms" in stats
        assert "overall_detection_rate" in stats
        assert "transforms" in stats

    def test_no_raw_text_in_records(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import evaluate_robustness

        text = "Secret \u00e9 text"
        records = evaluate_robustness(text, get_registry().create("unicode"))
        for r in records:
            d = r.to_dict()
            assert "text" not in d
            assert text not in json.dumps(d)

    def test_compute_statistics_empty(self):
        from provenance.robustness.evaluator import compute_statistics
        stats = compute_statistics([])
        assert stats["total_transforms"] == 0


# ---------------------------------------------------------------------------
# CLI robustness benchmark
# ---------------------------------------------------------------------------


class TestCLIRobustness:
    def test_robustness_benchmark_unicode_json(self, tmp_path):
        from provenance.cli import main

        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello World with \u00e9\u00e8\u00ea", encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--text", str(text_file),
            "--detector", "unicode",
            "--json",
        ])
        assert ret == 0

    def test_robustness_benchmark_text_output(self, tmp_path):
        from provenance.cli import main

        text_file = tmp_path / "test.txt"
        text_file.write_text("Test text with \u00e9.", encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--text", str(text_file),
            "--detector", "unicode",
        ])
        assert ret == 0

    def test_watermark_requires_config(self, tmp_path):
        from provenance.cli import main

        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello", encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--text", str(text_file),
            "--detector", "kgw",
        ])
        assert ret == 1

    def test_watermark_invalid_lengths_rejected(self):
        """Malformed --lengths must fail cleanly (exit 1), not traceback."""
        from provenance.cli import main

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--detector", "kgw",
            "--lengths", "abc",
        ])
        assert ret == 1

    def test_watermark_nonpositive_lengths_rejected(self):
        from provenance.cli import main

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--detector", "kgw",
            "--lengths", "50,0",
        ])
        assert ret == 1

    def test_watermark_zero_samples_rejected(self):
        from provenance.cli import main

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--detector", "kgw",
            "--samples", "0",
        ])
        assert ret == 1


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_valid_detectors_set(self):
        from provenance.api.models import _VALID_DETECTORS
        for name in ("unicode", "kgw", "kgw-reference", "synthid", "synthid-reference"):
            assert name in _VALID_DETECTORS

    def test_watermark_detectors_set(self):
        from provenance.api.models import _WATERMARK_DETECTORS
        assert "unicode" not in _WATERMARK_DETECTORS
        assert "kgw" in _WATERMARK_DETECTORS

    def test_unicode_analysis_still_works(self, client, auth_headers):
        resp = client.post(
            "/v1/analyze",
            json={"text": "Hello World", "detectors": ["unicode"]},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
