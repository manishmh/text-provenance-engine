"""Tests for Phase 6B: model/tokenizer loading cache, records, and docs.

All cache lifecycle tests use fake loaders — no Hugging Face downloads.
Real cold-vs-cached timing lives in /tmp scripts, not the suite.
"""
from __future__ import annotations

import threading

import pytest


# ---------------------------------------------------------------------------
# ResourceCache with fake loaders
# ---------------------------------------------------------------------------

def _make_cache(maxsize=2):
    from provenance.loading import ResourceCache
    return ResourceCache("test", maxsize=maxsize)


class TestResourceCacheHits:
    def test_hit_reuses_instance(self):
        cache = _make_cache()
        calls = {"n": 0}

        def _loader():
            calls["n"] += 1
            return object()

        a = cache.get_or_load("k", _loader, label="k")
        b = cache.get_or_load("k", _loader, label="k")
        assert a is b
        assert calls["n"] == 1
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

    def test_loader_exception_not_cached(self):
        cache = _make_cache()
        calls = {"n": 0}

        def _flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return "ok"

        with pytest.raises(RuntimeError):
            cache.get_or_load("k", _flaky)
        assert cache.get_or_load("k", _flaky) == "ok"
        assert calls["n"] == 2
        assert cache.stats()["size"] == 1


class TestCacheSeparation:
    def test_distinct_keys_distinct_instances(self):
        cache = _make_cache()
        a = cache.get_or_load("k1", lambda: object())
        b = cache.get_or_load("k2", lambda: object())
        assert a is not b
        assert cache.stats()["size"] == 2


class TestEviction:
    def test_lru_eviction(self):
        cache = _make_cache(maxsize=1)
        loads = {"n": 0}

        def _loader():
            loads["n"] += 1
            return f"v{loads['n']}"

        cache.get_or_load("k1", _loader)
        cache.get_or_load("k2", _loader)  # evicts k1
        assert cache.stats()["evictions"] == 1
        assert cache.stats()["size"] == 1
        cache.get_or_load("k1", _loader)  # miss again
        assert loads["n"] == 3

    def test_maxsize_zero_disables(self):
        cache = _make_cache(maxsize=0)
        calls = {"n": 0}

        def _loader():
            calls["n"] += 1
            return object()

        cache.get_or_load("k", _loader)
        cache.get_or_load("k", _loader)
        assert calls["n"] == 2
        assert cache.stats()["size"] == 0

    def test_clear_releases(self):
        cache = _make_cache()
        cache.get_or_load("k1", lambda: object())
        cache.get_or_load("k2", lambda: object())
        assert cache.clear() == 2
        assert cache.stats()["size"] == 0
        assert cache.clear() == 0


class TestConcurrency:
    def test_single_flight(self):
        """Concurrent first-use loads exactly once; all threads share."""
        cache = _make_cache()
        loads = {"n": 0}
        barrier = threading.Barrier(8)

        def _loader():
            loads["n"] += 1
            return object()

        seen = []

        def _worker():
            barrier.wait()
            seen.append(cache.get_or_load("k", _loader))

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert loads["n"] == 1
        assert all(v is seen[0] for v in seen)


class TestSharedModelLock:
    def test_inference_serialized(self):
        """Concurrent next_token_logits calls never overlap."""
        from provenance.loading import _SharedModel

        active = {"n": 0, "max": 0}
        guard = threading.Lock()

        class _FakeModel:
            identifier = "fake"
            vocab_size = 10

            def next_token_logits(self, input_ids):
                with guard:
                    active["n"] += 1
                    active["max"] = max(active["max"], active["n"])
                import time
                time.sleep(0.01)
                with guard:
                    active["n"] -= 1
                return [0.0] * 10

        shared = _SharedModel(_FakeModel())
        threads = [threading.Thread(target=shared.next_token_logits, args=([1],)) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert active["max"] == 1

    def test_attribute_passthrough(self):
        from provenance.loading import _SharedModel

        class _FakeModel:
            identifier = "fake"
            vocab_size = 32000
            resolved_revision = "abc"

        shared = _SharedModel(_FakeModel())
        assert shared.vocab_size == 32000
        assert shared.identifier == "fake"
        assert shared.resolved_revision == "abc"


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

class TestFingerprints:
    def test_model_key_covers_load_fields(self):
        from provenance.loading import model_cache_key
        base = dict(model_identifier="m", revision="r", device="cpu",
                    dtype="float32", local_files_only=False)
        same = model_cache_key(**base)
        for field, other in [("revision", "r2"), ("device", "cuda"),
                             ("dtype", "float16"), ("local_files_only", True),
                             ("model_identifier", "m2")]:
            varied = dict(base)
            varied[field] = other
            assert model_cache_key(**varied) != same, field

    def test_model_key_ignores_tokenizer_identifier(self):
        """tokenizer_identifier is not consumed by the model loader."""
        from provenance.loading import model_cache_key
        # No parameter exists for it: the signature is the allow-list.
        import inspect
        assert "tokenizer_identifier" not in inspect.signature(model_cache_key).parameters

    def test_tokenizer_key_covers_factory_dict(self):
        from provenance.loading import tokenizer_cache_key
        a = {"type": "huggingface", "tokenizer_identifier": "t", "tokenizer_revision": "r1"}
        b = dict(a, tokenizer_revision="r2")
        assert tokenizer_cache_key(a) != tokenizer_cache_key(b)
        assert tokenizer_cache_key(a) == tokenizer_cache_key(dict(a))


# ---------------------------------------------------------------------------
# get_model / get_tokenizer integration (fake HF layer)
# ---------------------------------------------------------------------------

class _FakeHFModel:
    instances = 0

    def __init__(self, config):
        type(self).instances += 1
        self.identifier = config.model_identifier
        self.vocab_size = 100
        self.resolved_revision = config.revision or "main"

    def next_token_logits(self, input_ids):
        return [1.0] * 100


@pytest.fixture()
def _fake_hf(monkeypatch):
    import provenance.models as models_mod
    import provenance.tokenizers as tok_mod
    monkeypatch.setattr(models_mod, "HuggingFaceCausalLM", _FakeHFModel)
    _FakeHFModel.instances = 0

    made = {"n": 0}

    def _fake_factory(factory_dict):
        made["n"] += 1

        class _T:
            identifier = factory_dict.get("tokenizer_identifier", "?")
            vocabulary_size = 100

            def encode(self, text):
                return [1, 2, 3]

            def decode(self, ids):
                return "x"

        return _T()

    monkeypatch.setattr(tok_mod, "tokenizer_from_config", _fake_factory)
    # loading.py imports the names lazily from the same modules, so the
    # monkeypatch applies.
    from provenance import loading as loading_mod
    monkeypatch.setattr(loading_mod, "_model_cache", loading_mod.ResourceCache("model-test", 2))
    monkeypatch.setattr(loading_mod, "_tokenizer_cache", loading_mod.ResourceCache("tok-test", 2))
    return made


def _model_config(**overrides):
    from provenance.configuration import ModelConfig
    base = dict(model_identifier="m", revision="r1", device="cpu",
                dtype="float32", local_files_only=False)
    base.update(overrides)
    return ModelConfig(**base)


class TestGetModel:
    def test_reuse_across_calls(self, _fake_hf):
        from provenance import loading as loading_mod
        a = loading_mod.get_model(_model_config())
        b = loading_mod.get_model(_model_config())
        assert a is b
        assert _FakeHFModel.instances == 1

    def test_revision_change_reloads(self, _fake_hf):
        from provenance import loading as loading_mod
        loading_mod.get_model(_model_config(revision="r1"))
        loading_mod.get_model(_model_config(revision="r2"))
        assert _FakeHFModel.instances == 2

    def test_stats_and_clear(self, _fake_hf):
        from provenance import loading as loading_mod
        loading_mod.get_model(_model_config())
        stats = loading_mod.cache_stats()
        assert stats["model"]["loads"] == 1
        assert stats["model"]["size"] == 1
        assert loading_mod.clear_caches() == {"model": 1, "tokenizer": 0}


class TestGetTokenizer:
    def test_hf_tokenizer_cached(self, _fake_hf):
        from provenance import loading as loading_mod
        fd = {"type": "huggingface", "tokenizer_identifier": "t"}
        a = loading_mod.get_tokenizer(fd)
        b = loading_mod.get_tokenizer(fd)
        assert a is b
        assert _fake_hf["n"] == 1

    def test_simple_tokenizer_bypasses_cache(self, _fake_hf):
        from provenance import loading as loading_mod
        fd = {"type": "simple-vocabulary", "vocabulary": ["a", "b"]}
        a = loading_mod.get_tokenizer(fd)
        b = loading_mod.get_tokenizer(fd)
        assert a is not b
        stats = loading_mod.cache_stats()
        assert stats["tokenizer"]["size"] == 0


# ---------------------------------------------------------------------------
# EvaluationRecord.transform_name compatibility
# ---------------------------------------------------------------------------

def _record_kwargs(**overrides):
    kw = dict(
        experiment_id="e", timestamp="t", benchmark_version="v",
        scheme="kgw", variant="v", configuration_id="c",
        configuration_version="1", implementation_kind="k",
        compatibility="gpt-2", model_identifier="m",
        model_revision=None, tokenizer_identifier="t",
        tokenizer_revision=None, vocab_size=10, hash_key_id="h",
        temperature=1.0, top_p=None, top_k=None, random_seed=1,
        prompt_id="p", target_length=10, watermarked=True,
        token_count=10, scored_token_count=10,
        detection_threshold=0.0, detected=True, score=5.0,
    )
    kw.update(overrides)
    return kw


class TestTransformNameField:
    def test_roundtrip(self):
        from provenance.benchmark.records import EvaluationRecord
        r = EvaluationRecord(**_record_kwargs(transform_name="lowercase"))
        restored = EvaluationRecord.from_dict(r.to_dict())
        assert restored == r
        assert restored.transform_name == "lowercase"

    def test_legacy_record_loads_with_none(self):
        """Records written before the field existed load with None."""
        from provenance.benchmark.records import EvaluationRecord
        r = EvaluationRecord(**_record_kwargs())
        d = r.to_dict()
        del d["transform_name"]
        restored = EvaluationRecord.from_dict(d)
        assert restored.transform_name is None
        assert restored == EvaluationRecord(**_record_kwargs())

    def test_jsonl_roundtrip(self, tmp_path):
        from provenance.benchmark.records import (
            EvaluationRecord, read_jsonl, write_jsonl,
        )
        records = [
            EvaluationRecord(**_record_kwargs(transform_name="identity")),
            EvaluationRecord(**_record_kwargs(transform_name="lowercase")),
        ]
        path = tmp_path / "records.jsonl"
        write_jsonl(records, path)
        loaded = read_jsonl(path)
        assert [r.transform_name for r in loaded] == ["identity", "lowercase"]

    def test_raw_key_still_rejected(self):
        from provenance.benchmark.records import EvaluationRecord
        d = EvaluationRecord(**_record_kwargs()).to_dict()
        d["hash_key"] = "secret"
        with pytest.raises(ValueError, match="hash_key"):
            EvaluationRecord.from_dict(d)


# ---------------------------------------------------------------------------
# Startup/config documentation guards
# ---------------------------------------------------------------------------

class TestStartupDocs:
    def _repo_root(self):
        from pathlib import Path
        return Path(__file__).resolve().parent.parent

    def test_readme_uses_factory_invocation(self):
        text = (self._repo_root() / "README.md").read_text()
        assert "uvicorn --factory provenance.api.app:create_app" in text
        assert "uvicorn provenance.api.app:app " not in text

    def test_env_example_documents_vars(self):
        text = (self._repo_root() / ".env.example").read_text()
        assert "PROVENANCE_ROBUSTNESS_DIR" in text
        assert "PROVENANCE_MODEL_CACHE_SIZE" in text

    def test_readme_documents_cache_and_downloads(self):
        text = (self._repo_root() / "README.md").read_text()
        assert "PROVENANCE_MODEL_CACHE_SIZE" in text
        assert "HF_HUB_OFFLINE" in text
