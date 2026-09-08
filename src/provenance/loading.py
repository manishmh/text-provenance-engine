"""Bounded, thread-safe cache for Hugging Face model/tokenizer instances.

Benchmark and generation paths repeatedly need the same heavy artifacts:
an ``AutoModelForCausalLM`` (seconds to minutes to load, gigabytes of RAM)
and an ``AutoTokenizer`` (seconds). This module caches them across calls,
specs, and threads within one process.

Safety rules (correctness over parallelism):

- Cache identity includes every configuration field that materially affects
  the loaded artifact. A different model revision, dtype, device, or
  tokenizer setting is a different cache entry — configs can never
  contaminate each other.
- Only Hugging Face-backed loads are cached. Cheap ``simple-vocabulary``
  tokenizers are constructed fresh on every call (microseconds, no shared
  state to reason about).
- All cache mutations are serialized by a single lock, so concurrent
  first-use of the same artifact loads it exactly once.
- Model *inference* is serialized per cached instance (``_SharedModel``):
  transformer forward passes are treated as not safely reentrant across
  threads. Tokenizer ``encode``/``decode`` is stateless and needs no lock.
- The cache is bounded (LRU eviction, default max 2 entries per kind) and
  never grows without limit. ``maxsize=0`` disables caching entirely.
- Cached outputs are bit-identical to fresh loads: the same weights and
  tokenizer produce the same logits, so benchmark determinism is preserved.

Nothing here ever handles raw benchmark text, watermark keys, or
credentials; log lines carry only identifiers, sizes, and durations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger("provenance.loading")

#: Environment variable controlling cache capacity per artifact kind.
CACHE_SIZE_ENV_VAR = "PROVENANCE_MODEL_CACHE_SIZE"

#: Default entries retained per kind (models are gigabytes; keep small).
DEFAULT_CACHE_SIZE = 2

#: Tokenizer factory ``type`` values routed through the cache.
CACHED_TOKENIZER_TYPES = frozenset({"hf", "huggingface", "transformers"})

T = TypeVar("T")


def _canonical_fingerprint(payload: Any) -> str:
    """Stable SHA-256 fingerprint of a JSON-serializable payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def model_cache_key(
    *,
    model_identifier: str,
    revision: str | None,
    device: str,
    dtype: str,
    local_files_only: bool,
) -> str:
    """Fingerprint every ``ModelConfig`` field consumed by the model load.

    ``tokenizer_identifier`` is deliberately excluded: the model loader
    never reads it, so it cannot affect the artifact.
    """
    return "model:" + _canonical_fingerprint({
        "model_identifier": model_identifier,
        "revision": revision,
        "device": device,
        "dtype": dtype,
        "local_files_only": local_files_only,
    })


def tokenizer_cache_key(factory_dict: dict[str, Any]) -> str:
    """Fingerprint the full tokenizer factory dict (every field matters)."""
    return "tokenizer:" + _canonical_fingerprint(factory_dict)


def _get_maxsize() -> int:
    """Read cache capacity from the environment (defensive)."""
    try:
        return max(0, int(os.environ.get(CACHE_SIZE_ENV_VAR, DEFAULT_CACHE_SIZE)))
    except (TypeError, ValueError):
        return DEFAULT_CACHE_SIZE


class ResourceCache(Generic[T]):
    """Bounded LRU cache with single-flight loading and usage stats."""

    def __init__(self, name: str, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        self._name = name
        self._maxsize = max(0, maxsize)
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._loads = 0
        self._load_ms_total = 0.0

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def get_or_load(self, key: str, loader: Callable[[], T], *, label: str = "") -> T:
        """Return the cached value, loading (and storing) it on a miss.

        The whole check-and-load is serialized: concurrent first-use from
        many threads invokes ``loader`` exactly once. When ``maxsize`` is 0
        the loader runs on every call and nothing is stored.
        """
        display = label or key[:24]
        with self._lock:
            if self._maxsize > 0 and key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                logger.debug("%s cache hit: %s", self._name, display)
                return self._entries[key]
            self._misses += 1
        logger.info("%s cache miss, loading: %s", self._name, display)
        started = time.perf_counter()
        value = loader()
        load_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._loads += 1
            self._load_ms_total += load_ms
            if self._maxsize <= 0:
                return value
            self._entries[key] = value
            while len(self._entries) > self._maxsize:
                evicted_key, _ = self._entries.popitem(last=False)
                self._evictions += 1
                logger.info("%s cache evict: %s", self._name, evicted_key[:24])
                _release_gpu_memory()
        logger.info("%s loaded in %.0fms: %s", self._name, load_ms, display)
        return value

    def clear(self) -> int:
        """Drop all cached references. Returns the number of entries released."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        if count:
            logger.info("%s cache cleared (%d entries released)", self._name, count)
            _release_gpu_memory()
        return count

    def stats(self) -> dict[str, Any]:
        """Snapshot of cache counters (safe to expose in diagnostics)."""
        with self._lock:
            return {
                "name": self._name,
                "size": len(self._entries),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "loads": self._loads,
                "load_ms_total": round(self._load_ms_total, 1),
            }


def _release_gpu_memory() -> None:
    """Best-effort GPU cache release without importing torch."""
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    try:
        torch_mod.cuda.empty_cache()
    except Exception:
        pass


class _SharedModel:
    """Thread-safe handle around a loaded causal LM.

    Inference (``next_token_logits``) is serialized with a per-instance
    lock; all other attribute reads delegate to the wrapped model and are
    immutable after construction.
    """

    def __init__(self, model: Any, *, label: str = "") -> None:
        object.__setattr__(self, "_wrapped", model)
        object.__setattr__(self, "_label", label or getattr(model, "identifier", "?"))
        object.__setattr__(self, "_io_lock", threading.Lock())

    def next_token_logits(self, input_ids: list[int]) -> list[float]:
        lock: threading.Lock = object.__getattribute__(self, "_io_lock")
        with lock:
            return object.__getattribute__(self, "_wrapped").next_token_logits(input_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)


# ── Process-global caches ──────────────────────────────────────────────

_model_cache = ResourceCache[Any]("model", _get_maxsize())
_tokenizer_cache = ResourceCache[Any]("tokenizer", _get_maxsize())


def get_model(config: Any) -> _SharedModel:
    """Return a cached shared model for a ``ModelConfig``.

    The cache key covers identifier, revision, device, dtype, and
    ``local_files_only`` — every field the loader consumes.
    """
    key = model_cache_key(
        model_identifier=config.model_identifier,
        revision=config.revision,
        device=config.device,
        dtype=config.dtype,
        local_files_only=config.local_files_only,
    )

    def _load() -> _SharedModel:
        from provenance.models import HuggingFaceCausalLM
        return _SharedModel(HuggingFaceCausalLM(config), label=config.model_identifier)

    return _model_cache.get_or_load(key, _load, label=config.model_identifier)


def get_tokenizer(factory_dict: dict[str, Any]) -> Any:
    """Return a tokenizer for a factory dict, cached when HF-backed.

    Cheap ``simple-vocabulary`` tokenizers bypass the cache and are built
    fresh (no shared state, negligible cost).
    """
    if str(factory_dict.get("type", "simple-vocabulary")) not in CACHED_TOKENIZER_TYPES:
        from provenance.tokenizers import tokenizer_from_config
        return tokenizer_from_config(factory_dict)

    def _load() -> Any:
        from provenance.tokenizers import tokenizer_from_config
        return tokenizer_from_config(factory_dict)

    label = str(factory_dict.get("tokenizer_identifier", "?"))
    return _tokenizer_cache.get_or_load(tokenizer_cache_key(factory_dict), _load, label=label)


def cache_stats() -> dict[str, dict[str, Any]]:
    """Snapshot of model/tokenizer cache counters for diagnostics."""
    return {"model": _model_cache.stats(), "tokenizer": _tokenizer_cache.stats()}


def clear_caches() -> dict[str, int]:
    """Release all cached models/tokenizers. Returns per-kind release counts."""
    return {
        "model": _model_cache.clear(),
        "tokenizer": _tokenizer_cache.clear(),
    }
