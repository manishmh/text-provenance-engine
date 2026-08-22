"""Reference-backed SynthID-Text detector.

Implements the SynthID-Text weighted-mean detection algorithm following the
Google DeepMind public reference implementation (synthid-text, Apache-2.0).

Key differences from the controlled simulation in ``detectors.simulated.synthid``:
- Hashing uses the same adapted linear congruential generator (LCG) as the
  reference, not ``blake2b``-based stable hashes.
- G-values are derived via iterative accumulate-hash + bit extraction, matching
  the reference ``get_gvals`` implementation.
- Context repetition detection uses hash-based sliding windows.

Classification: ``implementation_kind="reference"``,
``compatibility="synthid-reference"``.

This detector scores *known* SynthID configurations.  It is NOT general
AI-vs-human detection and does NOT identify Gemini unless the configuration is
independently known to correspond to Gemini.  Production Gemini
keys/configurations are unavailable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.preprocessing.text import stable_hash_int
from provenance.schemas import DetectionResult
from provenance.tokenizers import ProvenanceTokenizer, SimpleVocabularyTokenizer


# ---------------------------------------------------------------------------
# Pure-Python accumulate_hash  (matches DeepMind synthid-text)
# ---------------------------------------------------------------------------

# LCG parameters from newlib / musl (used in the reference implementation).
_LCG_MULTIPLIER = 6364136223846793005
_LCG_INCREMENT = 1
_UINT64_MOD = 1 << 64
_INT64_MAX = (1 << 63) - 1  # torch.iinfo(torch.int64).max


def _to_unsigned(v: int) -> int:
    """Convert to unsigned 64-bit value (matching PyTorch int64 wrapping)."""
    return v % _UINT64_MOD


def _to_signed(v: int) -> int:
    """Convert unsigned 64-bit value to signed int64."""
    v = v % _UINT64_MOD
    if v >= (1 << 63):
        return v - _UINT64_MOD
    return v


def accumulate_hash(
    current_hash: int,
    data: tuple[int, ...],
) -> int:
    """Accumulate hash of *data* on *current_hash* using an adapted LCG.

    This is a pure-Python reimplementation of
    ``synthid_text.hashing_function.accumulate_hash`` and produces identical
    results for equivalent integer inputs.  The LCG is iterated once per
    element in *data*.

    Wrapping matches PyTorch ``torch.int64`` arithmetic: intermediate values
    wrap at 2^64 (unsigned), and the result is stored as an unsigned 64-bit
    integer that can be interpreted as signed int64.

    Parameters
    ----------
    current_hash:
        Starting hash value (an integer).
    data:
        Sequence of integers to hash.

    Returns
    -------
    int
        The updated hash value as an unsigned 64-bit integer.
    """
    h = _to_unsigned(current_hash)
    for value in data:
        h = _to_unsigned(h + value)
        h = _to_unsigned(h * _LCG_MULTIPLIER)
        h = _to_unsigned(h + _LCG_INCREMENT)
    return h


def _signed_right_shift(v: int, shift: int) -> int:
    """Arithmetic right shift matching PyTorch int64 behaviour.

    PyTorch's ``>>`` on ``torch.int64`` performs arithmetic (sign-extending)
    right shift.  This helper converts the unsigned value to signed, shifts,
    and converts back to unsigned.
    """
    signed = _to_signed(v)
    shifted = signed >> shift
    return _to_unsigned(shifted)


def _hash_iv_from_keys(keys: tuple[int, ...]) -> int:
    """Derive the hash initialization vector (IV) from watermarking keys.

    Matches the reference: ``int.from_bytes(sha256(keys.tobytes()).digest(),
    'big') % torch.iinfo(torch.int64).max``.
    """
    key_bytes = b"".join(k.to_bytes(8, byteorder="little", signed=False) for k in keys)
    digest = hashlib.sha256(key_bytes).digest()
    return int.from_bytes(digest, byteorder="big") % _INT64_MAX


# ---------------------------------------------------------------------------
# G-value computation
# ---------------------------------------------------------------------------

def _g_value_for_ngram(
    ngram_ids: tuple[int, ...],
    *,
    hash_iv: int,
    key: int,
    depth: int,
) -> int:
    """Compute a single binary g-value for an n-gram at a given depth.

    Matches the reference ``SynthIDLogitsProcessor.get_gvals`` after
    ``compute_ngram_keys``.

    The computation is:
    1. Start with ``hash_iv``.
    2. ``accumulate_hash`` over the n-gram token IDs.
    3. ``accumulate_hash`` over the single key value.
    4. Apply 12 rounds of ``accumulate_hash([1])`` with a 5-bit arithmetic
       right shift (matching PyTorch int64 behaviour).
    5. Extract bit 30 as the g-value (0 or 1).
    """
    h = hash_iv
    h = accumulate_hash(h, ngram_ids)
    h = accumulate_hash(h, (key,))
    for _ in range(12):
        h = accumulate_hash(h, (1,))
        h = _signed_right_shift(h, 5)
    return (h >> 30) & 1


def _compute_g_values(
    token_ids: list[int],
    *,
    ngram_len: int,
    hash_iv: int,
    keys: tuple[int, ...],
) -> list[list[int]]:
    """Compute g-values for all valid n-gram positions in *token_ids*.

    Returns a list of length ``len(token_ids)``.  Positions before
    ``ngram_len - 1`` have all-zero g-values (insufficient context).
    """
    g_values: list[list[int]] = []
    num_tokens = len(token_ids)
    for index in range(num_tokens):
        if index < ngram_len - 1:
            g_values.append([0] * len(keys))
            continue
        ngram = tuple(token_ids[index - ngram_len + 1 : index + 1])
        g_vals = [
            _g_value_for_ngram(ngram, hash_iv=hash_iv, key=key, depth=depth)
            for depth, key in enumerate(keys)
        ]
        g_values.append(g_vals)
    return g_values


def _compute_context_repetition_mask(
    token_ids: list[int],
    *,
    ngram_len: int,
    hash_iv: int,
    context_history_size: int,
) -> list[bool]:
    """Compute a boolean mask where ``True`` means *not* a repeated context.

    Matches the reference ``compute_context_repetition_mask``.  The context
    hash for position *i* is ``accumulate_hash(hash_iv, ngram_ids)``, where
    ``ngram_ids`` is the (n-1)-gram ending at position *i* (i.e.
    ``token_ids[i - ngram_len + 1 : i]``).

    Positions before ``ngram_len - 1`` are marked ``False`` (insufficient
    context).
    """
    mask: list[bool] = []
    history: list[int] = []

    for index in range(len(token_ids)):
        if index < ngram_len - 1:
            mask.append(False)
            continue

        # (n-1)-gram context ending at this position.
        ctx_ids = tuple(token_ids[index - ngram_len + 1 : index])
        ctx_hash = accumulate_hash(hash_iv, ctx_ids)
        is_repeated = ctx_hash in history
        mask.append(not is_repeated)

        history.append(ctx_hash)
        if len(history) > context_history_size:
            history.pop(0)

    return mask


# ---------------------------------------------------------------------------
# Weighted-mean scoring
# ---------------------------------------------------------------------------

def _default_weights(depth: int) -> list[float]:
    """Default linearly decreasing weights from 10 to 1, normalised.

    Matches the reference ``jnp.linspace(start=10, stop=1, num=depth)``
    followed by normalisation so weights sum to *depth*.
    """
    if depth <= 1:
        return [1.0]
    weights = [10.0 - i * 9.0 / (depth - 1) for i in range(depth)]
    total = sum(weights)
    return [w * depth / total for w in weights]


def _weighted_mean_score(
    g_values: list[list[int]],
    mask: list[bool],
    weights: list[float],
    watermarking_depth: int,
) -> float:
    """Compute the weighted-mean score over unmasked g-values.

    Matches ``detector_mean.weighted_mean_score``.
    """
    usable_count = sum(mask)
    if usable_count == 0:
        return 0.0
    weighted_sum = 0.0
    for values, usable in zip(g_values, mask):
        if not usable:
            continue
        weighted_sum += sum(v * w for v, w in zip(values, weights))
    return weighted_sum / (watermarking_depth * usable_count)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYNTHID_REFERENCE_LIMITATIONS = [
    "Requires the same SynthID-Text configuration used during generation.",
    "This is evidence for the tested SynthID configuration only.",
    "It does not prove AI authorship or human authorship.",
    "It does not identify Gemini unless the configuration is independently known to correspond to Gemini.",
    "Production Gemini keys/configuration are unavailable.",
    "A negative result is not evidence of human authorship.",
    "Weighted-mean detection is less powerful than the Bayesian detector.",
]


@dataclass(frozen=True)
class SynthIDReferenceConfig:
    """Configuration for the reference-backed SynthID-Text detector.

    The ``hash_key_id`` field is a human-readable identifier; the raw key
    material is never stored in records or serialized output.
    """

    configuration_id: str
    version: str
    keys: tuple[int, ...]
    ngram_len: int
    threshold: float
    hash_key_id: str
    vocabulary: tuple[str, ...] = ()
    tokenizer: str = "simple"
    case_sensitive: bool = False
    sampling_table_size: int = 4096
    sampling_table_seed: int = 0
    context_history_size: int = 1024
    weights: tuple[float, ...] | None = None
    min_usable_tokens: int = 20
    # HuggingFace model/tokenizer support (optional)
    model_identifier: str | None = None
    model_revision: str | None = None
    tokenizer_identifier: str | None = None
    tokenizer_revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = False
    # Generation parameters
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_p: float | None = 0.95
    top_k: int | None = None
    seed: int = 42

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SynthIDReferenceConfig":
        required = [
            "configuration_id",
            "version",
            "keys",
            "ngram_len",
            "threshold",
            "hash_key_id",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetectorConfigurationError(
                f"missing SynthID reference config fields: {', '.join(missing)}"
            )

        weights = data.get("weights")
        vocab_raw = data.get("vocabulary", ())
        return cls(
            configuration_id=str(data["configuration_id"]),
            version=str(data["version"]),
            keys=tuple(int(key) for key in data["keys"]),
            ngram_len=int(data["ngram_len"]),
            threshold=float(data["threshold"]),
            hash_key_id=str(data["hash_key_id"]),
            vocabulary=tuple(str(token) for token in vocab_raw) if vocab_raw else (),
            tokenizer=str(data.get("tokenizer", "simple")),
            case_sensitive=bool(data.get("case_sensitive", False)),
            sampling_table_size=int(data.get("sampling_table_size", 4096)),
            sampling_table_seed=int(data.get("sampling_table_seed", 0)),
            context_history_size=int(data.get("context_history_size", 1024)),
            weights=tuple(float(w) for w in weights) if weights is not None else None,
            min_usable_tokens=int(data.get("min_usable_tokens", 20)),
            model_identifier=data.get("model_identifier"),
            model_revision=data.get("model_revision"),
            tokenizer_identifier=data.get("tokenizer_identifier"),
            tokenizer_revision=data.get("tokenizer_revision"),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
            local_files_only=bool(data.get("local_files_only", False)),
            max_new_tokens=int(data.get("max_new_tokens", 200)),
            temperature=float(data.get("temperature", 1.0)),
            top_p=float(data["top_p"]) if data.get("top_p") is not None else None,
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "SynthIDReferenceConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.tokenizer not in ("simple", "huggingface"):
            raise DetectorConfigurationError(
                "SynthID reference detector supports tokenizer='simple' or 'huggingface'"
            )
        if self.ngram_len < 2:
            raise DetectorConfigurationError("SynthID ngram_len must be at least 2")
        if not self.keys:
            raise DetectorConfigurationError("SynthID keys must not be empty")
        if self.weights is not None and len(self.weights) != len(self.keys):
            raise DetectorConfigurationError(
                "SynthID weights must match number of keys"
            )
        if self.tokenizer == "simple" and len(self.vocabulary) < 2:
            raise DetectorConfigurationError(
                "SynthID vocabulary must contain at least two tokens"
            )
        if self.min_usable_tokens < 1:
            raise DetectorConfigurationError(
                "SynthID min_usable_tokens must be at least 1")
        if not 0 <= self.threshold <= 1:
            raise DetectorConfigurationError(
                "SynthID weighted-mean threshold must be in [0, 1]"
            )
        if self.tokenizer == "huggingface":
            if not self.model_identifier:
                raise DetectorConfigurationError(
                    "HuggingFace tokenizer requires model_identifier"
                )
            if not self.tokenizer_identifier:
                raise DetectorConfigurationError(
                    "HuggingFace tokenizer requires tokenizer_identifier"
                )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class SynthIDReferenceDetector(WatermarkDetector):
    """Reference-backed SynthID-Text weighted-mean detector.

    Follows the Google DeepMind public reference implementation algorithm
    (https://github.com/google-deepmind/synthid-text).  Hashing, g-value
    derivation, context repetition masking, and weighted-mean scoring are
    reimplemented in pure Python to match the reference behaviour without
    requiring JAX or PyTorch at detection time.
    """

    name = "synthid-reference"
    version = "1.0.0-reference"

    def __init__(self, config: SynthIDReferenceConfig, *, tokenizer: ProvenanceTokenizer | None = None):
        self.config = config
        self._hash_iv = _hash_iv_from_keys(config.keys)
        self._weights = (
            list(config.weights)
            if config.weights is not None
            else _default_weights(len(config.keys))
        )
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif config.tokenizer == "huggingface":
            from provenance.tokenizers import HuggingFaceTokenizer
            self.tokenizer = HuggingFaceTokenizer(
                tokenizer_identifier=config.tokenizer_identifier,
                model_identifier=config.model_identifier,
                revision=config.tokenizer_revision,
                local_files_only=config.local_files_only,
            )
        else:
            self.tokenizer = SimpleVocabularyTokenizer(
                vocabulary=config.vocabulary,
                tokenizer_identifier=f"{config.configuration_id}:simple-regex",
                case_sensitive=config.case_sensitive,
                namespace="synthid-reference-token",
            )

    @classmethod
    def from_config_file(cls, path: str | Path) -> "SynthIDReferenceDetector":
        config = SynthIDReferenceConfig.from_file(path)
        config.validate()
        return cls(config)

    def score_token_ids(self, token_ids: list[int]) -> DetectionResult:
        """Score a list of token IDs directly (no text decode/encode round-trip).

        This is the authoritative path for model-backed experiments where the
        exact token IDs from generation are available.
        """
        g_values = _compute_g_values(
            token_ids,
            ngram_len=self.config.ngram_len,
            hash_iv=self._hash_iv,
            keys=self.config.keys,
        )
        usable_mask = _compute_context_repetition_mask(
            token_ids,
            ngram_len=self.config.ngram_len,
            hash_iv=self._hash_iv,
            context_history_size=self.config.context_history_size,
        )
        usable_count = sum(usable_mask)

        if usable_count < self.config.min_usable_tokens:
            return DetectionResult(
                detector=self.name,
                detector_version=self.version,
                status="insufficient_text",
                detected=None,
                implementation_kind="reference",
                compatibility="synthid-reference",
                score=None,
                threshold=self.config.threshold,
                confidence="insufficient_evidence",
                evidence={
                    "token_count": len(token_ids),
                    "usable_token_count": usable_count,
                    "configuration_id": self.config.configuration_id,
                    "configuration_version": self.config.version,
                    "hash_key_id": self.config.hash_key_id,
                },
                text_requirements={
                    "minimum_usable_tokens": self.config.min_usable_tokens,
                    "actual_usable_tokens": usable_count,
                    "ngram_len": self.config.ngram_len,
                },
                limitations=SYNTHID_REFERENCE_LIMITATIONS,
            )

        score = _weighted_mean_score(
            g_values,
            usable_mask,
            self._weights,
            watermarking_depth=len(self.config.keys),
        )
        detected = score >= self.config.threshold
        confidence = self._confidence(score, detected)

        return DetectionResult(
            detector=self.name,
            detector_version=self.version,
            status="ok",
            detected=detected,
            implementation_kind="reference",
            compatibility="synthid-reference",
            score=score,
            threshold=self.config.threshold,
            confidence=confidence,
            evidence={
                "raw_score": score,
                "token_count": len(token_ids),
                "usable_token_count": usable_count,
                "g_values": g_values,
                "watermarking_depth": len(self.config.keys),
                "scorer": "weighted_mean",
                "configuration_id": self.config.configuration_id,
                "configuration_version": self.config.version,
                "hash_key_id": self.config.hash_key_id,
                "implementation_kind": "reference",
                "compatibility": "synthid-reference",
            },
            text_requirements={
                "minimum_usable_tokens": self.config.min_usable_tokens,
                "actual_usable_tokens": usable_count,
                "ngram_len": self.config.ngram_len,
                "tokenizer": self.config.tokenizer,
            },
            limitations=SYNTHID_REFERENCE_LIMITATIONS,
            metadata={
                "sampling_table_size": self.config.sampling_table_size,
                "sampling_table_seed": self.config.sampling_table_seed,
                "context_history_size": self.config.context_history_size,
                "weights": self._weights,
                "tokenizer_identifier": self.tokenizer.identifier,
                "hash_key_id": self.config.hash_key_id,
            },
        )

    def detect(self, text: str) -> DetectionResult:
        """Analyze text and return structured evidence."""
        token_ids = self.tokenizer.encode(text)
        return self.score_token_ids(token_ids)

    def _confidence(self, score: float, detected: bool) -> str:
        margin = abs(score - self.config.threshold)
        if detected and margin >= 0.15:
            return "high"
        if detected:
            return "medium"
        if margin < 0.05:
            return "insufficient_evidence"
        return "low"
