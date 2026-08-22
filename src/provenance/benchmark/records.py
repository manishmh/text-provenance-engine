"""Structured evaluation records for benchmark experiments.

An :class:`EvaluationRecord` captures one generated-and-scored sample. It is the
atomic unit the statistics and reporting layers aggregate over. The record is
deliberately flat and JSON-serializable so it can be stored as JSONL and later
consumed by an API or dashboard without importing any heavyweight dependency.

Scientific boundary
-------------------
A positive result means "evidence consistent with the tested watermark
configuration", not "this text is AI-generated". Records therefore carry the
ground-truth ``watermarked`` flag (known because we generated the sample) and the
measured statistics, and never a global AI/human classification.

Secret handling
--------------
Raw secret keys are never stored. Only ``hash_key_id`` (a public
identifier or a truncated hash of the key) is recorded.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# Bump when the record schema or benchmark methodology changes in a way that
# makes previously-written results incomparable.
BENCHMARK_VERSION = "provenance-benchmark-v1"

WATERMARKED = "watermarked"
UNWATERMARKED = "unwatermarked"


@dataclass(frozen=True)
class EvaluationRecord:
    """One generated-and-scored sample.

    Every field is either provenance metadata (how the sample was produced) or a
    measurement (what the detector reported). ``hash_key_id`` stands in for the
    secret key, which is never stored.
    """

    # --- identity / provenance (required) ----------------------------------
    experiment_id: str
    timestamp: str
    benchmark_version: str
    scheme: str
    variant: str
    configuration_id: str
    configuration_version: str
    implementation_kind: str
    compatibility: str

    # --- model / tokenizer (required) --------------------------------------
    model_identifier: str
    model_revision: str | None
    tokenizer_identifier: str
    tokenizer_revision: str | None
    vocab_size: int

    # --- watermark configuration (required, no raw secret key) -------------
    hash_key_id: str

    # --- generation parameters (required) ----------------------------------
    temperature: float
    top_p: float | None
    top_k: int | None
    random_seed: int
    prompt_id: str
    target_length: int

    # --- ground truth + measurement (required) -----------------------------
    watermarked: bool
    token_count: int
    scored_token_count: int
    detection_threshold: float
    detected: bool
    score: float

    # --- KGW-specific (optional, None for non-KGW schemes) ----------------
    gamma: float | None = None
    delta: float | None = None
    prefix_length: int | None = None
    window_scheme: str | None = None
    seeding_scheme: str | None = None
    f_scheme: str | None = None
    green_token_count: int | None = None
    green_fraction: float | None = None
    expected_green_fraction: float | None = None
    z_score: float | None = None
    p_value: float | None = None

    # --- SynthID-specific (optional, None for non-SynthID schemes) --------
    ngram_len: int | None = None
    watermarking_depth: int | None = None

    @property
    def condition(self) -> str:
        return WATERMARKED if self.watermarked else UNWATERMARKED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRecord":
        if "hash_key" in data:  # defensive: never accept a raw secret key
            raise ValueError("EvaluationRecord must not carry a raw hash_key")
        fields = list(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        fields_set = set(fields)
        unknown = set(data) - fields_set
        if unknown:
            raise ValueError(f"unknown EvaluationRecord fields: {sorted(unknown)}")
        # Build kwargs: fill defaults for optional fields not in data
        kwargs = {}
        for f in fields:
            if f in data:
                kwargs[f] = data[f]
            else:
                # Use dataclass default (None for Optional fields)
                default = cls.__dataclass_fields__[f].default  # type: ignore[attr-defined]
                if default is MISSING:
                    raise ValueError(f"missing required EvaluationRecord field: {f}")
                kwargs[f] = default
        return cls(**kwargs)


def write_jsonl(records: Iterable[EvaluationRecord], path: str | Path) -> Path:
    """Serialize records as one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return path


def read_jsonl(path: str | Path) -> list[EvaluationRecord]:
    """Load records written by :func:`write_jsonl`."""
    records: list[EvaluationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(EvaluationRecord.from_dict(json.loads(line)))
    return records


def iter_condition(records: Iterable[EvaluationRecord], watermarked: bool) -> Iterator[EvaluationRecord]:
    for record in records:
        if record.watermarked == watermarked:
            yield record
