"""Detection statistics for KGW benchmark records.

Pure functions over :class:`~provenance.benchmark.records.EvaluationRecord`
lists. No model / torch dependency, so this layer is fully unit-testable with
small synthetic fixtures.

Definitions
-----------
- **TPR (true-positive / detection rate)**: fraction of *watermarked* samples
  whose z-score meets a threshold. Higher is better.
- **FPR (false-positive rate)**: fraction of *unwatermarked* samples whose
  z-score meets a threshold. Lower is better. Because unwatermarked z-scores are
  ~N(0, 1) under the null, the FPR at threshold ``t`` should approximate the
  one-sided normal tail probability.

Watermarked and unwatermarked samples are always summarized separately; TPR is
only ever computed over watermarked samples and FPR only over unwatermarked
ones.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from provenance.benchmark.records import EvaluationRecord, UNWATERMARKED, WATERMARKED


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two values."""
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


@dataclass(frozen=True)
class GroupStatistics:
    """Aggregated statistics for one condition, optionally at one token length."""

    condition: str  # "watermarked" | "unwatermarked"
    target_length: int | None  # None -> aggregated across all lengths
    n_samples: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    detection_rate: float  # fraction detected at each record's own threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdPoint:
    """TPR and FPR at one candidate z-threshold, optionally at one length."""

    threshold: float
    target_length: int | None
    tpr: float
    fpr: float
    n_watermarked: int
    n_unwatermarked: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filter(
    records: Sequence[EvaluationRecord],
    *,
    watermarked: bool | None = None,
    target_length: int | None = None,
) -> list[EvaluationRecord]:
    result = list(records)
    if watermarked is not None:
        result = [r for r in result if r.watermarked == watermarked]
    if target_length is not None:
        result = [r for r in result if r.target_length == target_length]
    return result


def summarize_group(
    records: Sequence[EvaluationRecord],
    *,
    watermarked: bool,
    target_length: int | None = None,
) -> GroupStatistics:
    """Summarize one condition (watermarked/unwatermarked) at an optional length."""
    subset = _filter(records, watermarked=watermarked, target_length=target_length)
    condition = WATERMARKED if watermarked else UNWATERMARKED
    if not subset:
        return GroupStatistics(
            condition=condition,
            target_length=target_length,
            n_samples=0,
            mean_score=0.0,
            std_score=0.0,
            min_score=0.0,
            max_score=0.0,
            detection_rate=0.0,
        )
    scores = [r.score for r in subset]
    detected = sum(1 for r in subset if r.detected)
    return GroupStatistics(
        condition=condition,
        target_length=target_length,
        n_samples=len(subset),
        mean_score=_mean(scores),
        std_score=_stdev(scores),
        min_score=min(scores),
        max_score=max(scores),
        detection_rate=detected / len(subset),
    )


def true_positive_rate(records: Sequence[EvaluationRecord], threshold: float) -> float:
    """Detection rate over *watermarked* samples at ``threshold`` (score >= t)."""
    watermarked = _filter(records, watermarked=True)
    if not watermarked:
        return 0.0
    hits = sum(1 for r in watermarked if r.score >= threshold)
    return hits / len(watermarked)


def false_positive_rate(records: Sequence[EvaluationRecord], threshold: float) -> float:
    """Detection rate over *unwatermarked* samples at ``threshold`` (score >= t)."""
    unwatermarked = _filter(records, watermarked=False)
    if not unwatermarked:
        return 0.0
    hits = sum(1 for r in unwatermarked if r.score >= threshold)
    return hits / len(unwatermarked)


def threshold_analysis(
    records: Sequence[EvaluationRecord],
    thresholds: Sequence[float],
    *,
    target_length: int | None = None,
) -> list[ThresholdPoint]:
    """TPR/FPR at each candidate threshold, optionally restricted to one length."""
    subset = _filter(records, target_length=target_length)
    n_watermarked = len(_filter(subset, watermarked=True))
    n_unwatermarked = len(_filter(subset, watermarked=False))
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        points.append(
            ThresholdPoint(
                threshold=float(threshold),
                target_length=target_length,
                tpr=true_positive_rate(subset, threshold),
                fpr=false_positive_rate(subset, threshold),
                n_watermarked=n_watermarked,
                n_unwatermarked=n_unwatermarked,
            )
        )
    return points


def observed_lengths(records: Sequence[EvaluationRecord]) -> list[int]:
    """Sorted unique target lengths present in ``records``."""
    return sorted({r.target_length for r in records})


@dataclass(frozen=True)
class LengthEffectRow:
    """One row of the length-vs-detection table."""

    target_length: int
    n_watermarked: int
    n_unwatermarked: int
    watermarked_detection_rate: float  # at the record threshold
    unwatermarked_fpr: float  # at the record threshold
    mean_score_watermarked: float
    mean_score_unwatermarked: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def length_effect(records: Sequence[EvaluationRecord]) -> list[LengthEffectRow]:
    """How detection changes with token length, at each record's own threshold."""
    rows: list[LengthEffectRow] = []
    for length in observed_lengths(records):
        wm = summarize_group(records, watermarked=True, target_length=length)
        un = summarize_group(records, watermarked=False, target_length=length)
        rows.append(
            LengthEffectRow(
                target_length=length,
                n_watermarked=wm.n_samples,
                n_unwatermarked=un.n_samples,
                watermarked_detection_rate=wm.detection_rate,
                unwatermarked_fpr=un.detection_rate,
                mean_score_watermarked=wm.mean_score,
                mean_score_unwatermarked=un.mean_score,
            )
        )
    return rows
