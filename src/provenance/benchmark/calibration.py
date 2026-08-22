"""Statistical calibration for KGW benchmark records.

Pure functions over :class:`~provenance.benchmark.records.EvaluationRecord`
lists -- no model / torch dependency, so this layer is unit-testable with small
synthetic fixtures.

Two families of statistic live here:

- **Wilson score confidence intervals** for an observed rate (TPR or FPR). An
  observed FPR of 0/20 is *not* proof the true FPR is zero; the Wilson interval
  makes the residual uncertainty explicit (e.g. 0/20 -> [0.000, 0.161] at 95%).
- **ROC curve / AUC** summarizing separability of watermarked vs unwatermarked
  z-scores across *all* thresholds, independent of any single operating point.

Nothing here classifies text as AI vs human; it only quantifies how well a
*known* KGW configuration separates its own watermarked from unwatermarked
samples.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from provenance.benchmark.records import EvaluationRecord
from provenance.benchmark.statistics import ThresholdPoint

# 95% two-sided normal quantile. Exposed so callers can request other levels.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(low, high)`` clamped to ``[0, 1]``. For ``n == 0`` the interval is
    undefined and reported as ``(0.0, 0.0)``. The Wilson interval is preferred
    over the normal (Wald) interval because it stays inside ``[0, 1]`` and behaves
    sensibly at the boundaries (0 or n successes), which is exactly where FPR/TPR
    estimates from small samples tend to land.
    """
    if n <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes ({successes}) must be in [0, {n}]")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


@dataclass(frozen=True)
class RateEstimate:
    """An observed rate with its Wilson confidence interval and sample count."""

    successes: int
    n_samples: int
    rate: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rate_estimate(successes: int, n: int, *, z: float = Z_95, confidence: float = 0.95) -> RateEstimate:
    """Observed rate ``successes/n`` plus its Wilson interval (0.0 rate if n==0)."""
    rate = successes / n if n > 0 else 0.0
    low, high = wilson_interval(successes, n, z)
    return RateEstimate(
        successes=successes,
        n_samples=n,
        rate=rate,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
    )


def calibrated_threshold(point: ThresholdPoint, *, z: float = Z_95) -> dict[str, Any]:
    """Enrich a :class:`ThresholdPoint` with Wilson intervals and success counts.

    Reuses the rates already computed by ``threshold_analysis`` (so the counting
    logic is not duplicated) and recovers the integer success counts from the
    exact rate * n products.
    """
    tp = round(point.tpr * point.n_watermarked)
    fp = round(point.fpr * point.n_unwatermarked)
    tpr = rate_estimate(tp, point.n_watermarked, z=z)
    fpr = rate_estimate(fp, point.n_unwatermarked, z=z)
    return {
        "threshold": point.threshold,
        "target_length": point.target_length,
        # observed rates (kept under the original keys for backwards compat)
        "tpr": tpr.rate,
        "fpr": fpr.rate,
        "n_watermarked": point.n_watermarked,
        "n_unwatermarked": point.n_unwatermarked,
        # calibration additions
        "tpr_successes": tp,
        "tpr_ci_low": tpr.ci_low,
        "tpr_ci_high": tpr.ci_high,
        "fpr_successes": fp,
        "fpr_ci_low": fpr.ci_low,
        "fpr_ci_high": fpr.ci_high,
        "confidence": tpr.confidence,
    }


@dataclass(frozen=True)
class ROCPoint:
    """One point on the ROC curve. ``threshold=None`` is the point above all scores."""

    threshold: float | None
    fpr: float
    tpr: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_scores(
    records: Sequence[EvaluationRecord], target_length: int | None
) -> tuple[list[float], list[float]]:
    wm: list[float] = []
    un: list[float] = []
    for r in records:
        if target_length is not None and r.target_length != target_length:
            continue
        (wm if r.watermarked else un).append(r.score)
    return wm, un


def roc_auc(records: Sequence[EvaluationRecord], *, target_length: int | None = None) -> float | None:
    """AUC = P(watermarked z > unwatermarked z) + 0.5 * P(tie).

    Equivalent to the normalized Mann-Whitney U statistic and to the area under
    the ROC curve. Returns ``None`` when either condition has no samples (AUC is
    undefined). 1.0 = perfect separation, 0.5 = chance.
    """
    wm, un = _split_scores(records, target_length)
    if not wm or not un:
        return None
    greater = 0.0
    for a in wm:
        for b in un:
            if a > b:
                greater += 1.0
            elif a == b:
                greater += 0.5
    return greater / (len(wm) * len(un))


def roc_curve(records: Sequence[EvaluationRecord], *, target_length: int | None = None) -> list[ROCPoint]:
    """ROC curve as (fpr, tpr) points swept over the observed z-scores.

    The first point (threshold ``None``, above every score) is ``(0, 0)``; each
    subsequent threshold is one observed z-score in descending order, ending at
    the minimum score with ``(1, 1)``. Empty when either condition is missing.
    """
    wm, un = _split_scores(records, target_length)
    if not wm or not un:
        return []
    thresholds: list[float | None] = [None]
    thresholds.extend(sorted({*wm, *un}, reverse=True))
    n_wm = len(wm)
    n_un = len(un)
    points: list[ROCPoint] = []
    for t in thresholds:
        if t is None:
            tpr = 0.0
            fpr = 0.0
        else:
            tpr = sum(1 for s in wm if s >= t) / n_wm
            fpr = sum(1 for s in un if s >= t) / n_un
        points.append(ROCPoint(threshold=t, fpr=fpr, tpr=tpr))
    return points


def roc_summary(
    records: Sequence[EvaluationRecord], *, target_length: int | None = None
) -> dict[str, Any]:
    """AUC plus the ROC curve points and per-condition sample counts."""
    wm, un = _split_scores(records, target_length)
    return {
        "target_length": target_length,
        "auc": roc_auc(records, target_length=target_length),
        "n_watermarked": len(wm),
        "n_unwatermarked": len(un),
        "points": [p.to_dict() for p in roc_curve(records, target_length=target_length)],
    }
