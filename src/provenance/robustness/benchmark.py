"""Benchmark result schema, aggregation, confidence intervals, and robustness matrix.

Phase 5C layer for comparing watermark robustness across detectors, models,
text lengths, and transformations.  Designed for future API/dashboard
consumption without rewriting core logic.

No raw text is stored in any benchmark result record.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ── Schema version ─────────────────────────────────────────────────────
ROBUSTNESS_SCHEMA_VERSION = "provenance-robustness-v1"

# ── Wilson confidence interval (reused from calibration module) ─────────

Z_95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns ``(low, high)`` clamped to ``[0, 1]``.
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
    """Observed rate with Wilson confidence interval."""

    successes: int
    n_samples: int
    rate: float
    ci_low: float
    ci_high: float
    confidence: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rate_estimate(
    successes: int, n: int, *, z: float = Z_95, confidence: float = 0.95
) -> RateEstimate:
    """Observed rate ``successes/n`` plus its Wilson interval."""
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


# ── Benchmark result schema ────────────────────────────────────────────


@dataclass(frozen=True)
class RobustnessBenchmarkResult:
    """Versioned, machine-readable benchmark result for one (detector, config, length, transform) cell."""

    schema_version: str
    detector_name: str
    config_identifier: str
    transform_name: str
    text_length: int | None
    sample_count: int
    seed: int
    baseline_detected: int
    transformed_detected: int
    detection_change_count: int
    mean_baseline_score: float | None
    mean_transformed_score: float | None
    mean_score_delta: float | None
    # Wilson CI for the robustness rate
    robustness_rate: float
    robustness_ci_low: float
    robustness_ci_high: float
    baseline_rate: float
    baseline_ci_low: float
    baseline_ci_high: float
    transformed_rate: float
    transformed_ci_low: float
    transformed_ci_high: float
    limitations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["limitations"] = list(d["limitations"])
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustnessBenchmarkResult:
        data = dict(data)  # shallow copy
        data["limitations"] = tuple(data.get("limitations", ()))
        data["metadata"] = data.get("metadata", {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[arg-type]

    @property
    def detection_change_rate(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.detection_change_count / self.sample_count

    @property
    def detection_preserved_count(self) -> int:
        return self.sample_count - self.detection_change_count


def build_benchmark_result(
    *,
    detector_name: str,
    config_identifier: str,
    transform_name: str,
    text_length: int | None,
    records: Sequence[dict[str, Any]],
    seed: int = 0,
    limitations: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> RobustnessBenchmarkResult:
    """Build a ``RobustnessBenchmarkResult`` from a list of record dicts.

    Each record dict must contain:
    ``original_detected``, ``transformed_detected``, ``original_score``,
    ``transformed_score``, ``score_delta``, ``detection_changed``.

    For ``identity`` transform, ``transformed_*`` fields are the same as ``original_*``.
    """
    n = len(records)
    if n == 0:
        return RobustnessBenchmarkResult(
            schema_version=ROBUSTNESS_SCHEMA_VERSION,
            detector_name=detector_name,
            config_identifier=config_identifier,
            transform_name=transform_name,
            text_length=text_length,
            sample_count=0,
            seed=seed,
            baseline_detected=0,
            transformed_detected=0,
            detection_change_count=0,
            mean_baseline_score=None,
            mean_transformed_score=None,
            mean_score_delta=None,
            robustness_rate=0.0,
            robustness_ci_low=0.0,
            robustness_ci_high=0.0,
            baseline_rate=0.0,
            baseline_ci_low=0.0,
            baseline_ci_high=0.0,
            transformed_rate=0.0,
            transformed_ci_low=0.0,
            transformed_ci_high=0.0,
            limitations=limitations,
            metadata=dict(metadata) if metadata else {},
        )

    baseline_detected = sum(1 for r in records if r.get("original_detected") is True)
    transformed_detected = sum(1 for r in records if r.get("transformed_detected") is True)
    detection_change_count = sum(1 for r in records if r.get("detection_changed") is True)

    baseline_scores = [r["original_score"] for r in records if r.get("original_score") is not None]
    transformed_scores = [r["transformed_score"] for r in records if r.get("transformed_score") is not None]
    deltas = [r["score_delta"] for r in records if r.get("score_delta") is not None]

    mean_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
    mean_transformed = sum(transformed_scores) / len(transformed_scores) if transformed_scores else None
    mean_delta = sum(deltas) / len(deltas) if deltas else None

    baseline_re = rate_estimate(baseline_detected, n)
    transformed_re = rate_estimate(transformed_detected, n)

    # Robustness rate: paired conditional rate P(transformed detected |
    # baseline detected). Each record carries both outcomes, so restrict to
    # baseline-positive records and count how many are still detected after
    # transformation. This keeps the rate in [0, 1] and the Wilson interval
    # valid (the aggregate-ratio approximation could exceed 1 and crash).
    baseline_positive = [r for r in records if r.get("original_detected") is True]
    if baseline_positive:
        still_detected = sum(1 for r in baseline_positive if r.get("transformed_detected") is True)
        robustness = rate_estimate(still_detected, len(baseline_positive))
    else:
        robustness = rate_estimate(0, n)

    return RobustnessBenchmarkResult(
        schema_version=ROBUSTNESS_SCHEMA_VERSION,
        detector_name=detector_name,
        config_identifier=config_identifier,
        transform_name=transform_name,
        text_length=text_length,
        sample_count=n,
        seed=seed,
        baseline_detected=baseline_detected,
        transformed_detected=transformed_detected,
        detection_change_count=detection_change_count,
        mean_baseline_score=mean_baseline,
        mean_transformed_score=mean_transformed,
        mean_score_delta=mean_delta,
        robustness_rate=robustness.rate,
        robustness_ci_low=robustness.ci_low,
        robustness_ci_high=robustness.ci_high,
        baseline_rate=baseline_re.rate,
        baseline_ci_low=baseline_re.ci_low,
        baseline_ci_high=baseline_re.ci_high,
        transformed_rate=transformed_re.rate,
        transformed_ci_low=transformed_re.ci_low,
        transformed_ci_high=transformed_re.ci_high,
        limitations=limitations,
        metadata=dict(metadata) if metadata else {},
    )


# ── Persistence ────────────────────────────────────────────────────────


def write_benchmark_results(
    results: Iterable[RobustnessBenchmarkResult], path: str | Path
) -> Path:
    """Write benchmark results as JSONL (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.to_dict(), sort_keys=True))
            f.write("\n")
    return path


def read_benchmark_results(path: str | Path) -> list[RobustnessBenchmarkResult]:
    """Load benchmark results from JSONL."""
    results: list[RobustnessBenchmarkResult] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(RobustnessBenchmarkResult.from_dict(json.loads(line)))
    return results


def load_results_from_directory(directory: str | Path) -> list[RobustnessBenchmarkResult]:
    """Load all ``*.jsonl`` benchmark results from a directory (recursively)."""
    directory = Path(directory)
    results: list[RobustnessBenchmarkResult] = []
    for p in sorted(directory.rglob("*.jsonl")):
        results.extend(read_benchmark_results(p))
    return results


# ── Cross-experiment aggregation ───────────────────────────────────────


@dataclass(frozen=True)
class AggregatedResult:
    """Aggregated robustness result across multiple experiments with the same grouping key."""

    detector_name: str
    config_identifier: str
    transform_name: str
    total_samples: int
    total_baseline_detected: int
    total_transformed_detected: int
    total_detection_changes: int
    mean_baseline_score: float | None
    mean_transformed_score: float | None
    mean_score_delta: float | None
    robustness_rate: float
    robustness_ci_low: float
    robustness_ci_high: float
    baseline_rate: float
    baseline_ci_low: float
    baseline_ci_high: float
    transformed_rate: float
    transformed_ci_low: float
    transformed_ci_high: float
    experiment_count: int
    text_lengths: tuple[int | None, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text_lengths"] = list(d["text_lengths"])
        d["seeds"] = list(d["seeds"])
        return d


def aggregate_results(
    results: Sequence[RobustnessBenchmarkResult],
) -> list[AggregatedResult]:
    """Group results by (detector, config, transform) and aggregate."""
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[RobustnessBenchmarkResult]] = defaultdict(list)
    for r in results:
        key = (r.detector_name, r.config_identifier, r.transform_name)
        groups[key].append(r)

    aggregated: list[AggregatedResult] = []
    for (detector, config, transform), group in sorted(groups.items()):
        total_n = sum(r.sample_count for r in group)
        total_baseline = sum(r.baseline_detected for r in group)
        total_transformed = sum(r.transformed_detected for r in group)
        total_changes = sum(r.detection_change_count for r in group)

        # Weighted mean of scores
        all_baseline_scores = []
        all_transformed_scores = []
        all_deltas = []
        for r in group:
            if r.mean_baseline_score is not None and r.sample_count > 0:
                all_baseline_scores.extend([r.mean_baseline_score] * r.sample_count)
            if r.mean_transformed_score is not None and r.sample_count > 0:
                all_transformed_scores.extend([r.mean_transformed_score] * r.sample_count)
            if r.mean_score_delta is not None and r.sample_count > 0:
                all_deltas.extend([r.mean_score_delta] * r.sample_count)

        mb = sum(all_baseline_scores) / len(all_baseline_scores) if all_baseline_scores else None
        mt = sum(all_transformed_scores) / len(all_transformed_scores) if all_transformed_scores else None
        md = sum(all_deltas) / len(all_deltas) if all_deltas else None

        baseline_re = rate_estimate(total_baseline, total_n) if total_n > 0 else rate_estimate(0, 0)
        transformed_re = rate_estimate(total_transformed, total_n) if total_n > 0 else rate_estimate(0, 0)
        # Pairing is lost at aggregation time (only counts survive), so the
        # aggregate ratio can exceed 1 when a transform flips a baseline
        # miss into a hit. Clamp to keep a valid rate and Wilson interval;
        # the baseline/transformed rates still show the full picture.
        if total_baseline > 0:
            robustness_re = rate_estimate(min(total_transformed, total_baseline), total_baseline)
        else:
            robustness_re = rate_estimate(0, total_n) if total_n > 0 else rate_estimate(0, 0)

        aggregated.append(AggregatedResult(
            detector_name=detector,
            config_identifier=config,
            transform_name=transform,
            total_samples=total_n,
            total_baseline_detected=total_baseline,
            total_transformed_detected=total_transformed,
            total_detection_changes=total_changes,
            mean_baseline_score=mb,
            mean_transformed_score=mt,
            mean_score_delta=md,
            robustness_rate=robustness_re.rate,
            robustness_ci_low=robustness_re.ci_low,
            robustness_ci_high=robustness_re.ci_high,
            baseline_rate=baseline_re.rate,
            baseline_ci_low=baseline_re.ci_low,
            baseline_ci_high=baseline_re.ci_high,
            transformed_rate=transformed_re.rate,
            transformed_ci_low=transformed_re.ci_low,
            transformed_ci_high=transformed_re.ci_high,
            experiment_count=len(group),
            text_lengths=tuple(sorted({r.text_length for r in group})),
            seeds=tuple(sorted({r.seed for r in group})),
        ))

    return aggregated


# ── Robustness matrix ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RobustnessMatrix:
    """Machine-readable robustness matrix for dashboard visualization."""

    detectors: list[str]
    transforms: list[str]
    cells: dict[str, dict[str, Any]]  # keyed by "detector|transform"
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROBUSTNESS_SCHEMA_VERSION,
            "detectors": self.detectors,
            "transforms": self.transforms,
            "cells": dict(self.cells),
            "limitations": list(self.limitations),
        }

    def get(self, detector: str, transform: str) -> dict[str, Any] | None:
        return self.cells.get(f"{detector}|{transform}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobustnessMatrix:
        return cls(
            detectors=data["detectors"],
            transforms=data["transforms"],
            cells=data["cells"],
            limitations=tuple(data.get("limitations", ())),
        )


def build_robustness_matrix(
    aggregated: Sequence[AggregatedResult],
    *,
    limitations: tuple[str, ...] = (),
) -> RobustnessMatrix:
    """Build a robustness matrix from aggregated results.

    The matrix has detectors as rows and transforms as columns.
    Each cell contains robustness rate, CI, and other metrics.
    """
    detectors = sorted({r.detector_name for r in aggregated})
    transforms = sorted({r.transform_name for r in aggregated})
    cells: dict[str, dict[str, Any]] = {}

    for r in aggregated:
        key = f"{r.detector_name}|{r.transform_name}"
        cells[key] = {
            "detector": r.detector_name,
            "transform": r.transform_name,
            "robustness_rate": r.robustness_rate,
            "robustness_ci_low": r.robustness_ci_low,
            "robustness_ci_high": r.robustness_ci_high,
            "baseline_rate": r.baseline_rate,
            "transformed_rate": r.transformed_rate,
            "mean_score_delta": r.mean_score_delta,
            "total_samples": r.total_samples,
            "total_detection_changes": r.total_detection_changes,
        }

    return RobustnessMatrix(
        detectors=detectors,
        transforms=transforms,
        cells=cells,
        limitations=limitations,
    )


# ── Human-readable report ──────────────────────────────────────────────

REPORT_LIMITATIONS = [
    "This benchmark evaluates robustness of implemented detectors under tested deterministic transformations.",
    "It does not establish resistance to adversarial attacks, watermark removal, paraphrasing, or generic AI-text detection.",
    "Results are specific to the tested configurations and transformations.",
    "A high robustness score does not mean the watermark is resistant to removal.",
    "Short texts carry less watermark signal, so detection rates at small token lengths are expected to be lower.",
    "TPR/FPR and robustness rates are estimated from finite samples; the standard error scales roughly as 1/sqrt(n_samples).",
]


def render_robustness_matrix_text(matrix: RobustnessMatrix) -> str:
    """Render a human-readable robustness matrix table."""
    lines = ["Robustness Matrix", "=" * 60]
    lines.append(
        f"  {'Detector/Model':<25} "
        + " ".join(f"{t:>20}" for t in matrix.transforms)
    )
    lines.append("  " + "-" * (25 + 21 * len(matrix.transforms)))

    for det in matrix.detectors:
        parts = [f"  {det:<25}"]
        for t in matrix.transforms:
            cell = matrix.get(det, t)
            if cell is None:
                parts.append(f"{'n/a':>20}")
            else:
                rate = cell["robustness_rate"]
                ci_lo = cell["robustness_ci_low"]
                ci_hi = cell["robustness_ci_high"]
                parts.append(f"{rate:>8.2%} [{ci_lo:.2f},{ci_hi:.2f}]")
        lines.append(" ".join(parts))

    lines.append("")
    lines.append("Limitations")
    lines.append("-" * 40)
    for lim in (matrix.limitations or REPORT_LIMITATIONS):
        lines.append(f"  - {lim}")
    return "\n".join(lines)


def render_benchmark_report_text(
    results: Sequence[RobustnessBenchmarkResult],
    *,
    matrix: RobustnessMatrix | None = None,
    aggregated: Sequence[AggregatedResult] | None = None,
    category_map: dict[str, str] | None = None,
) -> str:
    """Render a comprehensive human-readable benchmark report.

    Args:
        results: Benchmark results.
        matrix: Optional robustness matrix.
        aggregated: Optional pre-computed aggregation.
        category_map: Optional mapping of transform_name -> category.
            Used for category-level aggregation.
    """
    lines = [
        "Phase 5C Watermark Robustness Benchmark Report",
        "=" * 55,
        f"Schema version: {ROBUSTNESS_SCHEMA_VERSION}",
        f"Total results: {len(results)}",
        "",
    ]

    # Group by detector
    detectors = sorted({r.detector_name for r in results})
    for det in detectors:
        det_results = [r for r in results if r.detector_name == det]
        lines.append(f"Detector: {det}")
        lines.append("-" * 40)

        # Config identifiers
        configs = sorted({r.config_identifier for r in det_results})
        for cfg in configs:
            cfg_results = [r for r in det_results if r.config_identifier == cfg]
            lines.append(f"  Config: {cfg}")
            lines.append(f"  Results: {len(cfg_results)}")

            # Per-transform summary
            transforms = sorted({r.transform_name for r in cfg_results})
            for t in transforms:
                t_results = [r for r in cfg_results if r.transform_name == t]
                total_n = sum(r.sample_count for r in t_results)
                total_bl = sum(r.baseline_detected for r in t_results)
                total_tr = sum(r.transformed_detected for r in t_results)
                bl_rate = total_bl / total_n if total_n > 0 else 0.0
                tr_rate = total_tr / total_n if total_n > 0 else 0.0
                rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
                rob_ci = wilson_interval(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)

                lines.append(
                    f"    {t:<25} "
                    f"baseline={bl_rate:>5.1%} "
                    f"transformed={tr_rate:>5.1%} "
                    f"robustness={rob_rate:>5.1%} "
                    f"[{rob_ci[0]:.2f},{rob_ci[1]:.2f}] "
                    f"n={total_n}"
                )
            lines.append("")

        # Category-level aggregation if category_map provided
        if category_map:
            lines.append("  Category Aggregation")
            lines.append("  " + "-" * 38)
            lines.append(f"    {'Category':<25} {'Robustness':>10} {'Score Delta':>12} {'N':>6}")
            cat_groups: dict[str, list[RobustnessBenchmarkResult]] = {}
            for r in cfg_results:
                cat = category_map.get(r.transform_name, "unknown")
                cat_groups.setdefault(cat, []).append(r)
            for cat in sorted(cat_groups.keys()):
                cat_results = cat_groups[cat]
                total_n = sum(r.sample_count for r in cat_results)
                total_bl = sum(r.baseline_detected for r in cat_results)
                total_tr = sum(r.transformed_detected for r in cat_results)
                rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
                deltas = [r.mean_score_delta for r in cat_results if r.mean_score_delta is not None]
                mean_delta = sum(deltas) / len(deltas) if deltas else None
                delta_str = f"{mean_delta:+.4f}" if mean_delta is not None else "n/a"
                lines.append(f"    {cat:<25} {rob_rate:>10.1%} {delta_str:>12} {total_n:>6}")
            lines.append("")

    # Robustness matrix
    if matrix is not None:
        lines.append(render_robustness_matrix_text(matrix))
        lines.append("")

    # Limitations
    lines.append("Limitations")
    lines.append("-" * 40)
    for lim in REPORT_LIMITATIONS:
        lines.append(f"  - {lim}")

    return "\n".join(lines)


# ── Programmatic report (JSON-serializable) ───────────────────────────


def build_programmatic_report(
    results: Sequence[RobustnessBenchmarkResult],
    *,
    aggregated: Sequence[AggregatedResult] | None = None,
    matrix: RobustnessMatrix | None = None,
    category_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable report with stable field names.

    Args:
        results: Benchmark results.
        aggregated: Optional pre-computed aggregation.
        matrix: Optional robustness matrix.
        category_map: Optional mapping of transform_name -> category.
    """
    if aggregated is None:
        aggregated = aggregate_results(results)

    detectors = sorted({r.detector_name for r in results})
    transforms = sorted({r.transform_name for r in results})

    # Build category aggregation if category_map provided
    category_aggregation = None
    if category_map:
        from collections import defaultdict
        cat_groups: dict[tuple[str, str, str], list[RobustnessBenchmarkResult]] = defaultdict(list)
        for r in results:
            cat = category_map.get(r.transform_name, "unknown")
            key = (r.detector_name, r.config_identifier, cat)
            cat_groups[key].append(r)

        category_agg = []
        for (det, cfg, cat), group in sorted(cat_groups.items()):
            total_n = sum(r.sample_count for r in group)
            total_bl = sum(r.baseline_detected for r in group)
            total_tr = sum(r.transformed_detected for r in group)
            deltas = [r.mean_score_delta for r in group if r.mean_score_delta is not None]
            mean_delta = sum(deltas) / len(deltas) if deltas else None
            rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
            rob_ci = wilson_interval(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)
            category_agg.append({
                "detector_name": det,
                "config_identifier": cfg,
                "category": cat,
                "total_samples": total_n,
                "robustness_rate": rob_rate,
                "robustness_ci_low": rob_ci[0],
                "robustness_ci_high": rob_ci[1],
                "mean_score_delta": mean_delta,
                "result_count": len(group),
            })
        category_aggregation = category_agg

    report: dict[str, Any] = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "total_results": len(results),
        "detectors": detectors,
        "transforms": transforms,
        "results": [r.to_dict() for r in results],
        "aggregated": [a.to_dict() for a in aggregated],
        "matrix": matrix.to_dict() if matrix else None,
        "limitations": list(REPORT_LIMITATIONS),
    }
    if category_aggregation is not None:
        report["category_aggregation"] = category_aggregation
    return report
