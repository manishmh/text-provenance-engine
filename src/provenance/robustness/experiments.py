"""Watermark robustness experiment pipeline.

Generates watermarked samples, applies transformations, and evaluates
detection robustness. Supports both KGW and SynthID detectors.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from provenance.robustness.evaluator import EvalRecord, _hash_text, compute_statistics, evaluate_robustness
from provenance.robustness.transforms import Transform, get_all_transforms


class DetectorProtocol(Protocol):
    """Minimal protocol for detectors used in robustness experiments."""
    name: str
    def detect(self, text: str) -> Any: ...


@dataclass(frozen=True)
class WatermarkSample:
    """A generated watermarked sample with its metadata."""
    text: str
    token_count: int
    sample_seed: int
    prompt_id: str
    length: int
    watermarked: bool


@dataclass(frozen=True)
class RobustnessRecord:
    """Single robustness evaluation: original → transform → detect."""
    detector_name: str
    transform_name: str
    implementation_kind: str
    compatibility: str
    original_hash: str
    transformed_hash: str
    original_score: float | None
    transformed_score: float | None
    original_detected: bool | None
    transformed_detected: bool | None
    original_length: int
    transformed_length: int
    score_delta: float | None
    detection_changed: bool
    detection_preserved: bool
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "transform_name": self.transform_name,
            "implementation_kind": self.implementation_kind,
            "compatibility": self.compatibility,
            "original_hash": self.original_hash,
            "transformed_hash": self.transformed_hash,
            "original_score": self.original_score,
            "transformed_score": self.transformed_score,
            "original_detected": self.original_detected,
            "transformed_detected": self.transformed_detected,
            "original_length": self.original_length,
            "transformed_length": self.transformed_length,
            "score_delta": self.score_delta,
            "detection_changed": self.detection_changed,
            "detection_preserved": self.detection_preserved,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a robustness experiment."""
    detector_name: str
    config_path: str | None
    lengths: tuple[int, ...]
    samples_per_length: int
    seed: int
    transforms: list[Transform] | None = None  # None = all transforms

    @classmethod
    def from_cli_args(
        cls,
        detector_name: str,
        config_path: str | None,
        lengths: str | None,
        samples: int,
        seed: int,
        transform_names: str | None = None,
    ) -> "ExperimentConfig":
        if lengths:
            parsed = tuple(int(x.strip()) for x in lengths.split(",") if x.strip())
        else:
            parsed = (50, 100)

        transforms = None
        if transform_names:
            from provenance.robustness.transforms import get_transform
            transforms = [get_transform(n.strip()) for n in transform_names.split(",") if n.strip()]

        return cls(
            detector_name=detector_name,
            config_path=config_path,
            lengths=parsed,
            samples_per_length=samples,
            seed=seed,
            transforms=transforms,
        )


@dataclass(frozen=True)
class RobustnessReport:
    """Aggregated robustness experiment report."""
    detector_name: str
    config_path: str | None
    sample_count: int
    text_lengths: tuple[int, ...]
    seed: int
    baseline_detection_rate: float
    transformed_detection_rate: float
    robustness_rate: float
    mean_baseline_score: float | None
    mean_transformed_score: float | None
    mean_score_delta: float | None
    per_transform: dict[str, dict[str, Any]]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector_name,
            "config": self.config_path,
            "sample_count": self.sample_count,
            "text_lengths": list(self.text_lengths),
            "seed": self.seed,
            "baseline_detection_rate": self.baseline_detection_rate,
            "transformed_detection_rate": self.transformed_detection_rate,
            "robustness_rate": self.robustness_rate,
            "mean_baseline_score": self.mean_baseline_score,
            "mean_transformed_score": self.mean_transformed_score,
            "mean_score_delta": self.mean_score_delta,
            "per_transform": self.per_transform,
            "limitations": self.limitations,
        }


ROBUSTNESS_LIMITATIONS = [
    "This evaluates robustness of the implemented detector under the tested transformations.",
    "It does not establish resistance to adversarial attacks, watermark removal, paraphrasing, or generic AI-text detection.",
    "Results are specific to the tested configuration, model, and transformations.",
    "Short texts carry less watermark signal, so detection rates at small token lengths are expected to be lower.",
    "No universally-correct threshold exists; the results are reported as measured.",
    "A positive result is evidence for the configured watermark only.",
]


def _detect_score(detector: Any, text: str) -> tuple[float | None, bool | None]:
    """Extract score and detected from a DetectionResult."""
    result = detector.detect(text)
    return result.score, result.detected


def _detect_metadata(detector: Any, text: str) -> tuple[float | None, bool | None, str, str]:
    """Extract score, detected, implementation_kind, compatibility."""
    result = detector.detect(text)
    return result.score, result.detected, result.implementation_kind, result.compatibility


def evaluate_robustness_experiment(
    detector: Any,
    samples: list[WatermarkSample],
    transforms: list[Transform],
) -> list[RobustnessRecord]:
    """Run robustness evaluation across all samples and transforms.

    Pipeline: for each sample → detect original → for each transform →
    transform → detect transformed → record comparison.

    Model is loaded once; generation is done once per sample.
    """
    records: list[RobustnessRecord] = []

    for sample in samples:
        # Detect original
        t0 = time.perf_counter()
        orig_score, orig_detected, impl_kind, compat = _detect_metadata(detector, sample.text)
        orig_duration = (time.perf_counter() - t0) * 1000

        for transform in transforms:
            t_start = time.perf_counter()
            transform_result = transform.apply(sample.text)
            trans_score, trans_detected = _detect_score(detector, transform_result.text)
            duration_ms = (time.perf_counter() - t_start) * 1000

            # Compute deltas
            score_delta = None
            if orig_score is not None and trans_score is not None:
                score_delta = trans_score - orig_score

            orig_det = orig_detected is True
            trans_det = trans_detected is True
            detection_changed = orig_det != trans_det
            detection_preserved = orig_det == trans_det

            records.append(RobustnessRecord(
                detector_name=detector.name,
                transform_name=transform.name,
                implementation_kind=impl_kind,
                compatibility=compat,
                original_hash=_hash_text(sample.text),
                transformed_hash=_hash_text(transform_result.text),
                original_score=orig_score,
                transformed_score=trans_score,
                original_detected=orig_detected,
                transformed_detected=trans_detected,
                original_length=len(sample.text),
                transformed_length=len(transform_result.text),
                score_delta=score_delta,
                detection_changed=detection_changed,
                detection_preserved=detection_preserved,
                duration_ms=duration_ms,
                metadata={
                    "sample_seed": sample.sample_seed,
                    "prompt_id": sample.prompt_id,
                    "length": sample.length,
                    "watermarked": sample.watermarked,
                },
            ))

    return records


def compute_robustness_report(
    records: list[RobustnessRecord],
    config: ExperimentConfig,
) -> RobustnessReport:
    """Compute aggregated robustness metrics from experiment records.

    Metrics:
    - baseline_detection_rate: fraction of identity-transform samples detected
    - transformed_detection_rate: fraction of non-identity samples detected
    - robustness_rate: fraction of baseline-detected samples still detected
      after transformation, restricted to baseline-positive samples and their
      non-identity transform evaluations (always in [0, 1]).
    - mean baseline/transformed scores and delta
    """
    if not records:
        return RobustnessReport(
            detector_name=config.detector_name,
            config_path=config.config_path,
            sample_count=0,
            text_lengths=config.lengths,
            seed=config.seed,
            baseline_detection_rate=0.0,
            transformed_detection_rate=0.0,
            robustness_rate=0.0,
            mean_baseline_score=None,
            mean_transformed_score=None,
            mean_score_delta=None,
            per_transform={},
            limitations=ROBUSTNESS_LIMITATIONS,
        )

    # Baseline (identity) records
    baseline_records = [r for r in records if r.transform_name == "identity"]
    non_baseline = [r for r in records if r.transform_name != "identity"]

    # Baseline detection rate
    baseline_detected = sum(1 for r in baseline_records if r.original_detected is True)
    baseline_total = len(baseline_records)
    baseline_detection_rate = baseline_detected / baseline_total if baseline_total > 0 else 0.0

    # Baseline scores
    baseline_scores = [r.original_score for r in baseline_records if r.original_score is not None]
    mean_baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else None

    # Transformed detection rate (from non-identity transforms)
    transformed_detected = sum(1 for r in non_baseline if r.transformed_detected is True)
    transformed_total = len(non_baseline)
    transformed_detection_rate = transformed_detected / transformed_total if transformed_total > 0 else 0.0

    # Mean transformed scores
    trans_scores = [r.transformed_score for r in non_baseline if r.transformed_score is not None]
    mean_transformed_score = sum(trans_scores) / len(trans_scores) if trans_scores else None

    # Robustness rate: paired conditional rate P(transformed detected |
    # baseline detected). Restrict to non-identity evaluations whose sample
    # was detected at baseline, then take the fraction still detected.
    # Both numerator and denominator count evaluations (not samples), so the
    # result is always in [0, 1].
    baseline_positive_keys = {
        (b.metadata.get("sample_seed"), b.metadata.get("length"))
        for b in baseline_records if b.original_detected is True
    }
    matched = [
        r for r in non_baseline
        if (r.metadata.get("sample_seed"), r.metadata.get("length")) in baseline_positive_keys
    ]
    if matched:
        robust_positive = sum(1 for r in matched if r.transformed_detected is True)
        robustness_rate = robust_positive / len(matched)
    else:
        robustness_rate = 0.0

    # Mean score delta
    deltas = [r.score_delta for r in non_baseline if r.score_delta is not None]
    mean_score_delta = sum(deltas) / len(deltas) if deltas else None

    # Per-transform breakdown
    by_transform: dict[str, list[RobustnessRecord]] = {}
    for r in records:
        by_transform.setdefault(r.transform_name, []).append(r)

    per_transform = {}
    for tname, recs in by_transform.items():
        if tname == "identity":
            det_count = sum(1 for r in recs if r.original_detected is True)
            scores = [r.original_score for r in recs if r.original_score is not None]
            per_transform[tname] = {
                "detection_rate": det_count / len(recs) if recs else 0.0,
                "detected_count": det_count,
                "total": len(recs),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "score_change": 0.0,
            }
        else:
            det_count = sum(1 for r in recs if r.transformed_detected is True)
            scores = [r.transformed_score for r in recs if r.transformed_score is not None]
            deltas = [r.score_delta for r in recs if r.score_delta is not None]
            per_transform[tname] = {
                "detection_rate": det_count / len(recs) if recs else 0.0,
                "detected_count": det_count,
                "total": len(recs),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "mean_score_delta": sum(deltas) / len(deltas) if deltas else None,
            }

    return RobustnessReport(
        detector_name=config.detector_name,
        config_path=config.config_path,
        sample_count=baseline_total,
        text_lengths=config.lengths,
        seed=config.seed,
        baseline_detection_rate=baseline_detection_rate,
        transformed_detection_rate=transformed_detection_rate,
        robustness_rate=robustness_rate,
        mean_baseline_score=mean_baseline_score,
        mean_transformed_score=mean_transformed_score,
        mean_score_delta=mean_score_delta,
        per_transform=per_transform,
        limitations=ROBUSTNESS_LIMITATIONS,
    )


def render_robustness_report(report: RobustnessReport) -> str:
    """Render a human-readable robustness report."""
    lines = [
        "Watermark Robustness Report",
        "=" * 40,
        f"Detector: {report.detector_name}",
        f"Config: {report.config_path or 'default'}",
        f"Samples: {report.sample_count} per length",
        f"Lengths: {list(report.text_lengths)}",
        f"Seed: {report.seed}",
        "",
        "Results",
        "-" * 40,
        f"  Baseline detection rate:    {report.baseline_detection_rate:.2%}",
        f"  Transformed detection rate: {report.transformed_detection_rate:.2%}",
        f"  Robustness rate:            {report.robustness_rate:.2%}",
        f"  Mean baseline score:        {report.mean_baseline_score:.4f}" if report.mean_baseline_score is not None else "  Mean baseline score:        N/A",
        f"  Mean transformed score:     {report.mean_transformed_score:.4f}" if report.mean_transformed_score is not None else "  Mean transformed score:     N/A",
        f"  Mean score delta:           {report.mean_score_delta:+.4f}" if report.mean_score_delta is not None else "  Mean score delta:           N/A",
        "",
        "Per-Transformation",
        "-" * 40,
        f"  {'Transformation':<30} {'Baseline DR*':>12} {'Transformed DR':>15} {'Robustness':>12}",
    ]
    lines.append(
        "  *Baseline DR is the overall identity-transform rate (same for all rows)."
    )

    baseline_dr = report.per_transform.get("identity", {}).get("detection_rate", 0.0)
    for tname, stats in sorted(report.per_transform.items()):
        if tname == "identity":
            continue
        t_dr = stats["detection_rate"]
        # Compute per-transform robustness if possible
        robust = t_dr / baseline_dr if baseline_dr > 0 else 0.0
        lines.append(f"  {tname:<30} {baseline_dr:>12.2%} {t_dr:>15.2%} {robust:>12.2%}")

    lines.append("")
    lines.append("Limitations")
    lines.append("-" * 40)
    for lim in report.limitations:
        lines.append(f"  - {lim}")

    return "\n".join(lines)
