"""Robustness evaluator for testing detector resilience against text transformations.

Pipeline: original text → transformation → detector → score/detection result
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

from provenance.detectors.base import WatermarkDetector
from provenance.robustness.transforms import Transform, TransformResult, get_all_transforms


@dataclass(frozen=True)
class EvalRecord:
    """Single evaluation record: detector + transformation → result."""

    detector_name: str
    transform_name: str
    original_hash: str
    transformed_hash: str
    score: float | None
    detected: bool | None
    text_length: int
    transformed_length: int
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "transform_name": self.transform_name,
            "original_hash": self.original_hash,
            "transformed_hash": self.transformed_hash,
            "score": self.score,
            "detected": self.detected,
            "text_length": self.text_length,
            "transformed_length": self.transformed_length,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


def _hash_text(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evaluate_single(
    text: str,
    detector: WatermarkDetector,
    transform: Transform,
) -> EvalRecord:
    """Run one detector on one transformed version of text."""
    t0 = time.perf_counter()
    result = transform.apply(text)
    detection = detector.detect(result.text)
    duration_ms = (time.perf_counter() - t0) * 1000

    return EvalRecord(
        detector_name=detector.name,
        transform_name=transform.name,
        original_hash=_hash_text(text),
        transformed_hash=_hash_text(result.text),
        score=detection.score,
        detected=detection.detected,
        text_length=len(text),
        transformed_length=len(result.text),
        duration_ms=duration_ms,
        metadata={
            "detector_version": detection.detector_version,
            "status": detection.status,
            "confidence": detection.confidence,
            "implementation_kind": detection.implementation_kind,
        },
    )


def evaluate_robustness(
    text: str,
    detector: WatermarkDetector,
    transforms: list[Transform] | None = None,
) -> list[EvalRecord]:
    """Evaluate detector against multiple transformations.

    Returns one EvalRecord per (detector, transform) pair.
    Raw text is never persisted — only SHA-256 hashes are stored.
    """
    if transforms is None:
        transforms = get_all_transforms()

    records = []
    for transform in transforms:
        record = evaluate_single(text, detector, transform)
        records.append(record)
    return records


def compute_statistics(records: list[EvalRecord]) -> dict[str, Any]:
    """Compute summary statistics from evaluation records.

    Returns detection rates, score changes, and per-transform breakdowns.
    """
    if not records:
        return {"total_transforms": 0, "transforms": {}}

    # Group by transform
    by_transform: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_transform.setdefault(r.transform_name, []).append(r)

    # Baseline (identity) record
    baseline = None
    for r in records:
        if r.transform_name == "identity":
            baseline = r
            break

    transform_stats = {}
    for tname, recs in by_transform.items():
        detected_count = sum(1 for r in recs if r.detected is True)
        total = len(recs)
        scores = [r.score for r in recs if r.score is not None]
        avg_score = sum(scores) / len(scores) if scores else None

        transform_stats[tname] = {
            "total_evaluations": total,
            "detected_count": detected_count,
            "detection_rate": detected_count / total if total > 0 else 0.0,
            "avg_score": avg_score,
            "avg_duration_ms": sum(r.duration_ms for r in recs) / total if total > 0 else 0.0,
        }

    # Score change relative to baseline
    baseline_score = baseline.score if baseline and baseline.score is not None else None
    if baseline_score is not None:
        for tname, stats in transform_stats.items():
            if stats["avg_score"] is not None:
                stats["score_change"] = stats["avg_score"] - baseline_score
            else:
                stats["score_change"] = None

    # Overall detection rate
    total_detected = sum(1 for r in records if r.detected is True)
    total_evals = len(records)

    return {
        "total_transforms": len(by_transform),
        "total_evaluations": total_evals,
        "baseline_detected": baseline.detected if baseline else None,
        "baseline_score": baseline_score,
        "overall_detection_rate": total_detected / total_evals if total_evals > 0 else 0.0,
        "transforms": transform_stats,
    }
