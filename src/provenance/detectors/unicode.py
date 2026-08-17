"""Detector wrapper for deterministic Unicode artifacts."""

from __future__ import annotations

from collections import Counter

from provenance.detectors.base import WatermarkDetector
from provenance.preprocessing.unicode import analyze_unicode
from provenance.schemas import DetectionResult


class UnicodeArtifactDetector(WatermarkDetector):
    name = "unicode"
    version = "1.0.0"

    def detect(self, text: str) -> DetectionResult:
        findings = analyze_unicode(text)
        by_category = Counter(f.category for f in findings)
        by_severity = Counter(f.severity for f in findings)
        confidence = "high" if by_severity.get("high") else "medium" if findings else "none"

        return DetectionResult(
            detector=self.name,
            detector_version=self.version,
            status="ok",
            detected=bool(findings),
            implementation_kind="adapter",
            compatibility="unavailable",
            score=len(findings),
            threshold=1,
            confidence=confidence,
            evidence={
                "finding_count": len(findings),
                "findings": [finding.to_dict() for finding in findings],
                "by_category": dict(by_category),
                "by_severity": dict(by_severity),
            },
            text_requirements={
                "minimum_characters": 1,
                "requires_original_unicode_text": True,
            },
            limitations=[
                "Reports deterministic Unicode/text-level artifacts only.",
                "A clean Unicode report does not imply human authorship.",
                "Some formatting characters can be legitimate in specific languages or emoji sequences.",
            ],
        )
