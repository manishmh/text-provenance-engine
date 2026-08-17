"""Engine orchestration for Phase 1 provenance-signal analysis."""

from __future__ import annotations

from provenance.detectors import UnicodeArtifactDetector, WatermarkDetector
from provenance.preprocessing import basic_text_statistics, validate_text
from provenance.schemas import AnalysisResult, DetectionResult

ENGINE_VERSION = "0.1.0"


class ProvenanceEngine:
    """Run Unicode and selected watermark detectors over a text string."""

    def __init__(self, *, include_unicode: bool = True):
        self.include_unicode = include_unicode

    def analyze(
        self,
        text: str,
        detectors: list[WatermarkDetector] | None = None,
    ) -> AnalysisResult:
        validate_text(text)
        selected_detectors = list(detectors or [])
        results: list[DetectionResult] = []

        if self.include_unicode:
            results.append(UnicodeArtifactDetector().detect(text))

        for detector in selected_detectors:
            try:
                results.append(detector.detect(text))
            except Exception as exc:  # pragma: no cover - defensive boundary
                results.append(
                    DetectionResult(
                        detector=getattr(detector, "name", detector.__class__.__name__),
                        detector_version=getattr(detector, "version", "unknown"),
                        status="error",
                        detected=None,
                        confidence="unavailable",
                        evidence={"error": str(exc)},
                        limitations=["Detector raised an exception; no signal conclusion is available."],
                    )
                )

        return AnalysisResult(
            engine_version=ENGINE_VERSION,
            status="ok",
            text_stats=basic_text_statistics(text),
            results=results,
            limitations=[
                "Phase 1 reports detector-specific provenance signals only.",
                "The engine does not classify text as AI-generated or human-authored.",
                "Watermark detectors require matching known generation configurations.",
            ],
            metadata={"requested_watermark_detectors": [detector.name for detector in selected_detectors]},
        )
