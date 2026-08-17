"""Detector implementations."""

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.detectors.kgw import KGWConfig, KGWDetector
from provenance.detectors.reference import KGWReferenceConfig, KGWReferenceDetector, KGWReferenceScorer
from provenance.detectors.synthid import SynthIDConfig, SynthIDTextDetector
from provenance.detectors.unicode import UnicodeArtifactDetector

__all__ = [
    "DetectorConfigurationError",
    "KGWConfig",
    "KGWDetector",
    "KGWReferenceConfig",
    "KGWReferenceDetector",
    "KGWReferenceScorer",
    "SynthIDConfig",
    "SynthIDTextDetector",
    "UnicodeArtifactDetector",
    "WatermarkDetector",
]
