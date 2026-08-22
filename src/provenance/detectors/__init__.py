"""Detector implementations."""

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.detectors.kgw import KGWConfig, KGWDetector
from provenance.detectors.reference import (
    KGWMarkLLMConfig,
    KGWMarkLLMDetector,
    KGWMarkLLMScorer,
    KGWReferenceConfig,
    KGWReferenceDetector,
    KGWReferenceScorer,
    SynthIDReferenceConfig,
    SynthIDReferenceDetector,
)
from provenance.detectors.synthid import SynthIDConfig, SynthIDTextDetector
from provenance.detectors.unicode import UnicodeArtifactDetector

__all__ = [
    "DetectorConfigurationError",
    "KGWConfig",
    "KGWDetector",
    "KGWMarkLLMConfig",
    "KGWMarkLLMDetector",
    "KGWMarkLLMScorer",
    "KGWReferenceConfig",
    "KGWReferenceDetector",
    "KGWReferenceScorer",
    "SynthIDConfig",
    "SynthIDTextDetector",
    "SynthIDReferenceConfig",
    "SynthIDReferenceDetector",
    "UnicodeArtifactDetector",
    "WatermarkDetector",
]
