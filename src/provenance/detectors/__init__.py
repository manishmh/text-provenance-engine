"""Detector implementations.

All heavy imports (torch, transformers) are deferred so that the API
module can load without them.  The torch-free detectors (unicode, kgw)
are still imported eagerly; reference detectors that require torch are
loaded on first access.
"""

from __future__ import annotations

from provenance.detectors.base import DetectorConfigurationError, WatermarkDetector
from provenance.detectors.unicode import UnicodeArtifactDetector

# Eagerly import the torch-free detectors
from provenance.detectors.kgw import KGWConfig, KGWDetector
from provenance.detectors.synthid import SynthIDConfig, SynthIDTextDetector

# Reference detectors that require torch — lazily imported
_LAZY_REFERENCE = {
    "KGWMarkLLMConfig",
    "KGWMarkLLMDetector",
    "KGWMarkLLMScorer",
    "KGWReferenceConfig",
    "KGWReferenceDetector",
    "KGWReferenceScorer",
    "SynthIDReferenceConfig",
    "SynthIDReferenceDetector",
}

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


def __getattr__(name: str):  # pragma: no cover
    if name in _LAZY_REFERENCE:
        from provenance.detectors import reference as _ref

        return getattr(_ref, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
