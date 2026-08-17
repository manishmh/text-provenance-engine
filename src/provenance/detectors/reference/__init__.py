"""Reference-backed detector implementations."""

from provenance.detectors.reference.kgw import (
    KGWReferenceConfig,
    KGWReferenceDetector,
    KGWReferenceScorer,
)

__all__ = ["KGWReferenceConfig", "KGWReferenceDetector", "KGWReferenceScorer"]
