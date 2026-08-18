"""Reference-backed detector implementations."""

from provenance.detectors.reference.kgw import (
    KGWReferenceConfig,
    KGWReferenceDetector,
    KGWReferenceScorer,
)
from provenance.detectors.reference.kgw_markllm import (
    KGWMarkLLMConfig,
    KGWMarkLLMDetector,
    KGWMarkLLMScorer,
)

__all__ = [
    "KGWReferenceConfig",
    "KGWReferenceDetector",
    "KGWReferenceScorer",
    "KGWMarkLLMConfig",
    "KGWMarkLLMDetector",
    "KGWMarkLLMScorer",
]
