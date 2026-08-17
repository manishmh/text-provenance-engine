"""Controlled simulation detectors."""

from provenance.detectors.simulated.kgw import KGWConfig, KGWDetector, generate_controlled_kgw_text
from provenance.detectors.simulated.synthid import (
    SynthIDConfig,
    SynthIDTextDetector,
    generate_controlled_synthid_text,
)

__all__ = [
    "KGWConfig",
    "KGWDetector",
    "SynthIDConfig",
    "SynthIDTextDetector",
    "generate_controlled_kgw_text",
    "generate_controlled_synthid_text",
]
