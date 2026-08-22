"""Backward-compatible import path for the controlled SynthID simulation.

Reference-backed SynthID detection is provided in
``provenance.detectors.reference.synthid``.
"""

from provenance.detectors.reference.synthid import (
    SynthIDReferenceConfig,
    SynthIDReferenceDetector,
)
from provenance.detectors.simulated.synthid import (
    SynthIDConfig,
    SynthIDTextDetector,
    generate_controlled_synthid_text,
)

__all__ = ["SynthIDConfig", "SynthIDTextDetector", "SynthIDReferenceConfig", "SynthIDReferenceDetector", "generate_controlled_synthid_text"]
