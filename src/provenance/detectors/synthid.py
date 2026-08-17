"""Backward-compatible import path for the controlled SynthID simulation.

Reference-backed SynthID detection is intentionally not implemented in this
milestone.
"""

from provenance.detectors.simulated.synthid import (
    SynthIDConfig,
    SynthIDTextDetector,
    generate_controlled_synthid_text,
)

__all__ = ["SynthIDConfig", "SynthIDTextDetector", "generate_controlled_synthid_text"]
