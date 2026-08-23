"""Backward-compatible import path for the controlled SynthID simulation.

Reference-backed SynthID detection is provided in
``provenance.detectors.reference.synthid``.
"""

from __future__ import annotations

from provenance.detectors.simulated.synthid import (
    SynthIDConfig,
    SynthIDTextDetector,
    generate_controlled_synthid_text,
)

__all__ = ["SynthIDConfig", "SynthIDTextDetector", "SynthIDReferenceConfig", "SynthIDReferenceDetector", "generate_controlled_synthid_text"]


# Lazy: reference classes require torch/transformers
def __getattr__(name: str):  # pragma: no cover
    if name in ("SynthIDReferenceConfig", "SynthIDReferenceDetector"):
        from provenance.detectors.reference.synthid import (
            SynthIDReferenceConfig,
            SynthIDReferenceDetector,
        )
        _map = {
            "SynthIDReferenceConfig": SynthIDReferenceConfig,
            "SynthIDReferenceDetector": SynthIDReferenceDetector,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
