"""Backward-compatible import path for the controlled KGW simulation.

Use `provenance.detectors.simulated.kgw` for the simulation explicitly, or
`provenance.detectors.reference.kgw` for the model-token KGW research detector.
"""

from provenance.detectors.simulated.kgw import KGWConfig, KGWDetector, generate_controlled_kgw_text

__all__ = ["KGWConfig", "KGWDetector", "generate_controlled_kgw_text"]
