"""Common detector abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

from provenance.schemas import DetectionResult


class DetectorConfigurationError(ValueError):
    """Raised when a detector configuration is missing or unsupported."""


class WatermarkDetector(ABC):
    """Base interface for provenance signal detectors.

    The interface is intentionally not specific to KGW, SynthID, or any future
    watermarking algorithm.
    """

    name: str
    version: str

    @abstractmethod
    def detect(self, text: str) -> DetectionResult:
        """Analyze text and return structured evidence."""
