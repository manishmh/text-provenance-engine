"""Centralized detector registry with capability metadata.

All supported detectors are registered here. Selection is centralized rather
than scattered across CLI/API code. Each detector exposes machine-readable
capability metadata for API/dashboard consumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from provenance.detectors.base import WatermarkDetector


@dataclass(frozen=True)
class DetectorCapability:
    """Machine-readable capability description for a detector."""

    name: str
    display_name: str
    implementation_kind: str  # "unicode", "watermark-kgw", "watermark-synthid"
    compatibility: str  # "any", "gpt-2", "gemini", etc.
    requires_config: bool
    # supports_generation means: this detector name is accepted by the
    # reference sample-generation pipeline (provenance.robustness.execution).
    # Generation support alone does NOT imply a full benchmark run
    # (generate + evaluate) completes for that name: completing a run also
    # requires the detector to evaluate HF-generated text, which the
    # simulation KGW/SynthID implementations cannot do (verbatim-config
    # token space). The names that complete full runs are enumerated in
    # provenance.api.benchmarks.RUNNABLE_DETECTORS, not by this flag.
    supports_generation: bool
    supports_benchmarking: bool
    tokenizer_requirements: str | None  # e.g. "huggingface", None for any
    known_limitations: list[str] = field(default_factory=list)
    description: str = ""
    # Public-product classification is deliberately separate from whether a
    # detector exists or supports benchmarks. A detector can be useful for a
    # known experiment while being invalid for arbitrary pasted text.
    public_classification: str = "reference_config_specific"
    public_availability: str = "unavailable_without_key_or_config"
    compute_class: str = "cheap_configured"
    public_reason: str = "Requires a matching watermark configuration."

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "implementation_kind": self.implementation_kind,
            "compatibility": self.compatibility,
            "requires_config": self.requires_config,
            "supports_generation": self.supports_generation,
            "supports_benchmarking": self.supports_benchmarking,
            "tokenizer_requirements": self.tokenizer_requirements,
            "known_limitations": list(self.known_limitations),
            "description": self.description,
            "public_classification": self.public_classification,
            "public_availability": self.public_availability,
            "compute_class": self.compute_class,
            "public_reason": self.public_reason,
        }


@dataclass(frozen=True)
class DetectorEntry:
    """Internal registry entry linking metadata to factory."""

    capability: DetectorCapability
    factory: Callable[..., WatermarkDetector]


class DetectorRegistry:
    """Registry of all supported detectors."""

    def __init__(self) -> None:
        self._entries: dict[str, DetectorEntry] = {}

    def register(self, capability: DetectorCapability, factory: Callable[..., WatermarkDetector]) -> None:
        self._entries[capability.name] = DetectorEntry(capability=capability, factory=factory)

    def get(self, name: str) -> DetectorEntry | None:
        return self._entries.get(name)

    def create(self, name: str, **kwargs: Any) -> WatermarkDetector:
        entry = self._entries.get(name)
        if entry is None:
            raise ValueError(f"Unknown detector: {name!r}. Available: {self.names()}")
        return entry.factory(**kwargs)

    def names(self) -> list[str]:
        return list(self._entries.keys())

    def capabilities(self) -> list[DetectorCapability]:
        return [e.capability for e in self._entries.values()]

    def capability_dict(self, name: str) -> dict[str, Any] | None:
        entry = self._entries.get(name)
        return entry.capability.to_dict() if entry else None

    def all_capability_dicts(self) -> list[dict[str, Any]]:
        return [e.capability.to_dict() for e in self._entries.values()]


# ── Global registry singleton ──────────────────────────────────────────

_registry: DetectorRegistry | None = None


def get_registry() -> DetectorRegistry:
    """Get or build the global detector registry."""
    global _registry
    if _registry is not None:
        return _registry

    _registry = DetectorRegistry()

    # Register Unicode detector (no config needed)
    from provenance.detectors.unicode import UnicodeArtifactDetector

    _registry.register(
        DetectorCapability(
            name="unicode",
            display_name="Unicode Artifact Detection",
            implementation_kind="unicode",
            compatibility="any",
            requires_config=False,
            supports_generation=False,
            supports_benchmarking=True,
            tokenizer_requirements=None,
            known_limitations=[
                "Only detects deterministic Unicode artifacts",
                "Does not detect AI-generated text generally",
                "Sensitive to whitespace/normalization changes",
            ],
            description=(
                "Detects Unicode artifacts such as invisible characters, "
                "non-standard whitespace, and encoding anomalies."
            ),
            public_classification="publicly_usable_arbitrary_input",
            public_availability="available",
            compute_class="cheap_deterministic",
            public_reason=(
                "Works on arbitrary text without a watermark key, model, or "
                "generation-time configuration."
            ),
        ),
        lambda **kw: UnicodeArtifactDetector(),
    )

    # Register KGW detector (requires config)
    from provenance.detectors.kgw import KGWConfig, KGWDetector

    _registry.register(
        DetectorCapability(
            name="kgw",
            display_name="KGW Watermark Detection",
            implementation_kind="watermark-kgw",
            compatibility="gpt-2",
            requires_config=True,
            supports_generation=True,
            supports_benchmarking=True,
            tokenizer_requirements="huggingface",
            known_limitations=[
                "Requires knowledge of the watermark key",
                "Only works with GPT-2-compatible tokenization",
                "Sensitive to text edits that alter token boundaries",
            ],
            description=(
                "Kirchenbauer-Geiping-Wen (KGW) watermark detection. "
                "Requires a watermark configuration to test against."
            ),
            public_classification="benchmark_only",
            public_availability="not_applicable",
            compute_class="cheap_configured",
            public_reason=(
                "Controlled-token simulation for tests and benchmarks; it is "
                "not a universal detector for third-party text."
            ),
        ),
        lambda config_path, **kw: KGWDetector.from_config_file(config_path),
    )

    # Register KGW-Reference detector (requires config + torch)
    def _kgw_ref_factory(config_path: str, **kw: Any) -> WatermarkDetector:
        from provenance.detectors.reference.kgw import KGWReferenceDetector
        return KGWReferenceDetector.from_config_file(config_path)

    _registry.register(
        DetectorCapability(
            name="kgw-reference",
            display_name="KGW Reference Detection",
            implementation_kind="watermark-kgw",
            compatibility="gpt-2",
            requires_config=True,
            supports_generation=True,
            supports_benchmarking=True,
            tokenizer_requirements="huggingface",
            known_limitations=[
                "Requires torch and transformers",
                "Requires knowledge of the watermark key",
                "Uses model tokenizer instead of simulation",
            ],
            description=(
                "Reference KGW detection using the actual model tokenizer."
            ),
            public_classification="reference_config_specific",
            public_availability="unavailable_without_key_or_config",
            compute_class="potentially_model_backed",
            public_reason=(
                "Reference verification requires the matching KGW key, "
                "parameters, variant, and tokenizer configuration."
            ),
        ),
        _kgw_ref_factory,
    )

    # Register SynthID detector (requires config)
    from provenance.detectors.synthid import SynthIDConfig, SynthIDTextDetector

    _registry.register(
        DetectorCapability(
            name="synthid",
            display_name="SynthID Watermark Detection",
            implementation_kind="watermark-synthid",
            compatibility="gemini",
            requires_config=True,
            supports_generation=True,
            supports_benchmarking=True,
            tokenizer_requirements="huggingface",
            known_limitations=[
                "Requires the watermark key and configuration",
                "Only works with Gemini-compatible token space",
                "Synthesizer parameters must match the original",
            ],
            description=(
                "DeepMind SynthID watermark detection. "
                "Tests text against a known SynthID watermark configuration."
            ),
            public_classification="benchmark_only",
            public_availability="not_applicable",
            compute_class="cheap_configured",
            public_reason=(
                "Controlled-token simulation for tests and benchmarks; it is "
                "not a universal detector for third-party text."
            ),
        ),
        lambda config_path, **kw: SynthIDTextDetector.from_config_file(config_path),
    )

    # Register SynthID-Reference detector (requires config + torch)
    def _synthid_ref_factory(config_path: str, **kw: Any) -> WatermarkDetector:
        from provenance.detectors.reference.synthid import SynthIDReferenceDetector
        return SynthIDReferenceDetector.from_config_file(config_path)

    _registry.register(
        DetectorCapability(
            name="synthid-reference",
            display_name="SynthID Reference Detection",
            implementation_kind="watermark-synthid",
            compatibility="gemini",
            requires_config=True,
            supports_generation=True,
            supports_benchmarking=True,
            tokenizer_requirements="huggingface",
            known_limitations=[
                "Requires torch and transformers",
                "Cannot verify without the actual watermark key",
                "May produce false positives on non-watermarked text",
            ],
            description=(
                "Reference SynthID detection using the actual model tokenizer."
            ),
            public_classification="reference_config_specific",
            public_availability="unavailable_without_key_or_config",
            compute_class="potentially_model_backed",
            public_reason=(
                "Reference verification requires the matching SynthID key, "
                "parameters, and tokenizer configuration."
            ),
        ),
        _synthid_ref_factory,
    )

    return _registry
