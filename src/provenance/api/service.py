"""Analysis service layer for the provenance HTTP API.

Encapsulates the analysis business logic so it can be used by both
the synchronous and asynchronous HTTP routes.  Independent of FastAPI.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from provenance.engine import ENGINE_VERSION

logger = logging.getLogger("provenance.api")


def validate_detectors(detectors: list[str] | None) -> tuple[list[str], bool]:
    """Validate detector names and determine unicode inclusion.

    Returns (detector_names, include_unicode).
    """
    from provenance.api.models import _VALID_DETECTORS, _WATERMARK_DETECTORS

    names = detectors or ["unicode"]
    unknown = set(names) - _VALID_DETECTORS
    if unknown:
        raise ValueError(
            f"Unknown detector(s): {', '.join(sorted(unknown))}. "
            f"Valid options: {', '.join(sorted(_VALID_DETECTORS))}"
        )
    needs_config = set(names) & _WATERMARK_DETECTORS
    return names, "unicode" in names, needs_config


def execute_analysis(
    text: str,
    detector_names: list[str],
    include_unicode: bool,
    config_path: str | None,
    needs_config: set[str],
) -> dict[str, Any]:
    """Execute the ProvenanceEngine and return the result dict.

    Raises ValueError if watermark detectors require a config_path.
    """
    from provenance.api.routes import _build_detectors
    from provenance.engine import ProvenanceEngine

    if needs_config and config_path is None:
        raise ValueError(
            f"config_path is required when using watermark detector(s): "
            f"{', '.join(sorted(needs_config))}"
        )

    detectors, extra_results = _build_detectors(detector_names, config_path)
    engine = ProvenanceEngine(include_unicode=include_unicode)
    result = engine.analyze(text, detectors=detectors)

    if extra_results:
        result = type(result)(
            engine_version=result.engine_version,
            status=result.status,
            text_stats=result.text_stats,
            results=result.results + extra_results,
            limitations=result.limitations,
            metadata=result.metadata,
        )

    return result.to_dict()


def run_analysis(
    text: str,
    detector_names: list[str],
    config_path: str | None,
) -> dict[str, Any]:
    """High-level analysis: validate, execute, return result dict.

    This is the single entry point used by both sync and async paths.
    """
    names, include_unicode, needs_config = validate_detectors(detector_names)
    return execute_analysis(text, names, include_unicode, config_path, needs_config)
