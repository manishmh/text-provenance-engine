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


def get_detector_capabilities() -> list[dict]:
    """Return detector capabilities from the registry."""
    from provenance.detectors.registry import get_registry
    return get_registry().all_capability_dicts()


# ---------------------------------------------------------------------------
# Robustness benchmark artifacts (Phase 6A: read-only dashboard support)
# ---------------------------------------------------------------------------
#
# These functions expose CLI-produced Phase 5 benchmark artifacts
# (``benchmark_results.jsonl`` files) through the API for dashboard
# visualization. All statistics are computed by the existing Phase 5
# reporting layer — nothing is recalculated here. The schemas contain no
# raw text, keys, or secrets, so the payloads are safe to serve as-is.

#: Environment variable selecting the directory tree scanned for
#: ``*.jsonl`` robustness benchmark results.
ROBUSTNESS_RESULTS_ENV_VAR = "PROVENANCE_ROBUSTNESS_DIR"

#: Default directory scanned when the env var is unset.
DEFAULT_ROBUSTNESS_DIR = "data/robustness"


def get_robustness_dir() -> str:
    """Return the configured robustness results directory."""
    import os
    return os.environ.get(ROBUSTNESS_RESULTS_ENV_VAR, DEFAULT_ROBUSTNESS_DIR)


def get_transform_category_map() -> dict[str, str]:
    """Map transform name -> display category for filtering/grouping.

    Phase 5D advanced transforms map to their taxonomy category; the ten
    Phase 5A baseline transforms map to ``"baseline"``; anything else maps
    to ``"other"``. Display grouping only — no statistics involved.
    """
    try:
        from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
        mapping = {name: t.category.value for name, t in ADVANCED_TRANSFORM_MAP.items()}
    except Exception:
        mapping = {}
    try:
        from provenance.robustness.transforms import ALL_TRANSFORMS
        for t in ALL_TRANSFORMS:
            mapping.setdefault(t.name, "baseline")
    except Exception:
        pass
    return mapping


def load_robustness_results() -> tuple[list, list[dict]]:
    """Load all benchmark results under the robustness directory.

    Returns ``(results, warnings)``. A malformed file produces a warning
    entry instead of failing the whole load, so the dashboard can show a
    degraded-but-useful view. A missing directory yields ``([], [])``.
    """
    from pathlib import Path

    from provenance.robustness.benchmark import read_benchmark_results

    directory = Path(get_robustness_dir())
    results: list = []
    warnings: list[dict] = []
    if not directory.exists():
        return results, warnings
    for path in sorted(directory.rglob("*.jsonl")):
        try:
            results.extend(read_benchmark_results(path))
        except Exception as exc:
            warnings.append({
                "file": str(path),
                "error_type": type(exc).__name__,
                "message": str(exc)[:300],
            })
    return results, warnings


def _apply_result_filters(
    results: list,
    *,
    detector: str | None = None,
    config: str | None = None,
    transform: str | None = None,
    text_length: int | None = None,
    run_id: str | None = None,
) -> list:
    """Filter benchmark results by detector/config/transform/length/run.

    ``run_id`` matches the ``run_id`` stamped into result metadata by
    API-triggered benchmark runs (Phase 6C). Files predating stamping
    simply never match.
    """
    filtered = results
    if detector is not None:
        filtered = [r for r in filtered if r.detector_name == detector]
    if config is not None:
        filtered = [r for r in filtered if r.config_identifier == config]
    if transform is not None:
        filtered = [r for r in filtered if r.transform_name == transform]
    if text_length is not None:
        filtered = [r for r in filtered if r.text_length == text_length]
    if run_id is not None:
        filtered = [r for r in filtered if r.metadata.get("run_id") == run_id]
    return filtered


def get_robustness_report(
    *,
    detector: str | None = None,
    config: str | None = None,
    transform: str | None = None,
    text_length: int | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build the dashboard robustness report from stored artifacts.

    Reuses the Phase 5C programmatic report (aggregation, matrix, Wilson
    CIs) plus a summary block and per-file load warnings.
    """
    from provenance.robustness.benchmark import (
        aggregate_results,
        build_programmatic_report,
        build_robustness_matrix,
    )

    results, warnings = load_robustness_results()
    category_map = get_transform_category_map()
    filtered = _apply_result_filters(
        results, detector=detector, config=config,
        transform=transform, text_length=text_length, run_id=run_id,
    )
    aggregated = aggregate_results(filtered)
    matrix = build_robustness_matrix(aggregated)
    report = build_programmatic_report(
        filtered, aggregated=aggregated, matrix=matrix,
        category_map=category_map,
    )
    # Transform -> category mapping, so dashboard clients can group and
    # filter by category without recomputing anything.
    report["category_map"] = dict(category_map)
    report["summary"] = {
        "total_files_scanned": len(filtered),
        "results_dir": get_robustness_dir(),
        "detectors": sorted({r.detector_name for r in results}),
        "configs": sorted({r.config_identifier for r in results}),
        "transforms": sorted({r.transform_name for r in results}),
        "text_lengths": sorted({r.text_length for r in results if r.text_length is not None}),
        "categories": sorted(set(category_map.get(r.transform_name, "other") for r in results)),
        "filters": {
            "detector": detector,
            "config": config,
            "transform": transform,
            "text_length": text_length,
            "run_id": run_id,
        },
    }
    report["warnings"] = warnings
    return report


def get_robustness_comparison(
    *,
    detector: str | None = None,
    config: str | None = None,
    transform: str | None = None,
    text_length: int | None = None,
) -> dict[str, Any]:
    """Build the cross-model comparison report from stored artifacts.

    Reuses the Phase 5E comparison builder (rows, by_model/by_transform/
    by_category/by_length aggregations with Wilson CIs).
    """
    from provenance.robustness.orchestration import build_comparison_report

    results, warnings = load_robustness_results()
    filtered = _apply_result_filters(
        results, detector=detector, config=config,
        transform=transform, text_length=text_length,
    )
    payloads = [r.to_dict() for r in filtered]
    report = build_comparison_report(
        payloads,
        run_id="api",
        benchmark_name="robustness-comparison",
        category_map=get_transform_category_map(),
    )
    out = report.to_dict()
    out["warnings"] = warnings
    return out


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
