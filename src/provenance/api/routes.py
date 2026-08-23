"""FastAPI route handlers for the provenance HTTP API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query

from provenance.api.auth import RequireAPIKey
from provenance.api.db import AnalysisRepository  # Protocol
from provenance.api.middleware import rate_limit_dependency
from provenance.api.models import (
    AnalysisListResponse,
    AnalysisSummary,
    AnalyzeRequest,
    AnalyzeResponse,
    DetectionResultItem,
    HealthResponse,
    ReadyResponse,
)
from provenance.engine import ENGINE_VERSION
from provenance.schemas import DetectionResult

if TYPE_CHECKING:
    from provenance.detectors import WatermarkDetector

router = APIRouter()

# Module-level repository; set via configure_repo() at startup.
_repo: AnalysisRepository | None = None


def configure_repo(repo: AnalysisRepository) -> None:
    """Inject the persistence repository (called at app startup)."""
    global _repo
    _repo = repo


def _get_repo() -> AnalysisRepository:
    if _repo is None:
        raise RuntimeError("AnalysisRepository not configured")
    return _repo


def _unconfigured_result(detector_name: str) -> DetectionResult:
    return DetectionResult(
        detector=detector_name,
        detector_version="unavailable",
        status="not_configured",
        detected=None,
        implementation_kind="unavailable",
        compatibility="unavailable",
        confidence="unavailable",
        evidence={},
        text_requirements={"configuration_required": True},
        limitations=[
            f"{detector_name} detection requires an explicit known watermark configuration.",
            "No AI or human authorship conclusion is available.",
        ],
    )


def _build_detectors(
    detector_names: list[str], config_path: str | None
) -> tuple[list[WatermarkDetector], list[DetectionResult]]:
    """Instantiate detectors and collect unconfigured results."""
    # Lazy imports so the API module loads without torch/transformers
    from provenance.detectors import (
        KGWDetector,
        KGWReferenceDetector,
        SynthIDReferenceDetector,
        SynthIDTextDetector,
    )

    detectors: list[WatermarkDetector] = []
    extra: list[DetectionResult] = []

    for name in detector_names:
        if name == "unicode":
            continue
        if config_path is None:
            extra.append(_unconfigured_result(name))
            continue
        if name == "kgw":
            detectors.append(KGWDetector.from_config_file(config_path))
        elif name == "kgw-reference":
            detectors.append(KGWReferenceDetector.from_config_file(config_path))
        elif name == "synthid":
            detectors.append(SynthIDTextDetector.from_config_file(config_path))
        elif name == "synthid-reference":
            detectors.append(SynthIDReferenceDetector.from_config_file(config_path))

    return detectors, extra


# ---------------------------------------------------------------------------
# Public endpoints (no auth)
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(engine_version=ENGINE_VERSION)


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness probe — verifies the persistence backend is usable."""
    try:
        repo = _get_repo()
        repo.list_analyses(limit=1, offset=0)
        return ReadyResponse(status="ready", engine_version=ENGINE_VERSION)
    except Exception:
        return ReadyResponse(status="not_ready", engine_version=ENGINE_VERSION)


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------


@router.post("/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, _: RequireAPIKey = None, _rl: None = Depends(rate_limit_dependency)) -> AnalyzeResponse:
    from provenance.engine import ProvenanceEngine

    detector_names = request.detectors or ["unicode"]
    include_unicode = "unicode" in detector_names

    detectors, extra_results = _build_detectors(detector_names, request.config_path)

    engine = ProvenanceEngine(include_unicode=include_unicode)
    result = engine.analyze(request.text, detectors=detectors)

    if extra_results:
        result = type(result)(
            engine_version=result.engine_version,
            status=result.status,
            text_stats=result.text_stats,
            results=result.results + extra_results,
            limitations=result.limitations,
            metadata=result.metadata,
        )

    result_dict = result.to_dict()

    # Persist
    repo = _get_repo()
    analysis_id = repo.save(
        text=request.text,
        engine_version=ENGINE_VERSION,
        detectors=detector_names,
        result_dict=result_dict,
    )

    return AnalyzeResponse(
        analysis_id=analysis_id,
        engine_version=result.engine_version,
        status=result.status,
        text_stats=result.text_stats,
        results=[DetectionResultItem(**r) for r in result_dict["results"]],
        limitations=result.limitations,
        metadata=result.metadata,
    )


@router.get("/v1/analyses/{analysis_id}")
def get_analysis(analysis_id: str, _: RequireAPIKey = None, _rl: None = Depends(rate_limit_dependency)) -> dict:
    repo = _get_repo()
    record = repo.get(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return record


def _get_max_list_limit() -> int:
    return int(os.environ.get("MAX_LIST_LIMIT", "100"))


@router.get("/v1/analyses", response_model=AnalysisListResponse)
def list_analyses(
    _: RequireAPIKey = None,
    _rl: None = Depends(rate_limit_dependency),
    limit: int = Query(default=20, ge=1),
    offset: int = Query(default=0, ge=0),
) -> AnalysisListResponse:
    max_limit = _get_max_list_limit()
    if limit > max_limit:
        raise HTTPException(status_code=422, detail=f"limit must be <= {max_limit}")
    repo = _get_repo()
    summaries, total = repo.list_analyses(limit=limit, offset=offset)
    return AnalysisListResponse(
        analyses=[AnalysisSummary(**s) for s in summaries],
        total=total,
        limit=limit,
        offset=offset,
    )
