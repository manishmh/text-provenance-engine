"""Benchmark orchestration and cross-model comparison.

Phase 5E provides:
- Versioned benchmark plan schema
- Run manifests for reproducibility
- Deterministic experiment IDs
- Cross-model comparison reports
- Resume/partial-run support
- Failure isolation

No raw text is stored in any manifest or report.
No API keys, watermark keys, or credentials are stored in manifests.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


# ── Schema versions ───────────────────────────────────────────────────

PLAN_SCHEMA_VERSION = "provenance-benchmark-plan-v1"
REPORT_SCHEMA_VERSION = "provenance-benchmark-report-v1"
SPEC_SCHEMA_VERSION = "provenance-benchmark-spec-v1"


# ── Deterministic experiment ID ───────────────────────────────────────

def _canonical_json(obj: Any) -> str:
    """Produce a canonical JSON string for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_experiment_id(spec: dict[str, Any]) -> str:
    """Compute a stable experiment ID from a normalized specification.

    Uses SHA-256 of the canonical JSON of the experiment parameters.
    Does not include timestamps — the same spec always produces the same ID.
    """
    canonical = _canonical_json(spec)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ── Benchmark Spec (single experiment) ───────────────────────────────

@dataclass(frozen=True)
class BenchmarkSpec:
    """Versioned specification for a single benchmark experiment.

    ``config`` is the detector config path/identifier and doubles as the
    model/config identifier when ``model_identifier`` is unset. No raw
    generated text is stored — only hashes/identifiers.
    """

    detector: str
    config: str  # path or identifier
    lengths: tuple[int, ...]
    samples: int
    seed: int
    profile: str | None = None
    transforms: tuple[str, ...] | None = None
    benchmark_name: str | None = None
    model_identifier: str | None = None
    schema_version: str = SPEC_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_model(self) -> str:
        """Human-readable model/config identifier for reports."""
        return self.model_identifier or self.config

    @property
    def experiment_id(self) -> str:
        """Deterministic experiment ID from normalized spec."""
        return compute_experiment_id({
            "detector": self.detector,
            "config": self.config,
            "model_identifier": self.model_identifier,
            "lengths": sorted(self.lengths),
            "samples": self.samples,
            "seed": self.seed,
            "profile": self.profile,
            "transforms": sorted(self.transforms) if self.transforms else None,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detector": self.detector,
            "config": self.config,
            "model_identifier": self.model_identifier,
            "lengths": list(self.lengths),
            "samples": self.samples,
            "seed": self.seed,
            "profile": self.profile,
            "transforms": list(self.transforms) if self.transforms else None,
            "benchmark_name": self.benchmark_name,
            "metadata": dict(self.metadata),
            "experiment_id": self.experiment_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkSpec:
        data = dict(data)
        data.pop("experiment_id", None)  # computed, not stored
        lengths = data.get("lengths", ())
        transforms = data.get("transforms")
        return cls(
            detector=data["detector"],
            config=data["config"],
            lengths=tuple(lengths),
            samples=data.get("samples", 10),
            seed=data.get("seed", 42),
            profile=data.get("profile"),
            transforms=tuple(transforms) if transforms else None,
            benchmark_name=data.get("benchmark_name"),
            model_identifier=data.get("model_identifier"),
            schema_version=data.get("schema_version", SPEC_SCHEMA_VERSION),
            metadata=data.get("metadata", {}),
        )


# ── Benchmark Plan (multiple experiments) ─────────────────────────────

@dataclass(frozen=True)
class BenchmarkPlan:
    """A plan containing multiple benchmark experiments."""

    schema_version: str
    name: str
    specs: tuple[BenchmarkSpec, ...]
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plan_id(self) -> str:
        """Deterministic plan ID from specs."""
        spec_dicts = [s.to_dict() for s in self.specs]
        return compute_experiment_id({"name": self.name, "specs": spec_dicts})

    @property
    def experiment_ids(self) -> list[str]:
        return [s.experiment_id for s in self.specs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "specs": [s.to_dict() for s in self.specs],
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkPlan:
        data = dict(data)
        data.pop("plan_id", None)
        specs = tuple(BenchmarkSpec.from_dict(s) for s in data.get("specs", []))
        return cls(
            schema_version=data.get("schema_version", PLAN_SCHEMA_VERSION),
            name=data["name"],
            specs=specs,
            output_dir=data.get("output_dir"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkPlan:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


# ── Plan Validation ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationError:
    """A validation error with context."""

    field: str
    message: str
    spec_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"field": self.field, "message": self.message}
        if self.spec_index is not None:
            d["spec_index"] = self.spec_index
        return d


def validate_plan(plan: BenchmarkPlan) -> list[ValidationError]:
    """Validate a benchmark plan. Returns list of errors (empty = valid)."""
    errors: list[ValidationError] = []

    if not plan.name:
        errors.append(ValidationError(field="name", message="Plan name is required"))

    if not plan.specs:
        errors.append(ValidationError(field="specs", message="Plan must contain at least one experiment spec"))
        return errors

    # Known detectors — from the registry (single source of truth).
    try:
        from provenance.detectors.registry import get_registry
        known_detectors = set(get_registry().names())
    except Exception:
        known_detectors = {"unicode", "kgw", "kgw-reference", "synthid", "synthid-reference"}

    seen_ids: set[str] = set()
    for i, spec in enumerate(plan.specs):
        idx_str = f"spec[{i}]"

        if spec.detector not in known_detectors:
            errors.append(ValidationError(
                field="detector", spec_index=i,
                message=f"Unknown detector: {spec.detector!r}. Known: {sorted(known_detectors)}",
            ))

        # Config is required for watermark detectors, optional for unicode
        watermark_detectors = {"kgw", "kgw-reference", "synthid", "synthid-reference"}
        if spec.detector in watermark_detectors and not spec.config:
            errors.append(ValidationError(
                field="config", spec_index=i,
                message=f"Config path is required for {spec.detector} detector",
            ))

        if not spec.lengths:
            errors.append(ValidationError(
                field="lengths", spec_index=i,
                message="At least one length is required",
            ))

        for length in spec.lengths:
            if length <= 0:
                errors.append(ValidationError(
                    field="lengths", spec_index=i,
                    message=f"Length must be positive, got {length}",
                ))

        if spec.samples <= 0:
            errors.append(ValidationError(
                field="samples", spec_index=i,
                message=f"Samples must be positive, got {spec.samples}",
            ))

        # Known profiles — from the profiles module (single source of truth).
        try:
            from provenance.robustness.profiles import list_profile_names
            known_profiles: set = set(list_profile_names()) | {None}
        except Exception:
            known_profiles = {"formatting", "unicode", "whitespace", "casing", "punctuation", "lexical", "tokenization-sensitive", "all_safe", None}
        if spec.profile not in known_profiles:
            errors.append(ValidationError(
                field="profile", spec_index=i,
                message=f"Unknown profile: {spec.profile!r}. Known: {sorted(p for p in known_profiles if p)}",
            ))

        # Explicit transform names must exist (incompatible configuration)
        if spec.transforms:
            try:
                from provenance.robustness.advanced_transforms import (
                    get_all_advanced_transforms,
                )
                from provenance.robustness.transforms import get_all_transforms
                known = {t.name for t in get_all_transforms()}
                known |= {t.name for t in get_all_advanced_transforms()}
            except Exception:
                known = set()
            for name in spec.transforms:
                if known and name not in known:
                    errors.append(ValidationError(
                        field="transforms", spec_index=i,
                        message=f"Unknown transform: {name!r}",
                    ))

        # Duplicate detection
        eid = spec.experiment_id
        if eid in seen_ids:
            errors.append(ValidationError(
                field="experiment_id", spec_index=i,
                message=f"Duplicate experiment ID: {eid}",
            ))
        seen_ids.add(eid)

    return errors


# ── Plan-level convenience API ──────────────────────────────────────

def load_benchmark_plan(path: str | Path) -> BenchmarkPlan:
    """Load a benchmark plan from a JSON file."""
    return BenchmarkPlan.from_file(path)


def validate_benchmark_plan(plan: BenchmarkPlan) -> list[ValidationError]:
    """Validate a benchmark plan. Returns errors (empty = valid)."""
    return validate_plan(plan)


def run_benchmark_plan(
    plan: BenchmarkPlan,
    out_dir: str | Path,
    *,
    resume: bool = False,
    force: bool = False,
    experiment_fn: Callable[[BenchmarkSpec, Path], list[dict[str, Any]]] | None = None,
) -> RunManifest:
    """Validate and run a benchmark plan, returning the run manifest.

    Raises :class:`ValueError` if the plan fails validation.
    """
    errors = validate_plan(plan)
    if errors:
        details = "; ".join(
            f"{e.field}" + (f"[spec {e.spec_index}]" if e.spec_index is not None else "") + f": {e.message}"
            for e in errors
        )
        raise ValueError(f"Invalid benchmark plan: {details}")
    runner = BenchmarkRunner(plan=plan, resume=resume, force=force, experiment_fn=experiment_fn)
    return runner.run(out_dir)


# ── Run Manifest ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExperimentStatus:
    """Status of a single experiment within a run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExperimentManifest:
    """Manifest entry for a single experiment in a run."""

    experiment_id: str
    spec: dict[str, Any]
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    result_files: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "spec": dict(self.spec),
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_files": list(self.result_files),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentManifest:
        return cls(
            experiment_id=data["experiment_id"],
            spec=data["spec"],
            status=data["status"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result_files=data.get("result_files", []),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
        )


@dataclass
class RunManifest:
    """Manifest for an entire benchmark run."""

    run_id: str
    benchmark_version: str
    plan_name: str
    started_at: str
    completed_at: str | None = None
    experiments: list[ExperimentManifest] = field(default_factory=list)
    status: str = "running"
    output_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Deterministic ID of the plan this run belongs to. Used on resume to
    # refuse a manifest written for a different plan. None for manifests
    # written before plan tracking existed (treated as compatible).
    plan_id: str | None = None

    @property
    def completed_experiments(self) -> list[ExperimentManifest]:
        return [e for e in self.experiments if e.status == ExperimentStatus.COMPLETED]

    @property
    def failed_experiments(self) -> list[ExperimentManifest]:
        return [e for e in self.experiments if e.status == ExperimentStatus.FAILED]

    @property
    def pending_experiments(self) -> list[ExperimentManifest]:
        return [e for e in self.experiments if e.status in (ExperimentStatus.PENDING, ExperimentStatus.RUNNING)]

    @property
    def is_complete(self) -> bool:
        return all(e.status in (ExperimentStatus.COMPLETED, ExperimentStatus.SKIPPED)
                   for e in self.experiments)

    def get_experiment(self, experiment_id: str) -> ExperimentManifest | None:
        for e in self.experiments:
            if e.experiment_id == experiment_id:
                return e
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "benchmark_version": self.benchmark_version,
            "plan_name": self.plan_name,
            "plan_id": self.plan_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "experiments": [e.to_dict() for e in self.experiments],
            "status": self.status,
            "output_dir": self.output_dir,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        return cls(
            run_id=data["run_id"],
            benchmark_version=data["benchmark_version"],
            plan_name=data["plan_name"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            experiments=[ExperimentManifest.from_dict(e) for e in data.get("experiments", [])],
            status=data.get("status", "running"),
            output_dir=data.get("output_dir"),
            metadata=data.get("metadata", {}),
            plan_id=data.get("plan_id"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> RunManifest:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)


def create_run_manifest(plan: BenchmarkPlan) -> RunManifest:
    """Create a new run manifest from a plan."""
    now = datetime.now(timezone.utc).isoformat()
    return RunManifest(
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        benchmark_version=PLAN_SCHEMA_VERSION,
        plan_name=plan.name,
        plan_id=plan.plan_id,
        started_at=now,
        experiments=[
            ExperimentManifest(
                experiment_id=spec.experiment_id,
                spec=spec.to_dict(),
                status=ExperimentStatus.PENDING,
            )
            for spec in plan.specs
        ],
        status="running",
        output_dir=plan.output_dir,
    )


# ── Benchmark Runner ──────────────────────────────────────────────────

class BenchmarkRunner:
    """Orchestrates benchmark experiments with resume/force support."""

    def __init__(
        self,
        plan: BenchmarkPlan,
        *,
        resume: bool = False,
        force: bool = False,
        experiment_fn: Callable[[BenchmarkSpec, Path], list[dict[str, Any]]] | None = None,
    ):
        self.plan = plan
        self.resume = resume
        self.force = force
        self.experiment_fn = experiment_fn or self._default_experiment_fn
        self.manifest: RunManifest | None = None

    def _default_experiment_fn(
        self, spec: BenchmarkSpec, out_dir: Path,
    ) -> list[dict[str, Any]]:
        """Default experiment function — stub that returns empty results.

        In production this would invoke the actual generation/detection pipeline.
        """
        return []

    def run(self, out_dir: str | Path) -> RunManifest:
        """Execute the benchmark plan and return the manifest."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing manifest for resume
        manifest_path = out_dir / "manifest.json"
        if self.resume and manifest_path.exists():
            self.manifest = RunManifest.from_file(manifest_path)
            stored_plan_id = self.manifest.plan_id
            if stored_plan_id is not None and stored_plan_id != self.plan.plan_id:
                raise ValueError(
                    "Existing manifest belongs to a different plan "
                    f"(manifest plan_id={stored_plan_id}, current plan_id={self.plan.plan_id}). "
                    "Use a different --out-dir or delete manifest.json to start fresh."
                )
        else:
            self.manifest = create_run_manifest(self.plan)

        for em in self.manifest.experiments:
            # Skip completed experiments unless force
            if em.status == ExperimentStatus.COMPLETED and not self.force:
                continue

            # Retry failed/pending/running experiments on resume so a failure
            # doesn't permanently wedge the run: only COMPLETED is skipped
            # (above). Without resume the manifest is fresh, so every entry
            # is PENDING and this branch never triggers.

            # Mark as running (clearing any previous failure details on retry)
            em.status = ExperimentStatus.RUNNING
            em.started_at = datetime.now(timezone.utc).isoformat()
            em.error_type = None
            em.error_message = None
            self.manifest.save(manifest_path)

            try:
                spec = BenchmarkSpec.from_dict(em.spec)
                exp_out_dir = out_dir / spec.experiment_id
                exp_out_dir.mkdir(parents=True, exist_ok=True)

                results = self.experiment_fn(spec, exp_out_dir)

                em.status = ExperimentStatus.COMPLETED
                em.completed_at = datetime.now(timezone.utc).isoformat()
                em.result_files = [str(exp_out_dir / "results.jsonl")]

            except Exception as exc:
                em.status = ExperimentStatus.FAILED
                em.completed_at = datetime.now(timezone.utc).isoformat()
                em.error_type = type(exc).__name__
                em.error_message = str(exc)[:500]  # sanitize

            # Save after each experiment for crash recovery
            self.manifest.save(manifest_path)

        self.manifest.completed_at = datetime.now(timezone.utc).isoformat()
        self.manifest.status = "completed" if self.manifest.is_complete else "partial"
        self.manifest.save(manifest_path)

        return self.manifest


# ── Cross-Model Comparison Report ─────────────────────────────────────

def _wilson(successes: int, n: int) -> tuple[float, float]:
    """Wilson score interval (simplified)."""
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z = 1.959963984540054  # 95% CI
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class ComparisonRow:
    """Single row in the comparison table."""

    model_config: str
    detector: str
    transform: str
    length: int | None
    baseline_rate: float
    transformed_rate: float
    robustness_rate: float
    robustness_ci_low: float
    robustness_ci_high: float
    mean_score_delta: float | None
    n_samples: int
    # Raw paired counts, kept so group aggregations below don't have to
    # reconstruct them from rates (float truncation) and so the robustness
    # ratio can be clamped to a valid [0, 1] rate.
    baseline_detected: int = 0
    transformed_detected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComparisonReport:
    """Cross-model comparison report."""

    schema_version: str
    run_id: str
    benchmark_name: str
    rows: tuple[ComparisonRow, ...]
    by_model: tuple[dict[str, Any], ...]
    by_transform: tuple[dict[str, Any], ...]
    by_category: tuple[dict[str, Any], ...]
    by_length: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "benchmark_name": self.benchmark_name,
            "rows": [r.to_dict() for r in self.rows],
            "by_model": [dict(d) for d in self.by_model],
            "by_transform": [dict(d) for d in self.by_transform],
            "by_category": [dict(d) for d in self.by_category],
            "by_length": [dict(d) for d in self.by_length],
            "limitations": list(self.limitations),
            "metadata": dict(self.metadata),
        }

    def render_text(self) -> str:
        """Render human-readable comparison table."""
        lines = [
            "Cross-Model Benchmark Comparison Report",
            "=" * 60,
            f"Run ID: {self.run_id}",
            f"Benchmark: {self.benchmark_name}",
            f"Schema: {self.schema_version}",
            "",
        ]

        # Main table
        lines.append("Model | Detector | Transform | Length | Baseline | Transformed | Robustness [CI] | Delta | N")
        lines.append("-" * 95)
        for row in self.rows:
            length_str = str(row.length) if row.length else "all"
            delta_str = f"{row.mean_score_delta:+.4f}" if row.mean_score_delta is not None else "n/a"
            lines.append(
                f"{row.model_config:<15} | {row.detector:<12} | {row.transform:<12} | {length_str:>6} | "
                f"{row.baseline_rate:>8.1%} | {row.transformed_rate:>12.1%} | "
                f"{row.robustness_rate:>8.1%} [{row.robustness_ci_low:.2f},{row.robustness_ci_high:.2f}] | "
                f"{delta_str:>7} | {row.n_samples:>3}"
            )

        # By Model
        if self.by_model:
            lines.append("")
            lines.append("By Model")
            lines.append("-" * 40)
            for m in self.by_model:
                rob = m.get("robustness_rate", 0.0)
                ci_lo = m.get("robustness_ci_low", 0.0)
                ci_hi = m.get("robustness_ci_high", 0.0)
                lines.append(
                    f"  {m['model_config']:<25} "
                    f"robustness={rob:.1%} [{ci_lo:.2f},{ci_hi:.2f}] "
                    f"n={m['total_samples']}"
                )

        # By Transform
        if self.by_transform:
            lines.append("")
            lines.append("By Transformation")
            lines.append("-" * 40)
            for t in self.by_transform:
                rob = t.get("robustness_rate", 0.0)
                ci_lo = t.get("robustness_ci_low", 0.0)
                ci_hi = t.get("robustness_ci_high", 0.0)
                lines.append(
                    f"  {t['transform']:<25} "
                    f"robustness={rob:.1%} [{ci_lo:.2f},{ci_hi:.2f}] "
                    f"n={t['total_samples']}"
                )

        # By Length
        if self.by_length:
            lines.append("")
            lines.append("By Length")
            lines.append("-" * 40)
            for l in self.by_length:
                length_val = l.get("length", "all")
                rob = l.get("robustness_rate", 0.0)
                ci_lo = l.get("robustness_ci_low", 0.0)
                ci_hi = l.get("robustness_ci_high", 0.0)
                lines.append(
                    f"  length={length_val:<10} "
                    f"robustness={rob:.1%} [{ci_lo:.2f},{ci_hi:.2f}] "
                    f"n={l['total_samples']}"
                )

        # Limitations
        lines.append("")
        lines.append("Limitations")
        lines.append("-" * 40)
        for lim in self.limitations:
            lines.append(f"  - {lim}")

        return "\n".join(lines)


COMPARISON_LIMITATIONS = [
    "This comparison evaluates robustness under tested deterministic transformations only.",
    "It does NOT produce a misleading single 'best detector' score across incompatible watermark schemes.",
    "Each detector/model combination is evaluated independently.",
    "Results are specific to the tested configurations and transformations.",
    "A high robustness score does not mean the watermark is resistant to removal.",
    "Wilson confidence intervals reflect finite-sample uncertainty.",
]


def build_comparison_report(
    results: Sequence[dict[str, Any]],
    *,
    run_id: str = "",
    benchmark_name: str = "",
    category_map: dict[str, str] | None = None,
) -> ComparisonReport:
    """Build a cross-model comparison report from benchmark results.

    Each result dict should contain:
    detector_name, config_identifier, transform_name, text_length,
    sample_count, baseline_detected, transformed_detected, mean_score_delta.
    """
    rows: list[ComparisonRow] = []
    for r in results:
        n = r.get("sample_count", 0)
        bl = r.get("baseline_detected", 0)
        tr = r.get("transformed_detected", 0)
        # Paired conditional rate, clamped: a transform can flip a baseline
        # miss into a hit, but a preservation rate above 1 is meaningless.
        rob_rate = min(tr, bl) / bl if bl > 0 else 0.0
        rob_ci = _wilson(min(tr, bl), bl) if bl > 0 else (0.0, 0.0)
        bl_rate = bl / n if n > 0 else 0.0
        tr_rate = tr / n if n > 0 else 0.0

        rows.append(ComparisonRow(
            model_config=r.get("config_identifier", "unknown"),
            detector=r.get("detector_name", "unknown"),
            transform=r.get("transform_name", "unknown"),
            length=r.get("text_length"),
            baseline_rate=bl_rate,
            transformed_rate=tr_rate,
            robustness_rate=rob_rate,
            robustness_ci_low=rob_ci[0],
            robustness_ci_high=rob_ci[1],
            mean_score_delta=r.get("mean_score_delta"),
            n_samples=n,
            baseline_detected=bl,
            transformed_detected=tr,
        ))

    # By model aggregation
    from collections import defaultdict
    model_groups: dict[str, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        model_groups[row.model_config].append(row)
    by_model = []
    for model, group in sorted(model_groups.items()):
        total_n = sum(r.n_samples for r in group)
        total_bl = sum(r.baseline_detected for r in group)
        total_tr = sum(r.transformed_detected for r in group)
        rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
        rob_ci = _wilson(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)
        deltas = [r.mean_score_delta for r in group if r.mean_score_delta is not None]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        by_model.append({
            "model_config": model,
            "robustness_rate": rob_rate,
            "robustness_ci_low": rob_ci[0],
            "robustness_ci_high": rob_ci[1],
            "mean_score_delta": mean_delta,
            "total_samples": total_n,
            "detector_count": len({r.detector for r in group}),
        })

    # By transform
    transform_groups: dict[str, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        transform_groups[row.transform].append(row)
    by_transform = []
    for transform, group in sorted(transform_groups.items()):
        total_n = sum(r.n_samples for r in group)
        total_bl = sum(r.baseline_detected for r in group)
        total_tr = sum(r.transformed_detected for r in group)
        rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
        rob_ci = _wilson(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)
        by_transform.append({
            "transform": transform,
            "robustness_rate": rob_rate,
            "robustness_ci_low": rob_ci[0],
            "robustness_ci_high": rob_ci[1],
            "total_samples": total_n,
        })

    # By category
    by_category: list[dict[str, Any]] = []
    if category_map:
        cat_groups: dict[str, list[ComparisonRow]] = defaultdict(list)
        for row in rows:
            cat = category_map.get(row.transform, "unknown")
            cat_groups[cat].append(row)
        for cat, group in sorted(cat_groups.items()):
            total_n = sum(r.n_samples for r in group)
            total_bl = sum(r.baseline_detected for r in group)
            total_tr = sum(r.transformed_detected for r in group)
            rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
            rob_ci = _wilson(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)
            by_category.append({
                "category": cat,
                "robustness_rate": rob_rate,
                "robustness_ci_low": rob_ci[0],
                "robustness_ci_high": rob_ci[1],
                "total_samples": total_n,
            })

    # By length
    length_groups: dict[int | None, list[ComparisonRow]] = defaultdict(list)
    for row in rows:
        length_groups[row.length].append(row)
    by_length = []
    for length, group in sorted(length_groups.items(), key=lambda x: (x[0] is None, x[0] or 0)):
        total_n = sum(r.n_samples for r in group)
        total_bl = sum(r.baseline_detected for r in group)
        total_tr = sum(r.transformed_detected for r in group)
        rob_rate = min(total_tr, total_bl) / total_bl if total_bl > 0 else 0.0
        rob_ci = _wilson(min(total_tr, total_bl), total_bl) if total_bl > 0 else (0.0, 0.0)
        by_length.append({
            "length": length,
            "robustness_rate": rob_rate,
            "robustness_ci_low": rob_ci[0],
            "robustness_ci_high": rob_ci[1],
            "total_samples": total_n,
        })

    return ComparisonReport(
        schema_version=REPORT_SCHEMA_VERSION,
        run_id=run_id,
        benchmark_name=benchmark_name,
        rows=tuple(rows),
        by_model=tuple(by_model),
        by_transform=tuple(by_transform),
        by_category=tuple(by_category),
        by_length=tuple(by_length),
        limitations=COMPARISON_LIMITATIONS,
    )
