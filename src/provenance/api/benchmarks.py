"""Dashboard-triggered benchmark runs (Phase 6C).

Runs execute on the existing background :class:`ThreadPoolExecutor`
(no external queue) and reuse the shared
:func:`provenance.robustness.execution.run_plan_experiment` pipeline, so
API-triggered runs are byte-identical to CLI ``benchmark plan`` runs.

Concurrency: a process-global semaphore caps simultaneous benchmark
executions (default 1 — model inference is serialized per shared instance,
so parallel runs would only contend). Additional runs wait in the executor
queue. ``PROVENANCE_MAX_BENCHMARK_RUNS`` configures the cap.

Lifecycle: queued → running → completed / failed, with
cancellation_requested → cancelled. Cancellation is cooperative: the worker
checks before acquiring the execution slot and after the runner finishes
(generation itself has no checkpoints). State persists in the
``benchmark_runs`` table; restart recovery marks interrupted runs failed
(never successful) with artifacts preserved for resume-on-retry.

``EXPERIMENT_FN`` is an injectable seam for tests (fake executors, no HF).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("provenance.api")

#: Environment variable capping simultaneous benchmark executions.
MAX_RUNS_ENV_VAR = "PROVENANCE_MAX_BENCHMARK_RUNS"

#: Default cap: serialize benchmark executions.
DEFAULT_MAX_RUNS = 1

#: Validation constraints (also exposed via the options endpoint).
MAX_LENGTHS_COUNT = 10
MAX_TEXT_LENGTH = 500
MAX_SAMPLES = 50
MAX_TRANSFORMS = 50
MAX_CONFIG_BYTES = 1024 * 1024

#: Statuses that may be cancelled.
CANCELLABLE_STATUSES = frozenset({"queued", "running", "cancellation_requested"})

#: Statuses that may be retried (in place; artifacts resume via manifest).
RETRYABLE_STATUSES = frozenset({"failed", "cancelled"})

#: Detectors that can complete an end-to-end generation benchmark run.
#: Only the reference detectors work: generation is reference-pipeline
#: based and evaluation must accept HF experiment configs. The simulation
#: detectors (`kgw`, `synthid`) evaluate pre-generated controlled text —
#: simulation configs carry no model section (generation bails out) and HF
#: configs are rejected by their loaders — so no config can complete a run
#: with them. (`unicode` cannot generate at all.)
RUNNABLE_DETECTORS = ("kgw-reference", "synthid-reference")

#: Terminal statuses.
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

# Injectable experiment function for tests. Production path uses the shared
# CLI-identical pipeline from provenance.robustness.execution.
EXPERIMENT_FN: Callable[[Any, Path], list[dict[str, Any]]] | None = None

_semaphore: threading.Semaphore | None = None
_semaphore_lock = threading.Lock()


def _get_max_runs() -> int:
    """Read the concurrency cap from the environment (defensive)."""
    try:
        return max(1, int(os.environ.get(MAX_RUNS_ENV_VAR, DEFAULT_MAX_RUNS)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_RUNS


def _get_semaphore() -> threading.Semaphore:
    """Process-global execution semaphore (created lazily)."""
    global _semaphore
    with _semaphore_lock:
        if _semaphore is None:
            _semaphore = threading.Semaphore(_get_max_runs())
        return _semaphore


def get_runs_dir() -> Path:
    """Base directory for API-triggered run artifacts."""
    from provenance.api.service import get_robustness_dir
    return Path(get_robustness_dir()) / "runs"


# ── Validation ───────────────────────────────────────────────────────

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_benchmark_config(data: Any) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Validate a benchmark-run configuration dict.

    Returns ``(normalized_config, errors)``. ``errors`` is a list of
    ``{"field": ..., "message": ...}``; empty means valid. Normalized config
    keys: detector, config, profile, transforms, lengths, samples, seed.
    Backend validation is authoritative — the dashboard mirrors it for UX
    only.
    """
    errors: list[dict[str, str]] = []
    if not isinstance(data, dict):
        return None, [{"field": "", "message": "Configuration must be a JSON object"}]

    allowed = {"detector", "config", "profile", "transforms", "lengths", "samples", "seed"}
    for key in data:
        if key not in allowed:
            errors.append({"field": key, "message": f"Unknown field: {key!r}"})

    from provenance.detectors.registry import get_registry
    registry = get_registry()

    detector = data.get("detector")
    capability = registry.get(detector) if isinstance(detector, str) else None
    if not isinstance(detector, str) or capability is None:
        errors.append({
            "field": "detector",
            "message": f"Unknown detector: {detector!r}. Known: {registry.names()}",
        })
        capability = None
    elif detector == "unicode":
        errors.append({
            "field": "detector",
            "message": "unicode detector does not support sample generation; "
                       "evaluate it with text-file robustness mode instead",
        })
    elif detector in ("kgw", "synthid"):
        errors.append({
            "field": "detector",
            "message": f"Detector {detector!r} is a simulation detector and cannot drive "
                       f"generation benchmarks; use '{detector}-reference' with an HF "
                       f"experiment config instead",
        })
    elif not capability.capability.supports_benchmarking:
        errors.append({
            "field": "detector",
            "message": f"Detector {detector!r} does not support benchmarking",
        })

    # Config file: required for watermark detectors, validated structurally
    # (exists, JSON, size-capped). Never loaded as a model here.
    config = data.get("config")
    needs_config = bool(capability and capability.capability.requires_config)
    if needs_config:
        if not isinstance(config, str) or not config:
            errors.append({
                "field": "config",
                "message": f"Config path is required for the {detector!r} detector",
            })
        else:
            config_error = _validate_config_file(config, detector)
            if config_error is not None:
                errors.append({"field": "config", "message": config_error})
    elif config is not None and not isinstance(config, str):
        errors.append({"field": "config", "message": "Config must be a string path"})

    # Profile XOR transforms (both absent = all baseline transforms).
    profile = data.get("profile")
    transforms = data.get("transforms")
    if profile is not None:
        from provenance.robustness.profiles import PROFILES
        if not isinstance(profile, str) or profile not in PROFILES:
            errors.append({
                "field": "profile",
                "message": f"Unknown profile: {profile!r}. Known: {sorted(PROFILES)}",
            })
    if transforms is not None:
        if not isinstance(transforms, list) or not transforms:
            errors.append({"field": "transforms", "message": "transforms must be a non-empty list of names"})
        else:
            try:
                from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
                from provenance.robustness.transforms import TRANSFORM_MAP
                known = set(TRANSFORM_MAP) | set(ADVANCED_TRANSFORM_MAP)
            except Exception:
                known = set()
            if len(transforms) > MAX_TRANSFORMS:
                errors.append({
                    "field": "transforms",
                    "message": f"At most {MAX_TRANSFORMS} transforms allowed",
                })
            for name in transforms:
                if not isinstance(name, str) or (known and name not in known):
                    errors.append({"field": "transforms", "message": f"Unknown transform: {name!r}"})

    lengths = data.get("lengths")
    if not isinstance(lengths, list) or not lengths:
        errors.append({"field": "lengths", "message": "lengths must be a non-empty list of integers"})
    else:
        if len(lengths) > MAX_LENGTHS_COUNT:
            errors.append({
                "field": "lengths",
                "message": f"At most {MAX_LENGTHS_COUNT} lengths allowed",
            })
        for length in lengths:
            if not _is_int(length) or length < 1 or length > MAX_TEXT_LENGTH:
                errors.append({
                    "field": "lengths",
                    "message": f"Each length must be an integer in [1, {MAX_TEXT_LENGTH}], got {length!r}",
                })

    samples = data.get("samples")
    if not _is_int(samples) or samples < 1 or samples > MAX_SAMPLES:
        errors.append({
            "field": "samples",
            "message": f"samples must be an integer in [1, {MAX_SAMPLES}], got {samples!r}",
        })

    seed = data.get("seed")
    if not _is_int(seed):
        errors.append({"field": "seed", "message": f"seed must be an integer, got {seed!r}"})

    if errors:
        return None, errors
    return {
        "detector": detector,
        "config": config,
        "profile": profile,
        "transforms": list(transforms) if transforms is not None else None,
        "lengths": list(lengths),
        "samples": samples,
        "seed": seed,
    }, []


def _validate_config_file(config: str, detector: str | None) -> str | None:
    """Structural validation of a detector config file. Returns error or None.

    Reads and JSON-parses the file (size-capped, ``.json`` only) and checks
    for model/tokenizer sections. Error messages never include file
    contents — only the failure reason.
    """
    if not config.endswith(".json"):
        return "Config must be a .json file"
    try:
        resolved = Path(config).expanduser().resolve()
    except Exception:
        return f"Invalid config path: {config!r}"
    if not resolved.is_file():
        return f"Config file not found: {config!r}"
    try:
        if resolved.stat().st_size > MAX_CONFIG_BYTES:
            return "Config file exceeds 1MB"
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return "Config file is not valid JSON"
    if not isinstance(payload, dict):
        return "Config file must contain a JSON object"
    # Watermark detectors need an HF experiment config (model + tokenizer).
    if "model" not in payload or "tokenizer" not in payload:
        return (
            "Config does not look like an HF experiment config "
            "(missing 'model'/'tokenizer' sections); simulation configs "
            "cannot generate watermarked samples"
        )
    return None


def get_benchmark_options() -> dict[str, Any]:
    """API-visible schema of valid benchmark inputs (registries, not hardcoded)."""
    from provenance.detectors.registry import get_registry
    from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORMS
    from provenance.robustness.profiles import PROFILES
    from provenance.robustness.transforms import ALL_TRANSFORMS

    reg = get_registry()
    detectors = [
        {
            "name": cap.name,
            "display_name": cap.display_name,
            "implementation_kind": cap.implementation_kind,
            "compatibility": cap.compatibility,
            "requires_config": cap.requires_config,
            "supports_generation": cap.supports_generation,
            "supports_benchmarking": cap.supports_benchmarking,
        }
        for cap in reg.capabilities()
        if cap.name in RUNNABLE_DETECTORS
    ]
    profiles = [
        {"name": p.name, "description": p.description, "transform_names": list(p.transform_names)}
        for p in PROFILES.values()
    ]
    transforms = [
        {"name": t.name, "category": "baseline", "description": t.description}
        for t in ALL_TRANSFORMS
    ] + [
        {"name": t.name, "category": t.category.value, "description": t.description}
        for t in ADVANCED_TRANSFORMS
    ]
    transforms.sort(key=lambda t: t["name"])
    return {
        "detectors": detectors,
        "profiles": profiles,
        "transforms": transforms,
        "constraints": {
            "max_lengths_count": MAX_LENGTHS_COUNT,
            "max_text_length": MAX_TEXT_LENGTH,
            "max_samples": MAX_SAMPLES,
            "max_transforms": MAX_TRANSFORMS,
            "notes": "lengths/samples/seed must be integers; transforms is optional when profile is set",
        },
    }


# ── Submission & execution ───────────────────────────────────────────

def submit_benchmark_run(run_id: str) -> None:
    """Queue a benchmark run on the existing background executor."""
    from provenance.api.jobs import get_executor
    get_executor().submit(_run_worker, run_id)


def _experiment_fn_default(spec: Any, exp_out_dir: Path) -> list[dict[str, Any]]:
    from provenance.robustness.execution import run_plan_experiment
    return run_plan_experiment(spec, exp_out_dir)


def _run_worker(run_id: str) -> None:
    """Execute one benchmark run: single spec via BenchmarkRunner."""
    import time as _time

    from provenance.api.state import get_repo

    repo = get_repo()
    run = repo.get_benchmark_run(run_id)
    if run is None:
        logger.warning("Benchmark run %s disappeared before execution", run_id)
        return
    if run["status"] != "queued":
        logger.info("Benchmark run %s not queued at pickup (status=%s)", run_id, run["status"])
        return

    # Concurrency cap: block here until a slot frees. Runs pile up in the
    # executor queue instead of executing in parallel.
    sem = _get_semaphore()
    sem.acquire()
    try:
        return _execute_run(run_id)
    finally:
        sem.release()


def _execute_run(run_id: str) -> None:
    import time as _time

    from provenance.api.state import get_repo
    from provenance.robustness.orchestration import (
        PLAN_SCHEMA_VERSION,
        BenchmarkPlan,
        BenchmarkRunner,
        BenchmarkSpec,
    )

    repo = get_repo()
    run = repo.get_benchmark_run(run_id)
    if run is None or run["status"] != "queued":
        return  # cancelled while waiting for a slot

    cfg = run["config"]
    out_dir = Path(run["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    repo.update_benchmark_run(run_id=run_id, status="running", started_at=now)
    spec = BenchmarkSpec(
        detector=cfg["detector"],
        config=cfg.get("config") or "",
        lengths=tuple(cfg["lengths"]),
        samples=cfg["samples"],
        seed=cfg["seed"],
        profile=cfg.get("profile"),
        transforms=tuple(cfg["transforms"]) if cfg.get("transforms") else None,
    )
    _set_progress(repo, run_id, total=1, completed=0, failed=0, current=spec.experiment_id)
    started = _time.perf_counter()

    try:
        plan = BenchmarkPlan(
            schema_version=PLAN_SCHEMA_VERSION,
            name=f"api-run-{run_id}",
            specs=(spec,),
        )
        fn = EXPERIMENT_FN or _experiment_fn_default
        manifest = BenchmarkRunner(plan=plan, resume=True, experiment_fn=fn).run(out_dir)
        duration_ms = round((_time.perf_counter() - started) * 1000, 2)

        _stamp_run_metadata(out_dir, run_id, spec.experiment_id)

        entry = manifest.experiments[0] if manifest.experiments else None
        n_completed = len(manifest.completed_experiments)
        n_failed = len(manifest.failed_experiments)
        _set_progress(
            repo, run_id, total=len(manifest.experiments),
            completed=n_completed, failed=n_failed,
            current=spec.experiment_id if n_completed + n_failed == 0 else None,
        )

        # Cooperative cancellation: worker checks after the runner finishes.
        current = repo.get_benchmark_run(run_id)
        if current is not None and current["status"] == "cancellation_requested":
            repo.update_benchmark_run(
                run_id=run_id, status="cancelled", completed_at=now,
                error_message="Benchmark run cancelled by user",
                duration_ms=duration_ms,
            )
            logger.info("Benchmark run %s cancelled after execution", run_id)
            return

        summary = {
            "experiment_id": spec.experiment_id,
            "experiments_total": len(manifest.experiments),
            "experiments_completed": n_completed,
            "experiments_failed": n_failed,
            "out_dir": str(out_dir),
            "has_results": (out_dir / spec.experiment_id / "benchmark_results.jsonl").exists(),
            "error": (entry.error_message if entry and entry.status == "failed" else None),
        }
        if manifest.is_complete:
            repo.complete_benchmark_run(
                run_id=run_id, result_json=json.dumps(summary), duration_ms=duration_ms,
            )
            logger.info("Benchmark run %s completed in %.1fs", run_id, duration_ms / 1000)
        else:
            # Partial failure is a failure, never a success.
            detail = summary["error"] or f"{n_failed}/{len(manifest.experiments)} experiments failed"
            repo.fail_benchmark_run(
                run_id=run_id, error_message=detail[:500], duration_ms=duration_ms,
            )
            logger.warning("Benchmark run %s failed: %s", run_id, detail[:200])
    except Exception as exc:
        duration_ms = round((_time.perf_counter() - started) * 1000, 2)
        message = f"{type(exc).__name__}: {exc}"[:500]
        repo.fail_benchmark_run(run_id=run_id, error_message=message, duration_ms=duration_ms)
        logger.warning("Benchmark run %s failed: %s", run_id, message[:200])


def _set_progress(
    repo: Any, run_id: str, *, total: int, completed: int, failed: int, current: str | None,
) -> None:
    """Persist honest progress derived from completed/total work only."""
    repo.update_benchmark_run(
        run_id=run_id,
        progress_json=json.dumps({
            "experiments_total": total,
            "experiments_completed": completed,
            "experiments_failed": failed,
            "current_experiment": current,
        }),
    )


def _stamp_run_metadata(out_dir: Path, run_id: str, experiment_id: str) -> None:
    """Stamp run_id/experiment_id into stored benchmark result metadata.

    Enables deterministic run → results association via the existing
    ``run_id`` filter on ``GET /v1/robustness/results``. Files that fail
    to parse are left untouched (load warnings surface them).
    """
    for path in sorted(out_dir.rglob("benchmark_results.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            stamped = []
            for line in lines:
                if not line.strip():
                    continue
                obj = json.loads(line)
                meta = obj.get("metadata") or {}
                meta.setdefault("run_id", run_id)
                meta.setdefault("experiment_id", experiment_id)
                obj["metadata"] = meta
                stamped.append(json.dumps(obj, sort_keys=True))
            path.write_text("\n".join(stamped) + ("\n" if stamped else ""), encoding="utf-8")
        except Exception:
            logger.warning("Could not stamp run metadata into %s", path, exc_info=True)


def create_run_id() -> str:
    return f"brun-{uuid.uuid4().hex[:12]}"
