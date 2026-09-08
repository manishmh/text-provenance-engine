"""Development CLI for local text provenance analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from provenance.detectors import KGWDetector, KGWReferenceDetector, SynthIDReferenceDetector, SynthIDTextDetector
from provenance.schemas import DetectionResult
from provenance.engine import ProvenanceEngine


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="provenance", description="Text provenance signal engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a UTF-8 text file")
    analyze.add_argument("path", type=Path, help="Path to text file")
    analyze.add_argument(
        "--detector",
        action="append",
        choices=["unicode", "kgw", "kgw-reference", "synthid", "synthid-reference"],
        help="Detector to run. Repeat to run multiple detectors. Defaults to unicode.",
    )
    analyze.add_argument(
        "--config",
        type=Path,
        help="Configuration JSON for a selected KGW or SynthID detector",
    )
    analyze.add_argument("--json", action="store_true", help="Emit machine-readable JSON")

    benchmark = subparsers.add_parser("benchmark", help="Run evaluation benchmarks")
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    kgw_bench = benchmark_sub.add_parser(
        "kgw", help="Benchmark the MarkLLM-compatible KGW detector across lengths"
    )
    kgw_bench.add_argument("--config", type=Path, required=True, help="Experiment config JSON")
    kgw_bench.add_argument(
        "--lengths",
        default="50,100,200,500",
        help="Comma-separated token lengths to evaluate (default: 50,100,200,500)",
    )
    kgw_bench.add_argument(
        "--samples", type=int, default=10, help="Samples per length per condition (default: 10)"
    )
    kgw_bench.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    kgw_bench.add_argument(
        "--thresholds",
        default="2,3,4,5",
        help="Comma-separated z-thresholds for TPR/FPR analysis (default: 2,3,4,5)",
    )
    kgw_bench.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/benchmarks/kgw"),
        help="Directory for records.jsonl, report.json and report.txt",
    )
    kgw_bench.add_argument("--json", action="store_true", help="Print the JSON report to stdout")

    synthid_bench = benchmark_sub.add_parser(
        "synthid", help="Benchmark the SynthID-Text reference detector across lengths"
    )
    synthid_bench.add_argument("--config", type=Path, required=True, help="Experiment config JSON")
    synthid_bench.add_argument(
        "--lengths",
        default="50,100,200,500",
        help="Comma-separated token lengths to evaluate (default: 50,100,200,500)",
    )
    synthid_bench.add_argument(
        "--samples", type=int, default=10, help="Samples per length per condition (default: 10)"
    )
    synthid_bench.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    synthid_bench.add_argument(
        "--thresholds",
        default="0.5,0.55,0.6,0.65,0.7",
        help="Comma-separated thresholds for TPR/FPR analysis (default: 0.5,0.55,0.6,0.65,0.7)",
    )
    synthid_bench.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/benchmarks/synthid"),
        help="Directory for records.jsonl, report.json and report.txt",
    )
    synthid_bench.add_argument("--json", action="store_true", help="Print the JSON report to stdout")

    robustness_bench = benchmark_sub.add_parser(
        "robustness", help="Evaluate detector robustness against text transformations"
    )
    robustness_bench.add_argument(
        "--config", type=Path, default=None, help="Detector configuration JSON (required for watermark detectors)"
    )
    robustness_bench.add_argument(
        "--text", type=Path, default=None, help="Path to text file to evaluate (text-file mode)"
    )
    robustness_bench.add_argument(
        "--detector",
        choices=["unicode", "kgw", "kgw-reference", "synthid", "synthid-reference"],
        default="unicode",
        help="Detector to evaluate (default: unicode)",
    )
    robustness_bench.add_argument(
        "--transforms",
        default=None,
        help="Comma-separated transform names (default: all)",
    )
    robustness_bench.add_argument(
        "--profile",
        default=None,
        help="Predefined transform profile (formatting, unicode, whitespace, casing, punctuation, lexical, tokenization-sensitive, all_safe)",
    )
    robustness_bench.add_argument(
        "--lengths",
        default="50,100",
        help="Comma-separated token lengths for watermark generation (default: 50,100)",
    )
    robustness_bench.add_argument(
        "--samples", type=int, default=3,
        help="Samples per length for watermark generation (default: 3)",
    )
    robustness_bench.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    robustness_bench.add_argument("--json", action="store_true", help="Print JSON output")
    robustness_bench.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory for records.jsonl and report.json (watermark mode only)",
    )

    # --- robustness-report subcommand ---
    rr = benchmark_sub.add_parser(
        "robustness-report",
        help="Aggregate and compare previously saved robustness benchmark results",
    )
    rr.add_argument(
        "--input", type=Path, required=True,
        help="Directory containing .jsonl benchmark result files",
    )
    rr.add_argument("--json", action="store_true", help="Print JSON output")
    rr.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to write aggregated report.json")

    # --- plan subcommand ---
    plan_cmd = benchmark_sub.add_parser(
        "plan",
        help="Run a benchmark plan with multiple experiments",
    )
    plan_cmd.add_argument(
        "--plan", type=Path, required=True,
        help="Path to benchmark_plan.json",
    )
    plan_cmd.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory for results and manifest",
    )
    plan_cmd.add_argument(
        "--dry-run", action="store_true",
        help="Print experiments that would be executed without running",
    )
    plan_cmd.add_argument(
        "--resume", action="store_true",
        help="Resume a previously interrupted run",
    )
    plan_cmd.add_argument(
        "--force", action="store_true",
        help="Force re-run of completed experiments",
    )
    plan_cmd.add_argument("--json", action="store_true", help="Print JSON output")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "benchmark":
        if args.benchmark_command == "kgw":
            return _benchmark_kgw(args)
        if args.benchmark_command == "synthid":
            return _benchmark_synthid(args)
        if args.benchmark_command == "robustness":
            return _benchmark_robustness(args)
        if args.benchmark_command == "robustness-report":
            return _robustness_report(args)
        if args.benchmark_command == "plan":
            return _benchmark_plan(args)
        parser.error(f"unknown benchmark: {args.benchmark_command}")
    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part.strip())

def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(",") if part.strip())


def _wrap_advanced_transform(adv_transform, seed: int = 0):
    """Wrap an AdvancedTransform as a Transform-compatible object."""
    from provenance.robustness.transforms import Transform, TransformResult
    def _apply(text: str) -> TransformResult:
        result = adv_transform.apply(text, seed)
        return TransformResult(
            text=result,
            transform_name=adv_transform.name,
            metadata={
                "category": adv_transform.category.value,
                "severity": adv_transform.severity,
                "changed": result != text,
            },
        )
    return Transform(
        name=adv_transform.name,
        description=adv_transform.description,
        func=_apply,
    )


def _benchmark_kgw(args: argparse.Namespace) -> int:
    # Imported lazily: the benchmark runner pulls in torch via the MarkLLM-compatible
    # scorer, so `provenance analyze` stays usable without the optional 'hf' extra.
    from provenance.benchmark import BenchmarkSpec, render_text_report, run_benchmark

    spec = BenchmarkSpec(
        lengths=_parse_int_list(args.lengths),
        samples=args.samples,
        seed=args.seed,
        thresholds=_parse_float_list(args.thresholds),
    )
    report = run_benchmark(config_path=args.config, spec=spec, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))
        print(f"\nWrote records.jsonl, report.json, report.txt to {args.out_dir}")
    return 0


def _benchmark_synthid(args: argparse.Namespace) -> int:
    from provenance.benchmark import BenchmarkSpec, render_text_report
    from provenance.benchmark.runner import run_synthid_benchmark

    spec = BenchmarkSpec(
        lengths=_parse_int_list(args.lengths),
        samples=args.samples,
        seed=args.seed,
        thresholds=_parse_float_list(args.thresholds),
    )
    report = run_synthid_benchmark(config_path=args.config, spec=spec, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text_report(report))
        print(f"\nWrote records.jsonl, report.json, report.txt to {args.out_dir}")
    return 0


def _benchmark_robustness(args: argparse.Namespace) -> int:
    """Evaluate detector robustness against text transformations."""
    from provenance.robustness.transforms import get_all_transforms, get_transform

    # Build transforms from profile, explicit list, or default (all)
    if hasattr(args, 'profile') and args.profile:
        from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
        from provenance.robustness.profiles import get_profile
        profile = get_profile(args.profile)
        seed = getattr(args, "seed", 0)
        transforms = []
        for name in profile.transform_names:
            if name in ADVANCED_TRANSFORM_MAP:
                transforms.append(_wrap_advanced_transform(ADVANCED_TRANSFORM_MAP[name], seed))
            else:
                transforms.append(get_transform(name))
    elif args.transforms:
        transform_names = [t.strip() for t in args.transforms.split(",")]
        transforms = [get_transform(name) for name in transform_names]
    else:
        transforms = get_all_transforms()

    # Watermark mode: generate samples from config and run robustness
    if args.text is None:
        return _benchmark_robustness_watermark(args, transforms)

    # Text-file mode: read file and run single-sample robustness
    return _benchmark_robustness_text(args, transforms)


def _benchmark_robustness_text(args: argparse.Namespace, transforms) -> int:
    """Evaluate robustness against text transformations from a text file."""
    from provenance.detectors.registry import get_registry
    from provenance.robustness.evaluator import compute_statistics, evaluate_robustness

    text = args.text.read_text(encoding="utf-8")
    registry = get_registry()

    # Build detector
    if args.detector == "unicode":
        detector = registry.create("unicode")
    elif args.detector in ("kgw", "kgw-reference", "synthid", "synthid-reference"):
        if args.config is None:
            print(f"Error: --config is required for {args.detector} detector", flush=True)
            return 1
        detector = registry.create(args.detector, config_path=str(args.config))
    else:
        print(f"Error: unsupported detector for robustness: {args.detector}", flush=True)
        return 1

    records = evaluate_robustness(text, detector, transforms)
    stats = compute_statistics(records)

    if args.json:
        output = {
            "detector": args.detector,
            "text_length": len(text),
            "statistics": stats,
            "records": [r.to_dict() for r in records],
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Robustness evaluation: detector={args.detector}, text_length={len(text)}")
        print(f"Transforms evaluated: {stats['total_transforms']}")
        print(f"Overall detection rate: {stats['overall_detection_rate']:.2%}")
        if stats.get("baseline_detected") is not None:
            print(f"Baseline (identity) detected: {stats['baseline_detected']}")
        print()
        for tname, tstats in stats.get("transforms", {}).items():
            rate = tstats["detection_rate"]
            score_str = f"avg_score={tstats['avg_score']:.4f}" if tstats.get("avg_score") is not None else "score=N/A"
            change = tstats.get("score_change")
            change_str = f", delta={change:+.4f}" if change is not None else ""
            print(f"  {tname}: detected={tstats['detected_count']}/{tstats['total_evaluations']}, rate={rate:.2%}, {score_str}{change_str}")
    return 0


def _benchmark_robustness_watermark(args: argparse.Namespace, transforms) -> int:
    """Run watermark robustness: generate → transform → detect."""
    import json
    from pathlib import Path

    from provenance.benchmark.runner import derive_seed, DEFAULT_PROMPTS
    from provenance.robustness.experiments import (
        ExperimentConfig,
        RobustnessReport,
        WatermarkSample,
        compute_robustness_report,
        evaluate_robustness_experiment,
        render_robustness_report,
    )

    if args.config is None:
        print("Error: --config is required for watermark robustness evaluation", flush=True)
        return 1

    detector_name = args.detector
    try:
        lengths = tuple(int(x.strip()) for x in args.lengths.split(",") if x.strip())
    except ValueError:
        print(f"Error: --lengths must be comma-separated integers, got {args.lengths!r}", flush=True)
        return 1
    if not lengths or any(length <= 0 for length in lengths):
        print(f"Error: --lengths must contain positive integers, got {args.lengths!r}", flush=True)
        return 1
    samples = args.samples
    if samples < 1:
        print(f"Error: --samples must be >= 1, got {samples}", flush=True)
        return 1
    seed = args.seed

    print(f"Loading detector '{detector_name}' from {args.config}...", flush=True)

    # Load detector via registry
    from provenance.detectors.registry import get_registry
    registry = get_registry()

    # Determine which variant to load
    detector_variant = detector_name
    detector = registry.create(detector_variant, config_path=str(args.config))

    # Generate watermarked samples using existing generation pipeline
    print(f"Generating watermarked samples (lengths={list(lengths)}, samples={samples})...", flush=True)

    wm_samples = _generate_watermark_samples(
        detector_name=detector_name,
        config_path=str(args.config),
        lengths=lengths,
        samples_per_length=samples,
        seed=seed,
    )

    if wm_samples is None:
        return 1

    print(f"Generated {len(wm_samples)} samples. Running robustness evaluation...", flush=True)

    # Run robustness evaluation
    records = evaluate_robustness_experiment(detector, wm_samples, transforms)

    # Compute report
    experiment_config = ExperimentConfig(
        detector_name=detector_name,
        config_path=str(args.config),
        lengths=lengths,
        samples_per_length=samples,
        seed=seed,
        transforms=transforms,
    )
    report = compute_robustness_report(records, experiment_config)

    if args.json:
        output = {
            "report": report.to_dict(),
            "records": [r.to_dict() for r in records],
        }
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(render_robustness_report(report))

    # Write records to disk if out-dir specified
    if args.out_dir is not None:
        from provenance.benchmark.records import write_jsonl, EvaluationRecord, BENCHMARK_VERSION
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Convert RobustnessRecords to EvaluationRecords for persistence
        eval_records = _robustness_records_to_evaluation_records(records, experiment_config)
        if eval_records:
            write_jsonl(eval_records, out_dir / "records.jsonl")
        (out_dir / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        # Also save Phase 5C benchmark results (grouped by transform + length)
        _write_watermark_benchmark_results(
            records,
            detector_name=detector_name,
            config_identifier=str(args.config),
            seed=seed,
            out_dir=out_dir,
        )
        print(f"\nWrote records.jsonl, report.json, and benchmark_results.jsonl to {out_dir}")

    return 0


def _write_watermark_benchmark_results(records, *, detector_name, config_identifier, seed, out_dir):
    """Persist Phase 5C benchmark results grouped by (transform, length).

    Returns the list of ``RobustnessBenchmarkResult`` objects written to
    ``<out_dir>/benchmark_results.jsonl``. Shared by the ``robustness``
    watermark mode and the ``plan`` experiment runner so both produce the
    same artifact shape.
    """
    from collections import defaultdict

    from provenance.robustness.benchmark import build_benchmark_result, write_benchmark_results

    groups: dict[tuple[str, int | None], list] = defaultdict(list)
    for r in records:
        groups[(r.transform_name, r.metadata.get("length"))].append(r)

    benchmark_results = []
    for (tname, length), trecords in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        benchmark_results.append(build_benchmark_result(
            detector_name=detector_name,
            config_identifier=config_identifier,
            transform_name=tname,
            text_length=length,
            records=[r.to_dict() for r in trecords],
            seed=seed,
        ))
    write_benchmark_results(benchmark_results, Path(out_dir) / "benchmark_results.jsonl")
    return benchmark_results


def _resolve_plan_transforms(spec):
    """Resolve the transform list for a ``BenchmarkSpec``.

    Precedence: explicit ``transforms`` names, then ``profile``, then all
    Phase 5A baseline transforms. Profile and explicit names may refer to
    either the Phase 5A baseline namespace or the Phase 5D advanced
    namespace; advanced names take precedence on collision (the overlapping
    transforms are behaviorally equivalent).
    """
    from provenance.robustness.transforms import get_all_transforms, get_transform

    if spec.transforms:
        names = list(spec.transforms)
    elif spec.profile:
        from provenance.robustness.profiles import get_profile
        names = list(get_profile(spec.profile).transform_names)
    else:
        return get_all_transforms()

    from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
    transforms = []
    for name in names:
        if name in ADVANCED_TRANSFORM_MAP:
            transforms.append(_wrap_advanced_transform(ADVANCED_TRANSFORM_MAP[name], spec.seed))
        else:
            transforms.append(get_transform(name))
    return transforms


def _plan_experiment_fn(spec, exp_out_dir):
    """Execute one ``BenchmarkSpec`` for ``benchmark plan``.

    Runs the real watermark pipeline — generate samples, evaluate across
    transforms, persist ``results.jsonl`` — and returns the result dicts.
    Any exception is caught by the runner and recorded as a FAILED
    experiment (failure isolation), never aborting the rest of the plan.
    """
    from provenance.detectors.registry import get_registry
    from provenance.robustness.experiments import evaluate_robustness_experiment

    detector_name = spec.detector
    if detector_name == "unicode":
        raise ValueError(
            "unicode detector does not support watermark sample generation; "
            "evaluate it with `benchmark robustness --text <file> --detector unicode` instead"
        )
    if detector_name not in ("kgw", "kgw-reference", "synthid", "synthid-reference"):
        raise ValueError(f"Cannot generate samples for detector: {detector_name}")

    transforms = _resolve_plan_transforms(spec)
    detector = get_registry().create(detector_name, config_path=spec.config)

    wm_samples = _generate_watermark_samples(
        detector_name=detector_name,
        config_path=spec.config,
        lengths=spec.lengths,
        samples_per_length=spec.samples,
        seed=spec.seed,
    )
    if wm_samples is None:
        raise ValueError(
            f"Config {spec.config} does not support watermark sample generation "
            "(requires an HF experiment config with model/tokenizer sections)"
        )

    records = evaluate_robustness_experiment(detector, wm_samples, transforms)
    results = _write_watermark_benchmark_results(
        records,
        detector_name=detector_name,
        config_identifier=spec.config,
        seed=spec.seed,
        out_dir=exp_out_dir,
    )
    return [r.to_dict() for r in results]


def _robustness_report(args: argparse.Namespace) -> int:
    """Aggregate previously saved robustness benchmark results."""
    from provenance.robustness.benchmark import (
        aggregate_results,
        build_robustness_matrix,
        build_programmatic_report,
        load_results_from_directory,
        render_benchmark_report_text,
    )

    results = load_results_from_directory(args.input)
    if not results:
        print(f"Error: No benchmark results found in {args.input}", flush=True)
        return 1

    aggregated = aggregate_results(results)
    matrix = build_robustness_matrix(aggregated)
    report = build_programmatic_report(results, aggregated=aggregated, matrix=matrix)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        text = render_benchmark_report_text(results, matrix=matrix, aggregated=aggregated)
        print(text)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWrote aggregated report.json to {out_dir}")

    return 0


def _benchmark_plan(args: argparse.Namespace) -> int:
    """Run a benchmark plan with multiple experiments."""
    from provenance.robustness.orchestration import (
        BenchmarkRunner,
        load_benchmark_plan,
        validate_benchmark_plan,
    )

    # Load plan
    try:
        plan = load_benchmark_plan(args.plan)
    except Exception as exc:
        print(f"Error loading plan: {exc}", flush=True)
        return 1

    # Validate
    errors = validate_benchmark_plan(plan)
    if errors:
        print("Validation errors:", flush=True)
        for e in errors:
            loc = f" (spec[{e.spec_index}])" if e.spec_index is not None else ""
            print(f"  {e.field}{loc}: {e.message}", flush=True)
        return 1

    # Determine output directory
    out_dir = args.out_dir or Path(plan.output_dir or f"data/benchmarks/{plan.name}")

    # Dry run: print experiments and exit
    if args.dry_run:
        print("Benchmark Plan")
        print("=" * 25)
        print(f"Name: {plan.name}")
        print(f"Experiments: {len(plan.specs)}")
        print(f"Output: {out_dir}")
        print()
        for i, spec in enumerate(plan.specs, 1):
            lengths_str = ",".join(str(l) for l in spec.lengths)
            profile_str = spec.profile or "default"
            print(f"  {i}. {spec.detector} / {spec.config}")
            print(f"     lengths: {lengths_str}")
            print(f"     samples: {spec.samples}")
            print(f"     seed: {spec.seed}")
            print(f"     profile: {profile_str}")
            print(f"     experiment_id: {spec.experiment_id}")
            print()
        return 0

    # Run
    print(f"Running benchmark plan: {plan.name}", flush=True)
    print(f"Experiments: {len(plan.specs)}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    runner = BenchmarkRunner(
        plan=plan,
        resume=args.resume,
        force=args.force,
        experiment_fn=_plan_experiment_fn,
    )
    try:
        manifest = runner.run(out_dir)
    except ValueError as exc:
        # Resume plan mismatch (existing manifest belongs to another plan).
        print(f"Error: {exc}", flush=True)
        return 1

    # Summary
    completed = len(manifest.completed_experiments)
    failed = len(manifest.failed_experiments)
    total = len(manifest.experiments)

    print(f"\nCompleted: {completed}/{total}", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)
        for e in manifest.failed_experiments:
            print(f"  {e.experiment_id}: {e.error_type}: {e.error_message}", flush=True)

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"\nManifest: {out_dir / 'manifest.json'}")
        print(f"Status: {manifest.status}")

    return 0 if manifest.is_complete else 1


def _generate_watermark_samples(
    *,
    detector_name: str,
    config_path: str,
    lengths: tuple[int, ...],
    samples_per_length: int,
    seed: int,
) -> list | None:
    """Generate watermarked samples using the existing generation pipeline.

    Returns None if the config format doesn't support generation (e.g.
    simulation configs with simple tokenizer).
    """
    from provenance.benchmark.runner import derive_seed, DEFAULT_PROMPTS
    from provenance.robustness.experiments import WatermarkSample
    from provenance.configuration import ExperimentConfig as ExpConfig
    from provenance.tokenizers import tokenizer_from_config

    try:
        experiment = ExpConfig.from_file(config_path)
    except (AttributeError, KeyError, TypeError) as exc:
        print(
            f"Error: Config file {config_path} does not match the expected format for watermark generation.\n"
            f"Watermark generation requires an HF experiment config (with 'model' and 'tokenizer' sections).\n"
            f"Simulation configs (simple tokenizer) cannot generate watermarked samples.\n"
            f"Use --text <file> for text-file mode, or provide an HF experiment config.\n"
            f"Details: {exc}",
            flush=True,
        )
        return None
    tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())

    samples = []
    for length in lengths:
        for i in range(samples_per_length):
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            sample_seed = derive_seed(seed, length, i)

            if detector_name in ("kgw", "kgw-reference"):
                text = _generate_kgw_sample(
                    config_path=config_path,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    length=length,
                    seed=sample_seed,
                    watermarked=True,
                )
            elif detector_name in ("synthid", "synthid-reference"):
                text = _generate_synthid_sample(
                    config_path=config_path,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    length=length,
                    seed=sample_seed,
                    watermarked=True,
                )
            else:
                raise ValueError(f"Cannot generate samples for detector: {detector_name}")

            samples.append(WatermarkSample(
                text=text,
                token_count=len(tokenizer.encode(text)),
                sample_seed=sample_seed,
                prompt_id=f"prompt-{i % len(DEFAULT_PROMPTS)}",
                length=length,
                watermarked=True,
            ))

    return samples


def _generate_kgw_sample(
    *,
    config_path: str,
    tokenizer: Any,
    prompt: str,
    length: int,
    seed: int,
    watermarked: bool,
) -> str:
    """Generate a KGW watermarked text sample."""
    from provenance.detectors.reference.kgw_markllm import (
        KGWMarkLLMConfig,
        KGWMarkLLMScorer,
        classify_backend,
    )
    from provenance.generation import KGWGenerationConfig, generate_kgw_token_ids
    from provenance.models import HuggingFaceCausalLM
    from provenance.configuration import ExperimentConfig as ExpConfig

    experiment = ExpConfig.from_file(config_path)
    model = HuggingFaceCausalLM(experiment.model)

    kgw_config = KGWMarkLLMConfig.from_dict({
        "configuration_id": experiment.configuration_id,
        "version": experiment.version,
        **experiment.watermark_configuration,
        **experiment.detector_configuration,
    })
    scorer = KGWMarkLLMScorer(kgw_config, vocab_size=tokenizer.vocabulary_size)
    prompt_token_ids = tokenizer.encode(prompt)
    gen_config = KGWGenerationConfig(
        max_new_tokens=length,
        temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
        top_p=experiment.generation_configuration.get("top_p"),
        top_k=experiment.generation_configuration.get("top_k"),
        seed=seed,
    )
    token_ids = generate_kgw_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        scorer=scorer,
        generation_config=gen_config,
        watermarked=watermarked,
    )
    return tokenizer.decode(token_ids)


def _generate_synthid_sample(
    *,
    config_path: str,
    tokenizer: Any,
    prompt: str,
    length: int,
    seed: int,
    watermarked: bool,
) -> str:
    """Generate a SynthID watermarked text sample."""
    from provenance.detectors.reference.synthid import (
        SynthIDReferenceConfig,
        SynthIDReferenceDetector,
    )
    from provenance.generation import SynthIDGenerationConfig, SynthIDLogitsProcessor, generate_synthid_token_ids
    from provenance.models import HuggingFaceCausalLM
    from provenance.configuration import ExperimentConfig as ExpConfig

    experiment = ExpConfig.from_file(config_path)
    model = HuggingFaceCausalLM(experiment.model)

    synthid_config = SynthIDReferenceConfig.from_dict({
        "configuration_id": experiment.configuration_id,
        "version": experiment.version,
        **experiment.watermark_configuration,
        **experiment.detector_configuration,
        "tokenizer": "huggingface",
        "model_identifier": experiment.model.model_identifier,
        "model_revision": experiment.model.revision,
        "tokenizer_identifier": experiment.tokenizer.tokenizer_identifier,
        "tokenizer_revision": experiment.tokenizer.tokenizer_revision,
        **{k: v for k, v in experiment.generation_configuration.items()
           if k in ("max_new_tokens", "temperature", "top_p", "top_k", "seed")},
    })
    synthid_config.validate()

    processor = SynthIDLogitsProcessor(synthid_config)
    prompt_token_ids = tokenizer.encode(prompt)
    gen_config = SynthIDGenerationConfig(
        max_new_tokens=length,
        temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
        top_p=experiment.generation_configuration.get("top_p"),
        top_k=experiment.generation_configuration.get("top_k"),
        seed=seed,
    )
    token_ids = generate_synthid_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        processor=processor,
        generation_config=gen_config,
        watermarked=watermarked,
    )
    return tokenizer.decode(token_ids)


def _robustness_records_to_evaluation_records(records, config):
    """Convert RobustnessRecords to EvaluationRecords for JSONL persistence."""
    from provenance.benchmark.records import EvaluationRecord, BENCHMARK_VERSION
    from datetime import datetime, timezone

    eval_records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = f"robustness:{config.detector_name}:{config.seed}"

    for r in records:
        wm = r.metadata.get("watermarked", True)
        eval_records.append(EvaluationRecord(
            experiment_id=experiment_id,
            timestamp=timestamp,
            benchmark_version=BENCHMARK_VERSION,
            scheme=config.detector_name.split("-")[0],
            variant=r.implementation_kind,
            configuration_id=config.config_path or "unknown",
            configuration_version="1",
            implementation_kind=r.implementation_kind,
            compatibility=r.compatibility,
            model_identifier="unknown",
            model_revision=None,
            tokenizer_identifier="unknown",
            tokenizer_revision=None,
            vocab_size=0,
            hash_key_id="robustness-experiment",
            temperature=0.0,
            top_p=None,
            top_k=None,
            random_seed=r.metadata.get("sample_seed", 0),
            prompt_id=r.metadata.get("prompt_id", "unknown"),
            target_length=r.metadata.get("length", 0),
            watermarked=wm,
            token_count=0,
            scored_token_count=0,
            detection_threshold=0.0,
            detected=r.original_detected is True,
            score=r.original_score or 0.0,
        ))

    return eval_records


def _analyze(args: argparse.Namespace) -> int:
    text = args.path.read_text(encoding="utf-8")
    detector_names = args.detector or ["unicode"]
    include_unicode = "unicode" in detector_names
    detectors = []
    extra_results: list[DetectionResult] = []

    for detector_name in detector_names:
        if detector_name == "unicode":
            continue
        if args.config is None:
            extra_results.append(_unconfigured_result(detector_name))
            continue
        if detector_name == "kgw":
            detectors.append(KGWDetector.from_config_file(args.config))
        elif detector_name == "kgw-reference":
            detectors.append(KGWReferenceDetector.from_config_file(args.config))
        elif detector_name == "synthid":
            detectors.append(SynthIDTextDetector.from_config_file(args.config))
        elif detector_name == "synthid-reference":
            detectors.append(SynthIDReferenceDetector.from_config_file(args.config))

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

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(_human_output(result))
    return 0


def _human_output(result) -> str:
    lines = [
        f"Engine: {result.engine_version}",
        f"Characters: {result.text_stats['character_count']}",
        f"Tokens: {result.text_stats['token_count']}",
        "",
        "Detector results:",
    ]
    for detector_result in result.results:
        detected = "unavailable" if detector_result.detected is None else str(detector_result.detected).lower()
        lines.append(
            f"- {detector_result.detector} [{detector_result.status}]: "
            f"implementation={detector_result.implementation_kind}, "
            f"compatibility={detector_result.compatibility}, "
            f"detected={detected}, score={detector_result.score}, "
            f"threshold={detector_result.threshold}, confidence={detector_result.confidence}"
        )
        if detector_result.detector == "unicode":
            findings = detector_result.evidence.get("findings", [])
            for finding in findings[:10]:
                lines.append(
                    "  "
                    f"{finding['position']}: {finding['codepoint']} "
                    f"{finding['name']} ({finding['category']}, {finding['severity']})"
                )
            if len(findings) > 10:
                lines.append(f"  ... {len(findings) - 10} more findings")
    lines.append("")
    lines.append("No global AI-generated or human-authored classification is produced.")
    return "\n".join(lines)
