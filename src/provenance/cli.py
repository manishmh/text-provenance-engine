"""Development CLI for local text provenance analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
        parser.error(f"unknown benchmark: {args.benchmark_command}")
    parser.error(f"unknown command: {args.command}")
    return 2


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part) for part in raw.split(",") if part.strip())


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
