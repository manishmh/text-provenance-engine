"""Development CLI for local text provenance analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from provenance.detectors import KGWDetector, KGWReferenceDetector, SynthIDTextDetector
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
        choices=["unicode", "kgw", "kgw-reference", "synthid"],
        help="Detector to run. Repeat to run multiple detectors. Defaults to unicode.",
    )
    analyze.add_argument(
        "--config",
        type=Path,
        help="Configuration JSON for a selected KGW or SynthID detector",
    )
    analyze.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    parser.error(f"unknown command: {args.command}")
    return 2


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
