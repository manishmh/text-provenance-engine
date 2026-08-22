"""Aggregate KGW benchmark records into a JSON report and a text summary.

The JSON report is structured for later API/dashboard consumption; the text
report is a concise human-readable summary. Neither makes any AI-authorship
claim -- both state explicitly that this evaluates a *known* KGW watermark
configuration.
"""

from __future__ import annotations

from typing import Any, Sequence

from provenance.benchmark.calibration import calibrated_threshold, roc_summary
from provenance.benchmark.records import BENCHMARK_VERSION, EvaluationRecord
from provenance.benchmark.statistics import (
    length_effect,
    observed_lengths,
    summarize_group,
    threshold_analysis,
)

DEFAULT_THRESHOLDS: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)

REPORT_LIMITATIONS = [
    "This benchmark evaluates a single, known watermark configuration. A "
    "positive result is evidence consistent with THAT configuration only.",
    "It does not measure general AI-vs-human detection and does not attribute "
    "text to any model provider (ChatGPT/Claude/Gemini or otherwise).",
    "TPR/FPR are estimated from a finite number of samples; the standard error "
    "on a rate scales roughly as 1/sqrt(n_samples).",
    "Short texts carry less watermark signal, so detection rates at small token "
    "lengths are expected to be lower and are reported as measured.",
    "No universally-correct threshold exists; the threshold table exists to "
    "choose a calibrated operating point per deployment, not to assert one.",
    "An observed FPR of 0 does NOT mean the true false-positive rate is zero; "
    "read it together with the reported 95% Wilson confidence interval, which "
    "stays above zero for any finite sample.",
    "AUC summarizes watermarked-vs-unwatermarked score separation across all "
    "thresholds for this configuration only; it is not an AI-detection accuracy.",
]


def build_report(
    records: Sequence[EvaluationRecord],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    reproducibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable benchmark report from evaluation records."""
    lengths = observed_lengths(records)
    n_watermarked = sum(1 for r in records if r.watermarked)
    n_unwatermarked = len(records) - n_watermarked

    per_length: list[dict[str, Any]] = []
    for length in lengths:
        per_length.append(
            {
                "target_length": length,
                "watermarked": summarize_group(
                    records, watermarked=True, target_length=length
                ).to_dict(),
                "unwatermarked": summarize_group(
                    records, watermarked=False, target_length=length
                ).to_dict(),
                "thresholds": [
                    calibrated_threshold(p)
                    for p in threshold_analysis(records, thresholds, target_length=length)
                ],
                "roc": roc_summary(records, target_length=length),
            }
        )

    overall = {
        "watermarked": summarize_group(records, watermarked=True).to_dict(),
        "unwatermarked": summarize_group(records, watermarked=False).to_dict(),
        "thresholds": [
            calibrated_threshold(p) for p in threshold_analysis(records, thresholds)
        ],
        "roc": roc_summary(records),
    }

    # Detect scheme from records
    scheme = records[0].scheme if records else "unknown"
    scheme_label = {
        "kgw": "KGW",
        "synthid": "SynthID-Text",
    }.get(scheme, scheme)

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "meaning": (
            f"Evaluation of a known {scheme_label} watermark configuration. "
            "Not a general AI-detection or model-attribution benchmark."
        ),
        "reproducibility": reproducibility or {},
        "sample_counts": {
            "total": len(records),
            "watermarked": n_watermarked,
            "unwatermarked": n_unwatermarked,
            "lengths": lengths,
        },
        "thresholds": list(thresholds),
        "by_length": per_length,
        "overall": overall,
        "length_effect": [row.to_dict() for row in length_effect(records)],
        "limitations": REPORT_LIMITATIONS,
    }


def _fmt(value: float, width: int = 0, precision: int = 3) -> str:
    return f"{value:>{width}.{precision}f}" if width else f"{value:.{precision}f}"


def _threshold_line(point: dict[str, Any]) -> str:
    """Format one enriched threshold point with observed rate, CI and sample n.

    Falls back gracefully if CI keys are absent (e.g. a raw ThresholdPoint dict).
    """
    tpr_lo = point.get("tpr_ci_low")
    tpr_hi = point.get("tpr_ci_high")
    fpr_lo = point.get("fpr_ci_low")
    fpr_hi = point.get("fpr_ci_high")
    tpr_ci = (
        f"[{_fmt(tpr_lo)}, {_fmt(tpr_hi)}]" if tpr_lo is not None else "[      n/a     ]"
    )
    fpr_ci = (
        f"[{_fmt(fpr_lo)}, {_fmt(fpr_hi)}]" if fpr_lo is not None else "[      n/a     ]"
    )
    return (
        f"{point['threshold']:>9} | "
        f"{_fmt(point['tpr'], 6)} {tpr_ci} {point.get('n_watermarked', 0):>4} | "
        f"{_fmt(point['fpr'], 6)} {fpr_ci} {point.get('n_unwatermarked', 0):>4}"
    )


def render_text_report(report: dict[str, Any]) -> str:
    """Render a concise human-readable report from :func:`build_report` output."""
    lines: list[str] = []
    repro = report.get("reproducibility", {})
    counts = report["sample_counts"]

    scheme = report.get("reproducibility", {}).get("scheme", "kgw")
    scheme_label = {"kgw": "KGW", "synthid": "SynthID-Text"}.get(scheme, scheme.upper())
    lines.append(f"{scheme_label} Watermark Benchmark Report")
    lines.append("=" * 40)
    lines.append(report["meaning"])
    lines.append("")

    lines.append("Experiment configuration")
    lines.append("-" * 24)
    # Show scheme-relevant config keys
    config_keys = [
        "benchmark_version",
        "variant",
        "configuration_id",
        "implementation_kind",
        "compatibility",
        "model_identifier",
        "model_revision",
        "tokenizer_identifier",
        "tokenizer_revision",
        "hash_key_id",
        "temperature",
        "top_p",
        "top_k",
        "seed",
    ]
    if scheme == "kgw":
        config_keys.extend(["gamma", "delta", "prefix_length", "window_scheme", "seeding_scheme", "f_scheme"])
    elif scheme == "synthid":
        config_keys.extend(["ngram_len", "watermarking_depth"])
    for key in config_keys:
        if key in repro:
            lines.append(f"  {key}: {repro[key]}")
    lines.append("")

    lines.append("Sample counts")
    lines.append("-" * 13)
    lines.append(
        f"  total={counts['total']}  watermarked={counts['watermarked']}  "
        f"unwatermarked={counts['unwatermarked']}  lengths={counts['lengths']}"
    )
    lines.append("")

    lines.append("Results by token length")
    lines.append("-" * 23)
    if scheme == "kgw":
        lines.append(
            "  length | n(wm/un) | wm mean_score | wm det_rate | un mean_score | un FPR |   AUC | wm green_frac"
        )
    else:
        lines.append(
            "  length | n(wm/un) | wm mean_score | wm det_rate | un mean_score | un FPR |   AUC"
        )
    roc_by_length = {b["target_length"]: b.get("roc", {}) for b in report["by_length"]}
    for row in report["length_effect"]:
        by_len = next(
            b for b in report["by_length"] if b["target_length"] == row["target_length"]
        )
        wm = by_len["watermarked"]
        auc = roc_by_length.get(row["target_length"], {}).get("auc")
        auc_txt = "  n/a" if auc is None else _fmt(auc, 5)
        line = (
            f"  {row['target_length']:>6} | "
            f"{row['n_watermarked']:>3}/{row['n_unwatermarked']:<3} | "
            f"{_fmt(row['mean_score_watermarked'], 12)} | "
            f"{_fmt(row['watermarked_detection_rate'], 11)} | "
            f"{_fmt(row['mean_score_unwatermarked'], 12)} | "
            f"{_fmt(row['unwatermarked_fpr'], 6)} | "
            f"{auc_txt}"
        )
        if scheme == "kgw":
            expected_green = repro.get("gamma")
            green = _fmt(wm["mean_green_fraction"], 8) if wm.get("mean_green_fraction") is not None else "    n/a"
            if expected_green is not None and wm.get("mean_green_fraction") is not None:
                green = f"{green} (exp {expected_green:.3f})"
            line += f" | {green}"
        lines.append(line)
    lines.append("")

    overall_auc = report["overall"].get("roc", {}).get("auc")
    if overall_auc is not None:
        lines.append(f"Pooled ROC AUC: {overall_auc:.4f}  (1.0 = perfect separation, 0.5 = chance)")
        lines.append("")

    score_label = "score" if scheme != "kgw" else "z"
    lines.append(f"TPR / FPR by {score_label}-threshold (all lengths pooled, 95% Wilson CI)")
    lines.append("-" * 59)
    lines.append(
        "  threshold |    TPR [   95% CI    ]  n(wm) |    FPR [   95% CI    ]  n(un)"
    )
    for point in report["overall"]["thresholds"]:
        lines.append("  " + _threshold_line(point))
    lines.append("")

    lines.append(f"TPR / FPR by {score_label}-threshold and length (95% Wilson CI)")
    lines.append("-" * 51)
    lines.append(
        "  length | threshold |    TPR [   95% CI    ]  n(wm) |    FPR [   95% CI    ]  n(un)"
    )
    for by_len in report["by_length"]:
        for point in by_len["thresholds"]:
            lines.append(
                f"  {by_len['target_length']:>6} | " + _threshold_line(point)
            )
    lines.append("")

    lines.append("Limitations")
    lines.append("-" * 11)
    for limitation in report["limitations"]:
        lines.append(f"  - {limitation}")

    return "\n".join(lines)
