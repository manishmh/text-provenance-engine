"""Scheme-agnostic provenance evaluation and benchmarking subsystem.

Layers:
- ``records``: the flat, JSON-serializable :class:`EvaluationRecord` data model.
- ``statistics``: pure TPR/FPR/threshold/length aggregation over records.
- ``report``: build a JSON report and render a human-readable summary.
- ``runner``: model-backed generation of watermarked/unwatermarked samples.

Currently supports KGW and SynthID-Text watermark schemes.
"""

from provenance.benchmark.calibration import (
    RateEstimate,
    ROCPoint,
    calibrated_threshold,
    rate_estimate,
    roc_auc,
    roc_curve,
    roc_summary,
    wilson_interval,
)
from provenance.benchmark.records import (
    BENCHMARK_VERSION,
    EvaluationRecord,
    read_jsonl,
    write_jsonl,
)
from provenance.benchmark.report import (
    DEFAULT_THRESHOLDS,
    build_report,
    render_text_report,
)
from provenance.benchmark.runner import (
    BenchmarkSpec,
    derive_seed,
    generate_records,
    generate_synthid_records,
    run_benchmark,
    run_synthid_benchmark,
)
from provenance.benchmark.statistics import (
    GroupStatistics,
    LengthEffectRow,
    ThresholdPoint,
    false_positive_rate,
    length_effect,
    summarize_group,
    threshold_analysis,
    true_positive_rate,
)

__all__ = [
    "BENCHMARK_VERSION",
    "BenchmarkSpec",
    "DEFAULT_THRESHOLDS",
    "EvaluationRecord",
    "GroupStatistics",
    "LengthEffectRow",
    "RateEstimate",
    "ROCPoint",
    "ThresholdPoint",
    "build_report",
    "calibrated_threshold",
    "derive_seed",
    "false_positive_rate",
    "generate_records",
    "generate_synthid_records",
    "length_effect",
    "rate_estimate",
    "read_jsonl",
    "render_text_report",
    "roc_auc",
    "roc_curve",
    "roc_summary",
    "run_benchmark",
    "run_synthid_benchmark",
    "summarize_group",
    "threshold_analysis",
    "true_positive_rate",
    "wilson_interval",
    "write_jsonl",
]
