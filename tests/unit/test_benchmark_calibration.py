"""Unit tests for benchmark calibration: Wilson intervals and ROC/AUC.

Torch-free: imports benchmark submodules directly with small synthetic records.
"""

from __future__ import annotations

import json

import pytest

from provenance.benchmark.calibration import (
    ROCPoint,
    calibrated_threshold,
    rate_estimate,
    roc_auc,
    roc_curve,
    roc_summary,
    wilson_interval,
)
from provenance.benchmark.records import BENCHMARK_VERSION, EvaluationRecord
from provenance.benchmark.report import build_report, render_text_report
from provenance.benchmark.statistics import threshold_analysis


def make_record(*, watermarked: bool, z_score: float, target_length: int, **overrides) -> EvaluationRecord:
    """Minimal synthetic record for statistics/calibration tests (no model)."""
    base = dict(
        experiment_id="exp-1",
        timestamp="2026-08-19T00:00:00+00:00",
        benchmark_version=BENCHMARK_VERSION,
        scheme="kgw",
        variant="kgw-markllm-v1",
        configuration_id="kgw-markllm-distilgpt2-v1",
        configuration_version="1",
        implementation_kind="reference",
        compatibility="MarkLLM-compatible",
        model_identifier="distilgpt2",
        model_revision="abc123",
        tokenizer_identifier="distilgpt2",
        tokenizer_revision="abc123",
        vocab_size=50257,
        hash_key_id="kgw-demo-key-v1",
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        random_seed=42,
        prompt_id="prompt-0",
        target_length=target_length,
        watermarked=watermarked,
        token_count=target_length,
        scored_token_count=target_length - 1,
        detection_threshold=4.0,
        detected=z_score >= 4.0,
        score=z_score,
        gamma=0.25,
        delta=2.0,
        prefix_length=1,
        window_scheme="left",
        seeding_scheme="left-hash/additive",
        f_scheme="additive",
        green_token_count=int((target_length - 1) * (0.6 if watermarked else 0.25)),
        green_fraction=0.6 if watermarked else 0.25,
        expected_green_fraction=0.25,
        z_score=z_score,
        p_value=0.5,
    )
    base.update(overrides)
    return EvaluationRecord(**base)


# --- Wilson score interval -------------------------------------------------


def test_wilson_zero_successes_interval_is_above_zero():
    # 0/20: observed rate 0, but the true rate is not proven to be zero.
    low, high = wilson_interval(0, 20)
    assert low == pytest.approx(0.0)
    assert 0.15 < high < 0.17  # ~0.161 at 95%


def test_wilson_all_successes_interval_below_one():
    low, high = wilson_interval(20, 20)
    assert high == pytest.approx(1.0)
    assert 0.83 < low < 0.85  # ~0.839 at 95%


def test_wilson_symmetric_midpoint():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    # Wilson center for phat=0.5 is exactly 0.5, so the interval is symmetric.
    assert (0.5 - low) == pytest.approx(high - 0.5)


def test_wilson_zero_n_is_degenerate():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_rejects_out_of_range():
    with pytest.raises(ValueError):
        wilson_interval(21, 20)


def test_wilson_interval_narrows_with_n():
    _, high_small = wilson_interval(0, 10)
    _, high_large = wilson_interval(0, 100)
    assert high_large < high_small  # more samples -> tighter bound


def test_rate_estimate_fields():
    est = rate_estimate(3, 4)
    assert est.rate == pytest.approx(0.75)
    assert est.successes == 3
    assert est.n_samples == 4
    assert est.ci_low < 0.75 < est.ci_high
    assert est.confidence == 0.95


def test_rate_estimate_zero_n():
    est = rate_estimate(0, 0)
    assert est.rate == 0.0
    assert (est.ci_low, est.ci_high) == (0.0, 0.0)


# --- ROC / AUC -------------------------------------------------------------


def _roc_records():
    # Perfectly separable: all watermarked z above all unwatermarked z.
    recs = [make_record(watermarked=True, z_score=z, target_length=100) for z in (5.0, 6.0, 7.0)]
    recs += [make_record(watermarked=False, z_score=z, target_length=100) for z in (-1.0, 0.0, 1.0)]
    return recs


def test_auc_perfect_separation():
    assert roc_auc(_roc_records()) == pytest.approx(1.0)


def test_auc_chance_when_identical():
    recs = [make_record(watermarked=True, z_score=1.0, target_length=100) for _ in range(3)]
    recs += [make_record(watermarked=False, z_score=1.0, target_length=100) for _ in range(3)]
    # All ties -> 0.5.
    assert roc_auc(recs) == pytest.approx(0.5)


def test_auc_partial_overlap():
    # wm z: [2, 4]; un z: [1, 3]. Pairs: (2>1)y (2>3)n (4>1)y (4>3)y => 3/4.
    recs = [
        make_record(watermarked=True, z_score=2.0, target_length=100),
        make_record(watermarked=True, z_score=4.0, target_length=100),
        make_record(watermarked=False, z_score=1.0, target_length=100),
        make_record(watermarked=False, z_score=3.0, target_length=100),
    ]
    assert roc_auc(recs) == pytest.approx(0.75)


def test_auc_undefined_without_both_conditions():
    only_wm = [make_record(watermarked=True, z_score=5.0, target_length=100)]
    assert roc_auc(only_wm) is None
    assert roc_curve(only_wm) == []
    assert roc_summary(only_wm)["auc"] is None


def test_roc_curve_endpoints_and_monotonicity():
    points = roc_curve(_roc_records())
    assert points[0] == ROCPoint(threshold=None, fpr=0.0, tpr=0.0)
    assert points[-1].fpr == pytest.approx(1.0)
    assert points[-1].tpr == pytest.approx(1.0)
    # ROC is monotone non-decreasing in both coordinates as the threshold drops.
    for a, b in zip(points, points[1:]):
        assert b.fpr >= a.fpr - 1e-12
        assert b.tpr >= a.tpr - 1e-12


def test_roc_curve_perfect_reaches_top_left_corner():
    # Perfect separation must pass through (fpr=0, tpr=1) at some threshold.
    points = roc_curve(_roc_records())
    assert any(p.fpr == pytest.approx(0.0) and p.tpr == pytest.approx(1.0) for p in points)


# --- calibrated threshold + report integration -----------------------------


def test_calibrated_threshold_recovers_counts_and_ci():
    recs = _roc_records()  # 3 wm, 3 un
    point = threshold_analysis(recs, [2.0])[0]  # all 3 wm >= 2, 0 un >= 2
    enriched = calibrated_threshold(point)
    assert enriched["tpr"] == pytest.approx(1.0)
    assert enriched["tpr_successes"] == 3
    assert enriched["fpr"] == pytest.approx(0.0)
    assert enriched["fpr_successes"] == 0
    # 0/3 FPR still has a non-zero upper bound.
    assert enriched["fpr_ci_high"] > 0.0
    assert enriched["fpr_ci_low"] == pytest.approx(0.0)
    assert enriched["n_watermarked"] == 3
    assert enriched["n_unwatermarked"] == 3


def test_build_report_includes_calibration_and_roc():
    recs = _roc_records()
    report = build_report(recs, thresholds=[2.0, 4.0])
    overall = report["overall"]
    assert overall["roc"]["auc"] == pytest.approx(1.0)
    assert overall["roc"]["points"]
    point = overall["thresholds"][0]
    for key in ("tpr_ci_low", "tpr_ci_high", "fpr_ci_low", "fpr_ci_high", "tpr_successes"):
        assert key in point
    # Per-length ROC present too.
    assert report["by_length"][0]["roc"]["auc"] == pytest.approx(1.0)
    # Report stays JSON-serializable (no inf thresholds leaking through).
    dumped = json.dumps(report)
    assert "Infinity" not in dumped


def test_text_report_shows_ci_and_auc():
    recs = _roc_records()
    report = build_report(recs, thresholds=[2.0, 4.0], reproducibility={"gamma": 0.25})
    text = render_text_report(report)
    assert "95% Wilson CI" in text
    assert "Pooled ROC AUC" in text
    assert "AUC" in text  # column header in the length table
