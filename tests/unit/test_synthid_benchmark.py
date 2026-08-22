"""Unit tests for SynthID benchmark records and statistics.

Torch-free: uses synthetic EvaluationRecord fixtures with scheme="synthid".
"""

from __future__ import annotations

import json

import pytest

from provenance.benchmark.calibration import (
    rate_estimate,
    roc_auc,
    roc_curve,
    wilson_interval,
)
from provenance.benchmark.records import BENCHMARK_VERSION, EvaluationRecord, read_jsonl, write_jsonl
from provenance.benchmark.report import build_report, render_text_report
from provenance.benchmark.statistics import (
    false_positive_rate,
    length_effect,
    observed_lengths,
    summarize_group,
    threshold_analysis,
    true_positive_rate,
)


def make_synthid_record(*, watermarked: bool, score_val: float, target_length: int, **overrides):
    """Create a synthetic SynthID EvaluationRecord."""
    base = dict(
        experiment_id="synthid-exp-1",
        timestamp="2026-08-22T00:00:00+00:00",
        benchmark_version=BENCHMARK_VERSION,
        scheme="synthid",
        variant="synthid-reference-v1",
        configuration_id="synthid-test-v1",
        configuration_version="1",
        implementation_kind="reference",
        compatibility="synthid-reference",
        model_identifier="distilgpt2",
        model_revision="abc123",
        tokenizer_identifier="distilgpt2",
        tokenizer_revision="abc123",
        vocab_size=50257,
        hash_key_id="synthid-test-key-v1",
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        random_seed=42,
        prompt_id="prompt-0",
        target_length=target_length,
        watermarked=watermarked,
        token_count=target_length,
        scored_token_count=target_length - 5,
        detection_threshold=0.5,
        detected=score_val >= 0.5,
        score=score_val,
        ngram_len=5,
        watermarking_depth=5,
    )
    base.update(overrides)
    return EvaluationRecord(**base)


# --- Record serialization ---


def test_synthid_record_round_trip():
    record = make_synthid_record(watermarked=True, score_val=0.7, target_length=100)
    assert EvaluationRecord.from_dict(record.to_dict()) == record


def test_synthid_record_has_no_kgw_fields_by_default():
    record = make_synthid_record(watermarked=True, score_val=0.7, target_length=100)
    assert record.gamma is None
    assert record.delta is None
    assert record.green_token_count is None
    assert record.green_fraction is None
    assert record.ngram_len == 5
    assert record.watermarking_depth == 5


def test_synthid_record_no_raw_key():
    record = make_synthid_record(watermarked=True, score_val=0.7, target_length=100)
    result_json = json.dumps(record.to_dict())
    assert "synthid-test-key-v1" in result_json  # hash_key_id present
    assert record.hash_key_id == "synthid-test-key-v1"


# --- Statistics ---


def test_synthid_summarize_group():
    records = [
        make_synthid_record(watermarked=True, score_val=0.6, target_length=50),
        make_synthid_record(watermarked=True, score_val=0.7, target_length=50),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=50),
        make_synthid_record(watermarked=False, score_val=0.4, target_length=50),
    ]
    wm_stats = summarize_group(records, watermarked=True, target_length=50)
    assert wm_stats.n_samples == 2
    assert wm_stats.mean_score == pytest.approx(0.65)
    assert wm_stats.detection_rate == pytest.approx(1.0)  # both >= 0.5

    un_stats = summarize_group(records, watermarked=False, target_length=50)
    assert un_stats.n_samples == 2
    assert un_stats.detection_rate == pytest.approx(0.0)  # both < 0.5


def test_synthid_tpr_fpr():
    records = [
        make_synthid_record(watermarked=True, score_val=0.6, target_length=100),
        make_synthid_record(watermarked=True, score_val=0.7, target_length=100),
        make_synthid_record(watermarked=True, score_val=0.4, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.45, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.55, target_length=100),
    ]
    # At threshold 0.5: 2/3 watermarked detected, 1/3 unwatermarked detected
    assert true_positive_rate(records, 0.5) == pytest.approx(2 / 3)
    assert false_positive_rate(records, 0.5) == pytest.approx(1 / 3)

    # At threshold 0.6: 2/3 watermarked (0.6, 0.7 >= 0.6), 0/3 unwatermarked
    assert true_positive_rate(records, 0.6) == pytest.approx(2 / 3)
    assert false_positive_rate(records, 0.6) == pytest.approx(0.0)


def test_synthid_threshold_analysis():
    records = [
        make_synthid_record(watermarked=True, score_val=0.6, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
    ]
    points = threshold_analysis(records, [0.5, 0.6])
    assert len(points) == 2
    assert points[0].tpr == pytest.approx(1.0)
    assert points[0].fpr == pytest.approx(0.0)


def test_synthid_roc_auc_perfect():
    records = [
        make_synthid_record(watermarked=True, score_val=0.8, target_length=100),
        make_synthid_record(watermarked=True, score_val=0.9, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.2, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
    ]
    assert roc_auc(records) == pytest.approx(1.0)


def test_synthid_roc_auc_chance():
    records = [
        make_synthid_record(watermarked=True, score_val=0.5, target_length=100),
        make_synthid_record(watermarked=True, score_val=0.5, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.5, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.5, target_length=100),
    ]
    assert roc_auc(records) == pytest.approx(0.5)


def test_synthid_length_effect():
    records = [
        make_synthid_record(watermarked=True, score_val=0.6, target_length=50),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=50),
        make_synthid_record(watermarked=True, score_val=0.8, target_length=200),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=200),
    ]
    rows = {row.target_length: row for row in length_effect(records)}
    assert rows[50].mean_score_watermarked == pytest.approx(0.6)
    assert rows[200].mean_score_watermarked == pytest.approx(0.8)


# --- Report ---


def test_synthid_report():
    records = [
        make_synthid_record(watermarked=True, score_val=0.7, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
    ]
    report = build_report(records, thresholds=[0.5, 0.6])
    assert report["benchmark_version"] == BENCHMARK_VERSION
    assert "SynthID-Text" in report["meaning"]
    assert report["sample_counts"]["total"] == 2


def test_synthid_text_report():
    records = [
        make_synthid_record(watermarked=True, score_val=0.7, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
    ]
    report = build_report(records, thresholds=[0.5], reproducibility={"scheme": "synthid"})
    text = render_text_report(report)
    assert "SynthID-Text Watermark Benchmark Report" in text
    assert "Limitations" in text


def test_synthid_report_json_serializable():
    records = [
        make_synthid_record(watermarked=True, score_val=0.7, target_length=100),
        make_synthid_record(watermarked=False, score_val=0.3, target_length=100),
    ]
    report = build_report(records, thresholds=[0.5])
    dumped = json.dumps(report)
    assert "Infinity" not in dumped


# --- No secret leakage ---


def test_synthid_no_raw_key_in_report():
    records = [
        make_synthid_record(watermarked=True, score_val=0.7, target_length=100),
    ]
    report = build_report(records, reproducibility={"hash_key_id": "synthid-test-key-v1"})
    report_json = json.dumps(report)
    # The hash_key_id should be present as an identifier
    assert "synthid-test-key-v1" in report_json


# --- Wilson CI ---


def test_synthid_wilson_ci():
    est = rate_estimate(3, 4)
    assert est.rate == pytest.approx(0.75)
    assert est.ci_low < 0.75 < est.ci_high
