"""Unit tests for the KGW benchmark subsystem.

These use small synthetic EvaluationRecord fixtures and never load a model. They
import benchmark submodules directly (records/statistics/report) rather than the
package root, so they stay independent of the optional torch-backed runner.
"""

from __future__ import annotations

import json

import pytest

from provenance.benchmark.records import (
    BENCHMARK_VERSION,
    EvaluationRecord,
    read_jsonl,
    write_jsonl,
)
from provenance.benchmark.report import build_report, render_text_report
from provenance.benchmark.runner import derive_seed
from provenance.benchmark.statistics import (
    false_positive_rate,
    length_effect,
    observed_lengths,
    summarize_group,
    threshold_analysis,
    true_positive_rate,
)


def make_record(*, watermarked: bool, z_score: float, target_length: int, **overrides):
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


@pytest.fixture
def records():
    # Two lengths. Watermarked z-scores rise with length; unwatermarked ~0.
    out = []
    for length, wm_zs, un_zs in [
        (50, [3.0, 3.5, 5.0], [0.2, -0.5, 1.0]),
        (200, [8.0, 9.0, 10.0], [0.1, -0.2, 0.3]),
    ]:
        for z in wm_zs:
            out.append(make_record(watermarked=True, z_score=z, target_length=length))
        for z in un_zs:
            out.append(make_record(watermarked=False, z_score=z, target_length=length))
    return out


# --- schema / serialization ------------------------------------------------


def test_record_round_trip():
    record = make_record(watermarked=True, z_score=7.0, target_length=100)
    assert EvaluationRecord.from_dict(record.to_dict()) == record


def test_record_rejects_raw_secret_key():
    data = make_record(watermarked=True, z_score=7.0, target_length=100).to_dict()
    data["hash_key"] = 15485863
    with pytest.raises(ValueError, match="raw hash_key"):
        EvaluationRecord.from_dict(data)


def test_record_rejects_unknown_and_missing_fields():
    data = make_record(watermarked=True, z_score=7.0, target_length=100).to_dict()
    with pytest.raises(ValueError, match="unknown"):
        EvaluationRecord.from_dict({**data, "bogus": 1})
    # Pop a required field to trigger the missing-field error
    incomplete = dict(data)
    incomplete.pop("experiment_id")
    with pytest.raises(ValueError, match="missing"):
        EvaluationRecord.from_dict(incomplete)


def test_jsonl_round_trip(tmp_path, records):
    path = write_jsonl(records, tmp_path / "records.jsonl")
    loaded = read_jsonl(path)
    assert loaded == records


# --- statistics ------------------------------------------------------------


def test_observed_lengths(records):
    assert observed_lengths(records) == [50, 200]


def test_summarize_group_values(records):
    stats = summarize_group(records, watermarked=True, target_length=200)
    assert stats.n_samples == 3
    assert stats.min_score == 8.0
    assert stats.max_score == 10.0
    assert stats.mean_score == pytest.approx(9.0)
    assert stats.detection_rate == pytest.approx(1.0)  # all >= 4.0


def test_summarize_group_empty():
    stats = summarize_group([], watermarked=True)
    assert stats.n_samples == 0
    assert stats.mean_score == 0.0
    assert stats.detection_rate == 0.0


def test_tpr_and_fpr(records):
    # Watermarked z: 50-> [3.0,3.5,5.0]; 200-> [8,9,10]. At threshold 4: 4/6.
    assert true_positive_rate(records, 4.0) == pytest.approx(4 / 6)
    assert true_positive_rate(records, 5.0) == pytest.approx(4 / 6)
    assert true_positive_rate(records, 3.0) == pytest.approx(6 / 6)
    # Unwatermarked z max is 1.0 -> FPR is 0 at every threshold >= 2.
    assert false_positive_rate(records, 2.0) == pytest.approx(0.0)


def test_threshold_analysis_overall(records):
    points = {p.threshold: p for p in threshold_analysis(records, [2.0, 4.0, 5.0])}
    assert points[4.0].tpr == pytest.approx(4 / 6)
    assert points[4.0].fpr == pytest.approx(0.0)
    assert points[4.0].n_watermarked == 6
    assert points[4.0].n_unwatermarked == 6


def test_threshold_analysis_by_length(records):
    points = threshold_analysis(records, [4.0], target_length=50)
    assert points[0].n_watermarked == 3
    assert points[0].tpr == pytest.approx(1 / 3)  # only z=5.0 passes at length 50


def test_length_effect(records):
    rows = {row.target_length: row for row in length_effect(records)}
    assert rows[50].watermarked_detection_rate == pytest.approx(1 / 3)
    assert rows[200].watermarked_detection_rate == pytest.approx(1.0)
    assert rows[50].unwatermarked_fpr == pytest.approx(0.0)
    assert rows[200].mean_score_watermarked == pytest.approx(9.0)


# --- report ----------------------------------------------------------------


def test_build_report_structure(records):
    report = build_report(records, thresholds=[2.0, 3.0, 4.0, 5.0])
    assert report["benchmark_version"] == BENCHMARK_VERSION
    assert report["sample_counts"] == {
        "total": 12,
        "watermarked": 6,
        "unwatermarked": 6,
        "lengths": [50, 200],
    }
    assert [b["target_length"] for b in report["by_length"]] == [50, 200]
    assert len(report["overall"]["thresholds"]) == 4
    assert report["limitations"]


def test_report_never_leaks_raw_key(records):
    report = build_report(records, reproducibility={"hash_key_id": "kgw-demo-key-v1"})
    assert "15485863" not in json.dumps(report)


def test_render_text_report_smoke(records):
    report = build_report(records, reproducibility={"gamma": 0.25, "variant": "kgw-markllm-v1"})
    text = render_text_report(report)
    assert "KGW Watermark Benchmark Report" in text
    assert "TPR / FPR by z-threshold" in text
    assert "Limitations" in text


# --- reproducibility -------------------------------------------------------


def test_derive_seed_is_deterministic_and_length_disjoint():
    assert derive_seed(42, 50, 0) == derive_seed(42, 50, 0)
    # Same (length, index) shared by both conditions -> single value, by design.
    assert derive_seed(42, 50, 3) != derive_seed(42, 200, 3)
    assert derive_seed(42, 50, 0) != derive_seed(42, 50, 1)
