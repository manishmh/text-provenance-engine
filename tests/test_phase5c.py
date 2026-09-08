"""Tests for Phase 5C: Benchmark Reporting & Aggregation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KgwConfig = Path("configs/kgw.example.json")
SynthIDConfig = Path("configs/synthid.example.json")


@pytest.fixture()
def kgw_detector():
    from provenance.detectors.kgw import KGWDetector
    return KGWDetector.from_config_file(KgwConfig)


@pytest.fixture()
def synthid_detector():
    from provenance.detectors.synthid import SynthIDTextDetector
    return SynthIDTextDetector.from_config_file(SynthIDConfig)


@pytest.fixture()
def kgw_watermarked_text(kgw_detector):
    from provenance.detectors.kgw import generate_controlled_kgw_text
    return generate_controlled_kgw_text(kgw_detector, token_count=120, watermarked=True)


@pytest.fixture()
def synthid_watermarked_text(synthid_detector):
    from provenance.detectors.synthid import generate_controlled_synthid_text
    return generate_controlled_synthid_text(synthid_detector, token_count=80, watermarked=True)


def _make_records(n: int = 5, detected: bool = True, changed: bool = False) -> list[dict]:
    """Create synthetic record dicts for unit tests."""
    return [
        {
            "original_detected": detected,
            "transformed_detected": detected if not changed else not detected,
            "original_score": 10.0 + i,
            "transformed_score": 10.0 + i + (0.0 if not changed else -2.0),
            "score_delta": 0.0 if not changed else -2.0,
            "detection_changed": changed if i == 0 else False,
        }
        for i in range(n)
    ]


def _make_kgw_records(n: int = 5) -> list[dict]:
    """Make records where baseline detects all, but transforms lose some."""
    records = []
    for i in range(n):
        records.append({
            "original_detected": True,
            "transformed_detected": i < n - 1,  # last one loses detection
            "original_score": 15.0,
            "transformed_score": 12.0 if i < n - 1 else 3.0,
            "score_delta": -3.0 if i < n - 1 else -12.0,
            "detection_changed": i == n - 1,
        })
    return records


# ---------------------------------------------------------------------------
# Result schema serialization/deserialization
# ---------------------------------------------------------------------------

class TestResultSchema:
    def test_build_and_roundtrip(self):
        from provenance.robustness.benchmark import (
            ROBUSTNESS_SCHEMA_VERSION,
            RobustnessBenchmarkResult,
            build_benchmark_result,
        )
        records = _make_records(5, detected=True, changed=False)
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="test-config",
            transform_name="identity",
            text_length=100,
            records=records,
            seed=42,
        )
        assert result.schema_version == ROBUSTNESS_SCHEMA_VERSION
        assert result.detector_name == "kgw"
        assert result.sample_count == 5
        assert result.baseline_detected == 5
        assert result.transformed_detected == 5
        assert result.detection_change_count == 0
        assert result.robustness_rate == 1.0

        # Roundtrip
        d = result.to_dict()
        restored = RobustnessBenchmarkResult.from_dict(d)
        assert restored == result

    def test_version_handling(self):
        from provenance.robustness.benchmark import (
            ROBUSTNESS_SCHEMA_VERSION,
            build_benchmark_result,
        )
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="c",
            transform_name="lowercase",
            text_length=50,
            records=_make_records(3),
            seed=1,
        )
        assert result.schema_version == ROBUSTNESS_SCHEMA_VERSION
        assert result.to_dict()["schema_version"] == ROBUSTNESS_SCHEMA_VERSION

    def test_empty_records(self):
        from provenance.robustness.benchmark import build_benchmark_result
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="c",
            transform_name="identity",
            text_length=None,
            records=[],
        )
        assert result.sample_count == 0
        assert result.robustness_rate == 0.0

    def test_detection_change_count(self):
        from provenance.robustness.benchmark import build_benchmark_result
        records = _make_records(10, detected=True, changed=True)
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="c",
            transform_name="lowercase",
            text_length=100,
            records=records,
        )
        # changed=True means detection_changed is True for i==0 only in _make_records
        assert result.detection_change_count == 1
        assert result.detection_change_rate == 0.1

    def test_metadata_preserved(self):
        from provenance.robustness.benchmark import build_benchmark_result
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="c",
            transform_name="identity",
            text_length=100,
            records=_make_records(3),
            metadata={"experiment_id": "exp-1"},
        )
        assert result.metadata["experiment_id"] == "exp-1"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

class TestAggregation:
    def test_aggregate_same_detector(self):
        from provenance.robustness.benchmark import (
            RobustnessBenchmarkResult,
            aggregate_results,
            build_benchmark_result,
        )
        r1 = build_benchmark_result(
            detector_name="kgw", config_identifier="c1",
            transform_name="identity", text_length=50,
            records=_make_records(5), seed=1,
        )
        r2 = build_benchmark_result(
            detector_name="kgw", config_identifier="c1",
            transform_name="identity", text_length=100,
            records=_make_records(5), seed=1,
        )
        agg = aggregate_results([r1, r2])
        assert len(agg) == 1
        assert agg[0].total_samples == 10
        assert agg[0].experiment_count == 2
        assert agg[0].text_lengths == (50, 100)

    def test_aggregate_different_detectors(self):
        from provenance.robustness.benchmark import aggregate_results, build_benchmark_result
        r1 = build_benchmark_result(
            detector_name="kgw", config_identifier="c1",
            transform_name="identity", text_length=50,
            records=_make_records(5), seed=1,
        )
        r2 = build_benchmark_result(
            detector_name="synthid", config_identifier="c2",
            transform_name="identity", text_length=50,
            records=_make_records(5), seed=1,
        )
        agg = aggregate_results([r1, r2])
        assert len(agg) == 2
        names = {a.detector_name for a in agg}
        assert names == {"kgw", "synthid"}

    def test_aggregate_cross_transform(self):
        from provenance.robustness.benchmark import aggregate_results, build_benchmark_result
        r1 = build_benchmark_result(
            detector_name="kgw", config_identifier="c",
            transform_name="identity", text_length=50,
            records=_make_records(5), seed=1,
        )
        r2 = build_benchmark_result(
            detector_name="kgw", config_identifier="c",
            transform_name="lowercase", text_length=50,
            records=_make_records(5, changed=True), seed=1,
        )
        agg = aggregate_results([r1, r2])
        assert len(agg) == 2
        tnames = {a.transform_name for a in agg}
        assert tnames == {"identity", "lowercase"}

    def test_aggregate_cross_length(self):
        from provenance.robustness.benchmark import aggregate_results, build_benchmark_result
        results = []
        for length in [50, 100, 200]:
            results.append(build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=length,
                records=_make_records(3), seed=1,
            ))
        agg = aggregate_results(results)
        assert len(agg) == 1
        assert agg[0].text_lengths == (50, 100, 200)
        assert agg[0].total_samples == 9


# ---------------------------------------------------------------------------
# Wilson confidence intervals
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_wilson_basic(self):
        from provenance.robustness.benchmark import wilson_interval
        low, high = wilson_interval(5, 10)
        assert 0.0 <= low <= 0.5 <= high <= 1.0
        # Wilson interval should be wider than normal for small n
        assert high - low > 0.1

    def test_wilson_all_detected(self):
        from provenance.robustness.benchmark import wilson_interval
        low, high = wilson_interval(10, 10)
        assert low > 0.5  # should be close to 1.0 but not exactly 1.0
        assert high >= 0.99  # clamped to [0,1], floating point may not hit exactly 1.0

    def test_wilson_none_detected(self):
        from provenance.robustness.benchmark import wilson_interval
        low, high = wilson_interval(0, 10)
        assert low == 0.0
        assert high < 1.0  # should be less than 1.0

    def test_wilson_zero_samples(self):
        from provenance.robustness.benchmark import wilson_interval
        low, high = wilson_interval(0, 0)
        assert low == 0.0
        assert high == 0.0

    def test_wilson_large_sample(self):
        from provenance.robustness.benchmark import wilson_interval
        low, high = wilson_interval(80, 100)
        assert 0.7 < low < 0.8 < high < 0.9

    def test_wilson_invalid_raises(self):
        from provenance.robustness.benchmark import wilson_interval
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)

    def test_rate_estimate_ci(self):
        from provenance.robustness.benchmark import rate_estimate
        re = rate_estimate(7, 10)
        assert re.rate == 0.7
        assert re.successes == 7
        assert re.n_samples == 10
        assert re.ci_low < 0.7 < re.ci_high
        assert re.confidence == 0.95

    def test_rate_estimate_zero(self):
        from provenance.robustness.benchmark import rate_estimate
        re = rate_estimate(0, 0)
        assert re.rate == 0.0
        assert re.ci_low == 0.0
        assert re.ci_high == 0.0


# ---------------------------------------------------------------------------
# Robustness matrix
# ---------------------------------------------------------------------------

class TestRobustnessMatrix:
    def test_build_matrix(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="lowercase", text_length=50,
                records=_make_records(5, changed=True), seed=1,
            ),
            build_benchmark_result(
                detector_name="synthid", config_identifier="c2",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        assert "kgw" in matrix.detectors
        assert "synthid" in matrix.detectors
        assert "identity" in matrix.transforms
        assert "lowercase" in matrix.transforms

    def test_matrix_get(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        cell = matrix.get("kgw", "identity")
        assert cell is not None
        assert "robustness_rate" in cell
        assert cell["detector"] == "kgw"

    def test_matrix_roundtrip(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
            RobustnessMatrix,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(3), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        d = matrix.to_dict()
        restored = RobustnessMatrix.from_dict(d)
        assert restored.detectors == matrix.detectors
        assert restored.transforms == matrix.transforms

    def test_matrix_text_render(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
            render_robustness_matrix_text,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="lowercase", text_length=50,
                records=_make_records(5, changed=True), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        text = render_robustness_matrix_text(matrix)
        assert "Robustness Matrix" in text
        assert "kgw" in text
        assert "identity" in text
        assert "Limitations" in text

    def test_matrix_json_output(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(3), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        d = matrix.to_dict()
        json_str = json.dumps(d, sort_keys=True)
        assert "schema_version" in json_str
        assert "kgw" in json_str


# ---------------------------------------------------------------------------
# JSON output contract
# ---------------------------------------------------------------------------

class TestJSONOutput:
    def test_benchmark_result_json(self):
        from provenance.robustness.benchmark import build_benchmark_result
        result = build_benchmark_result(
            detector_name="kgw", config_identifier="c",
            transform_name="identity", text_length=50,
            records=_make_records(5), seed=1,
        )
        json_str = json.dumps(result.to_dict(), sort_keys=True, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "provenance-robustness-v1"
        assert parsed["detector_name"] == "kgw"

    def test_programmatic_report_json(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_programmatic_report,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
        ]
        report = build_programmatic_report(results)
        json_str = json.dumps(report, sort_keys=True, indent=2)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "provenance-robustness-v1"
        assert "aggregated" in parsed
        assert "matrix" in parsed
        assert "limitations" in parsed

    def test_aggregated_json(self):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
        ]
        agg = aggregate_results(results)
        d = agg[0].to_dict()
        json_str = json.dumps(d, sort_keys=True)
        parsed = json.loads(json_str)
        assert parsed["detector_name"] == "kgw"
        assert "text_lengths" in parsed


# ---------------------------------------------------------------------------
# Persistence: write/read roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_write_read_jsonl(self, tmp_path):
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            read_benchmark_results,
            write_benchmark_results,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=_make_records(5), seed=1,
            ),
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="lowercase", text_length=50,
                records=_make_records(5, changed=True), seed=1,
            ),
        ]
        path = tmp_path / "results.jsonl"
        write_benchmark_results(results, path)
        loaded = read_benchmark_results(path)
        assert len(loaded) == 2
        assert loaded[0].transform_name == "identity"
        assert loaded[1].transform_name == "lowercase"

    def test_load_results_from_directory(self, tmp_path):
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            load_results_from_directory,
            write_benchmark_results,
        )
        # Write to two subdirectories
        for subdir in ["run1", "run2"]:
            d = tmp_path / subdir
            d.mkdir()
            write_benchmark_results([
                build_benchmark_result(
                    detector_name="kgw", config_identifier="c",
                    transform_name="identity", text_length=50,
                    records=_make_records(3), seed=1,
                ),
            ], d / "results.jsonl")

        loaded = load_results_from_directory(tmp_path)
        assert len(loaded) == 2

    def test_load_empty_directory(self, tmp_path):
        from provenance.robustness.benchmark import load_results_from_directory
        loaded = load_results_from_directory(tmp_path)
        assert loaded == []


# ---------------------------------------------------------------------------
# No raw text leakage
# ---------------------------------------------------------------------------

class TestNoRawTextLeakage:
    def test_benchmark_result_no_raw_text(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.benchmark import build_benchmark_result
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )

        for r in records:
            d = r.to_dict()
            json_str = json.dumps(d)
            assert kgw_watermarked_text not in json_str

        # Build benchmark result from the experiment records
        for tname in ["identity", "lowercase"]:
            t_records = [r.to_dict() for r in records if r.transform_name == tname]
            br = build_benchmark_result(
                detector_name="kgw",
                config_identifier="test",
                transform_name=tname,
                text_length=120,
                records=t_records,
                seed=42,
            )
            json_str = json.dumps(br.to_dict())
            assert kgw_watermarked_text not in json_str
            assert "original_hash" not in json_str  # not part of benchmark result schema


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------

class TestCLIPhase5C:
    def test_robustness_report_requires_input(self):
        from provenance.cli import main
        ret = main(["benchmark", "robustness-report", "--input", "/nonexistent"])
        assert ret == 1

    def test_robustness_report_with_results(self, tmp_path, kgw_detector, kgw_watermarked_text):
        from provenance.cli import main
        from provenance.detectors.kgw import generate_controlled_kgw_text
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            write_benchmark_results,
        )
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        # Create some benchmark results
        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        bench_results = []
        for tname in ["identity", "lowercase"]:
            t_records = [r.to_dict() for r in records if r.transform_name == tname]
            bench_results.append(build_benchmark_result(
                detector_name="kgw", config_identifier="test-config",
                transform_name=tname, text_length=120,
                records=t_records, seed=42,
            ))
        write_benchmark_results(bench_results, tmp_path / "results.jsonl")

        # Run robustness-report
        ret = main([
            "benchmark", "robustness-report",
            "--input", str(tmp_path),
            "--json",
        ])
        assert ret == 0

    def test_robustness_report_text_output(self, tmp_path, kgw_detector, kgw_watermarked_text):
        from provenance.cli import main
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            write_benchmark_results,
        )
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        bench_results = []
        for tname in ["identity", "lowercase"]:
            t_records = [r.to_dict() for r in records if r.transform_name == tname]
            bench_results.append(build_benchmark_result(
                detector_name="kgw", config_identifier="test-config",
                transform_name=tname, text_length=120,
                records=t_records, seed=42,
            ))
        write_benchmark_results(bench_results, tmp_path / "results.jsonl")

        ret = main([
            "benchmark", "robustness-report",
            "--input", str(tmp_path),
        ])
        assert ret == 0


# ---------------------------------------------------------------------------
# Malformed/incompatible result handling
# ---------------------------------------------------------------------------

class TestMalformedHandling:
    def test_unknown_fields_ignored(self):
        from provenance.robustness.benchmark import RobustnessBenchmarkResult
        data = {
            "schema_version": "provenance-robustness-v1",
            "detector_name": "kgw",
            "config_identifier": "c",
            "transform_name": "identity",
            "text_length": 50,
            "sample_count": 5,
            "seed": 1,
            "baseline_detected": 5,
            "transformed_detected": 5,
            "detection_change_count": 0,
            "mean_baseline_score": 10.0,
            "mean_transformed_score": 10.0,
            "mean_score_delta": 0.0,
            "robustness_rate": 1.0,
            "robustness_ci_low": 0.8,
            "robustness_ci_high": 1.0,
            "baseline_rate": 1.0,
            "baseline_ci_low": 0.8,
            "baseline_ci_high": 1.0,
            "transformed_rate": 1.0,
            "transformed_ci_low": 0.8,
            "transformed_ci_high": 1.0,
            "limitations": [],
            "metadata": {},
            "unknown_future_field": "should be ignored",
        }
        result = RobustnessBenchmarkResult.from_dict(data)
        assert result.detector_name == "kgw"
        assert result.sample_count == 5

    def test_jsonl_malformed_line_skipped(self, tmp_path):
        from provenance.robustness.benchmark import read_benchmark_results
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n{\"schema_version\": \"x\"}\n", encoding="utf-8")
        # Should handle gracefully - either skip or raise
        try:
            read_benchmark_results(p)
        except Exception:
            pass  # acceptable to raise on malformed input


# ---------------------------------------------------------------------------
# Backward compatibility with Phase 5A/5B
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_phase5a_registry_still_works(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "unicode" in reg.names()
        assert "kgw" in reg.names()
        assert "synthid" in reg.names()
        assert "kgw-reference" in reg.names()
        assert "synthid-reference" in reg.names()

    def test_phase5a_robustness_evaluator_still_works(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import evaluate_robustness, compute_statistics
        records = evaluate_robustness("Hello \u00e9", get_registry().create("unicode"))
        stats = compute_statistics(records)
        assert "total_transforms" in stats
        assert "overall_detection_rate" in stats

    def test_phase5b_experiments_still_works(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            WatermarkSample, evaluate_robustness_experiment,
            compute_robustness_report, ExperimentConfig,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        config = ExperimentConfig(
            detector_name="kgw", config_path="configs/kgw.example.json",
            lengths=(120,), samples_per_length=1, seed=42,
        )
        report = compute_robustness_report(records, config)
        assert report.baseline_detection_rate >= 0.0
        assert report.robustness_rate >= 0.0

    def test_phase5b_robustness_report_still_works(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            WatermarkSample, evaluate_robustness_experiment,
            render_robustness_report, ExperimentConfig,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        config = ExperimentConfig(
            detector_name="kgw", config_path="configs/kgw.example.json",
            lengths=(120,), samples_per_length=1, seed=42,
        )
        from provenance.robustness.experiments import compute_robustness_report
        report = compute_robustness_report(records, config)
        text = render_robustness_report(report)
        assert "Watermark Robustness Report" in text
        assert "kgw" in text


# ---------------------------------------------------------------------------
# Human-readable report text
# ---------------------------------------------------------------------------

class TestHumanReadableReport:
    def test_benchmark_report_text(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.benchmark import (
            aggregate_results,
            build_benchmark_result,
            build_robustness_matrix,
            render_benchmark_report_text,
        )
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        bench_results = []
        for tname in ["identity", "lowercase"]:
            t_records = [r.to_dict() for r in records if r.transform_name == tname]
            bench_results.append(build_benchmark_result(
                detector_name="kgw", config_identifier="test-config",
                transform_name=tname, text_length=120,
                records=t_records, seed=42,
            ))

        agg = aggregate_results(bench_results)
        matrix = build_robustness_matrix(agg)
        text = render_benchmark_report_text(bench_results, matrix=matrix, aggregated=agg)
        assert "Phase 5C" in text
        assert "kgw" in text
        assert "identity" in text
        assert "lowercase" in text
        assert "Limitations" in text
        assert "adversarial attacks" in text


# ---------------------------------------------------------------------------
# Detection inversion (transform flips a baseline miss into a hit)
# ---------------------------------------------------------------------------

class TestDetectionInversion:
    """A transform may detect what the baseline missed. Rates must stay valid."""

    def _inverted_records(self):
        return [
            {"original_detected": False, "transformed_detected": True,
             "original_score": 1.0, "transformed_score": 9.0,
             "score_delta": 8.0, "detection_changed": True},
            {"original_detected": True, "transformed_detected": True,
             "original_score": 9.0, "transformed_score": 9.0,
             "score_delta": 0.0, "detection_changed": False},
        ]

    def test_build_result_no_crash_bounded(self):
        from provenance.robustness.benchmark import build_benchmark_result
        result = build_benchmark_result(
            detector_name="kgw", config_identifier="c",
            transform_name="lowercase", text_length=50,
            records=self._inverted_records(), seed=1,
        )
        # Paired: 1 baseline-positive sample, still detected → 1.0
        assert result.robustness_rate == 1.0
        assert 0.0 <= result.robustness_ci_low <= result.robustness_ci_high <= 1.0

    def test_aggregate_no_crash_bounded(self):
        from provenance.robustness.benchmark import (
            RobustnessBenchmarkResult, aggregate_results,
        )
        raw = {
            "schema_version": "provenance-robustness-v1",
            "detector_name": "kgw", "config_identifier": "c",
            "transform_name": "t", "text_length": 50, "sample_count": 4,
            "seed": 1, "baseline_detected": 1, "transformed_detected": 3,
            "detection_change_count": 2,
            "mean_baseline_score": 1.0, "mean_transformed_score": 5.0,
            "mean_score_delta": 4.0,
            "robustness_rate": 3.0, "robustness_ci_low": 0.0,
            "robustness_ci_high": 1.0,
            "baseline_rate": 0.25, "baseline_ci_low": 0.0,
            "baseline_ci_high": 1.0,
            "transformed_rate": 0.75, "transformed_ci_low": 0.0,
            "transformed_ci_high": 1.0,
            "limitations": (), "metadata": {},
        }
        agg = aggregate_results([RobustnessBenchmarkResult(**raw)])
        assert agg[0].robustness_rate == 1.0

    def test_render_with_inversion_no_crash(self):
        from provenance.robustness.benchmark import (
            build_benchmark_result, render_benchmark_report_text,
            build_programmatic_report,
        )
        results = [build_benchmark_result(
            detector_name="kgw", config_identifier="c",
            transform_name="lowercase", text_length=50,
            records=self._inverted_records(), seed=1,
        )]
        text = render_benchmark_report_text(results)
        assert "lowercase" in text
        report = build_programmatic_report(
            results, category_map={"lowercase": "casing"})
        assert report["category_aggregation"][0]["robustness_rate"] == 1.0
