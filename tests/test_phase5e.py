"""Tests for Phase 5E: Benchmark Orchestration and Cross-Model Comparison."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spec(
    detector: str = "unicode",
    config: str = "test.json",
    lengths: tuple[int, ...] = (50, 100),
    samples: int = 5,
    seed: int = 42,
    profile: str | None = None,
) -> dict:
    return {
        "detector": detector,
        "config": config,
        "lengths": list(lengths),
        "samples": samples,
        "seed": seed,
        "profile": profile,
    }


def _make_plan_dict(
    name: str = "test-plan",
    specs: list[dict] | None = None,
) -> dict:
    if specs is None:
        specs = [_make_spec()]
    return {
        "schema_version": "provenance-benchmark-plan-v1",
        "name": name,
        "specs": specs,
        "output_dir": "data/benchmarks/test",
    }


def _make_result(
    detector: str = "kgw",
    config: str = "test.json",
    transform: str = "identity",
    length: int | None = 50,
    n: int = 10,
    bl_detected: int = 10,
    tr_detected: int = 10,
    delta: float = 0.0,
) -> dict:
    return {
        "detector_name": detector,
        "config_identifier": config,
        "transform_name": transform,
        "text_length": length,
        "sample_count": n,
        "seed": 42,
        "baseline_detected": bl_detected,
        "transformed_detected": tr_detected,
        "detection_change_count": 0,
        "mean_baseline_score": 10.0,
        "mean_transformed_score": 10.0 + delta,
        "mean_score_delta": delta,
        "robustness_rate": tr_detected / bl_detected if bl_detected > 0 else 0.0,
        "robustness_ci_low": 0.5,
        "robustness_ci_high": 1.0,
        "baseline_rate": bl_detected / n if n > 0 else 0.0,
        "baseline_ci_low": 0.5,
        "baseline_ci_high": 1.0,
        "transformed_rate": tr_detected / n if n > 0 else 0.0,
        "transformed_ci_low": 0.5,
        "transformed_ci_high": 1.0,
        "limitations": (),
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# BenchmarkSpec serialization
# ---------------------------------------------------------------------------

class TestBenchmarkSpec:
    def test_spec_roundtrip(self):
        from provenance.robustness.orchestration import BenchmarkSpec
        spec = BenchmarkSpec(
            detector="kgw",
            config="configs/kgw.model_a.json",
            lengths=(50, 100),
            samples=10,
            seed=42,
            profile="all_safe",
        )
        d = spec.to_dict()
        restored = BenchmarkSpec.from_dict(d)
        assert restored.detector == "kgw"
        assert restored.config == "configs/kgw.model_a.json"
        assert restored.lengths == (50, 100)
        assert restored.samples == 10
        assert restored.seed == 42
        assert restored.profile == "all_safe"

    def test_spec_experiment_id_deterministic(self):
        from provenance.robustness.orchestration import BenchmarkSpec
        s1 = BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)
        s2 = BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)
        assert s1.experiment_id == s2.experiment_id

    def test_spec_different_params_different_id(self):
        from provenance.robustness.orchestration import BenchmarkSpec
        s1 = BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)
        s2 = BenchmarkSpec(detector="kgw", config="b.json", lengths=(50,), samples=5, seed=42)
        assert s1.experiment_id != s2.experiment_id

    def test_spec_from_dict_ignores_experiment_id(self):
        from provenance.robustness.orchestration import BenchmarkSpec
        data = _make_spec()
        data["experiment_id"] = "should-be-ignored"
        spec = BenchmarkSpec.from_dict(data)
        # experiment_id is computed, not stored
        assert spec.to_dict()["experiment_id"] != "should-be-ignored"

    def test_spec_json_serializable(self):
        from provenance.robustness.orchestration import BenchmarkSpec
        spec = BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)
        json_str = json.dumps(spec.to_dict(), sort_keys=True)
        parsed = json.loads(json_str)
        assert parsed["detector"] == "kgw"
        assert "experiment_id" in parsed


# ---------------------------------------------------------------------------
# Plan loading and validation
# ---------------------------------------------------------------------------

class TestBenchmarkPlan:
    def test_plan_roundtrip(self):
        from provenance.robustness.orchestration import BenchmarkPlan
        plan = BenchmarkPlan(
            schema_version="provenance-benchmark-plan-v1",
            name="test",
            specs=(),
        )
        d = plan.to_dict()
        restored = BenchmarkPlan.from_dict(d)
        assert restored.name == "test"
        assert len(restored.specs) == 0

    def test_plan_with_specs(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec
        specs = [
            BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42),
            BenchmarkSpec(detector="synthid", config="b.json", lengths=(100,), samples=3, seed=43),
        ]
        plan = BenchmarkPlan(
            schema_version="provenance-benchmark-plan-v1",
            name="multi-model",
            specs=tuple(specs),
        )
        assert len(plan.specs) == 2
        assert len(plan.experiment_ids) == 2

    def test_plan_id_deterministic(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec
        specs = [BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)]
        p1 = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        p2 = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        assert p1.plan_id == p2.plan_id

    def test_plan_save_load(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec
        specs = [BenchmarkSpec(detector="kgw", config="a.json", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        path = tmp_path / "plan.json"
        plan.save(path)
        loaded = BenchmarkPlan.from_file(path)
        assert loaded.name == "test"
        assert len(loaded.specs) == 1

    def test_plan_validation_empty(self):
        from provenance.robustness.orchestration import BenchmarkPlan, validate_plan
        plan = BenchmarkPlan(schema_version="v1", name="", specs=())
        errors = validate_plan(plan)
        assert any("name" in e.field for e in errors)

    def test_plan_validation_no_specs(self):
        from provenance.robustness.orchestration import BenchmarkPlan, validate_plan
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=())
        errors = validate_plan(plan)
        assert any("specs" in e.field for e in errors)

    def test_plan_validation_unknown_detector(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [BenchmarkSpec(detector="nonexistent", config="a.json", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert any("detector" in e.field for e in errors)

    def test_plan_validation_negative_length(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [BenchmarkSpec(detector="unicode", config="", lengths=(-1,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert any("lengths" in e.field for e in errors)

    def test_plan_validation_zero_samples(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=0, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert any("samples" in e.field for e in errors)

    def test_plan_validation_unknown_profile(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42, profile="nonexistent")]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert any("profile" in e.field for e in errors)

    def test_plan_validation_duplicate_experiments(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [
            BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42),
            BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42),
        ]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert any("experiment_id" in e.field for e in errors)

    def test_plan_validation_valid(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, validate_plan
        specs = [
            BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42),
            BenchmarkSpec(detector="kgw", config="a.json", lengths=(100,), samples=3, seed=43),
        ]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        errors = validate_plan(plan)
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

class TestExperimentID:
    def test_id_deterministic(self):
        from provenance.robustness.orchestration import compute_experiment_id
        spec = {"detector": "kgw", "config": "a.json", "lengths": [50], "samples": 5, "seed": 42}
        id1 = compute_experiment_id(spec)
        id2 = compute_experiment_id(spec)
        assert id1 == id2

    def test_id_length(self):
        from provenance.robustness.orchestration import compute_experiment_id
        spec = {"detector": "kgw", "config": "a.json", "lengths": [50], "samples": 5, "seed": 42}
        eid = compute_experiment_id(spec)
        assert len(eid) == 16  # truncated SHA-256

    def test_id_no_timestamp(self):
        from provenance.robustness.orchestration import compute_experiment_id
        # Same spec should produce same ID regardless of when called
        spec = {"detector": "kgw", "config": "a.json"}
        id1 = compute_experiment_id(spec)
        id2 = compute_experiment_id(spec)
        assert id1 == id2

    def test_id_differs_for_different_specs(self):
        from provenance.robustness.orchestration import compute_experiment_id
        id1 = compute_experiment_id({"detector": "kgw", "config": "a.json"})
        id2 = compute_experiment_id({"detector": "kgw", "config": "b.json"})
        assert id1 != id2


# ---------------------------------------------------------------------------
# RunManifest
# ---------------------------------------------------------------------------

class TestRunManifest:
    def test_create_manifest(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, create_run_manifest
        specs = [BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        manifest = create_run_manifest(plan)
        assert manifest.plan_name == "test"
        assert len(manifest.experiments) == 1
        assert manifest.experiments[0].status == "pending"

    def test_manifest_roundtrip(self):
        from provenance.robustness.orchestration import RunManifest, ExperimentManifest, ExperimentStatus
        manifest = RunManifest(
            run_id="run-test",
            benchmark_version="v1",
            plan_name="test",
            started_at="2024-01-01T00:00:00",
            experiments=[
                ExperimentManifest(
                    experiment_id="exp-1",
                    spec={"detector": "unicode"},
                    status=ExperimentStatus.COMPLETED,
                ),
            ],
        )
        d = manifest.to_dict()
        restored = RunManifest.from_dict(d)
        assert restored.run_id == "run-test"
        assert len(restored.experiments) == 1

    def test_manifest_save_load(self, tmp_path):
        from provenance.robustness.orchestration import (
            BenchmarkPlan, BenchmarkSpec, RunManifest, create_run_manifest,
        )
        specs = [BenchmarkSpec(detector="unicode", config="", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        manifest = create_run_manifest(plan)
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = RunManifest.from_file(path)
        assert loaded.plan_name == "test"

    def test_manifest_completed_count(self):
        from provenance.robustness.orchestration import ExperimentManifest, ExperimentStatus, RunManifest
        manifest = RunManifest(
            run_id="r", benchmark_version="v1", plan_name="p", started_at="t",
            experiments=[
                ExperimentManifest(experiment_id="e1", spec={}, status=ExperimentStatus.COMPLETED),
                ExperimentManifest(experiment_id="e2", spec={}, status=ExperimentStatus.FAILED),
                ExperimentManifest(experiment_id="e3", spec={}, status=ExperimentStatus.PENDING),
            ],
        )
        assert len(manifest.completed_experiments) == 1
        assert len(manifest.failed_experiments) == 1
        assert len(manifest.pending_experiments) == 1
        assert not manifest.is_complete

    def test_manifest_is_complete(self):
        from provenance.robustness.orchestration import ExperimentManifest, ExperimentStatus, RunManifest
        manifest = RunManifest(
            run_id="r", benchmark_version="v1", plan_name="p", started_at="t",
            experiments=[
                ExperimentManifest(experiment_id="e1", spec={}, status=ExperimentStatus.COMPLETED),
                ExperimentManifest(experiment_id="e2", spec={}, status=ExperimentStatus.SKIPPED),
            ],
        )
        assert manifest.is_complete

    def test_manifest_get_experiment(self):
        from provenance.robustness.orchestration import ExperimentManifest, ExperimentStatus, RunManifest
        manifest = RunManifest(
            run_id="r", benchmark_version="v1", plan_name="p", started_at="t",
            experiments=[
                ExperimentManifest(experiment_id="e1", spec={}, status=ExperimentStatus.COMPLETED),
            ],
        )
        assert manifest.get_experiment("e1") is not None
        assert manifest.get_experiment("e2") is None


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    def test_runner_creates_manifest(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkRunner
        plan = BenchmarkPlan.from_dict(_make_plan_dict("runner-test"))
        runner = BenchmarkRunner(plan=plan)
        manifest = runner.run(tmp_path)
        assert manifest.plan_name == "runner-test"
        assert manifest.is_complete

    def test_runner_resume(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkRunner
        # Create a plan and run it
        plan = BenchmarkPlan.from_dict(_make_plan_dict("resume-test"))
        runner = BenchmarkRunner(plan=plan)
        manifest = runner.run(tmp_path)
        assert manifest.is_complete

        # Resume — should detect completed experiments
        runner2 = BenchmarkRunner(plan=plan, resume=True)
        manifest2 = runner2.run(tmp_path)
        # All experiments should still be completed (not re-run)
        assert manifest2.is_complete

    def test_runner_force_rerun(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkRunner
        plan = BenchmarkPlan.from_dict(_make_plan_dict("force-test"))
        runner = BenchmarkRunner(plan=plan)
        manifest = runner.run(tmp_path)
        assert manifest.is_complete

        # Force — should re-run
        runner2 = BenchmarkRunner(plan=plan, resume=True, force=True)
        manifest2 = runner2.run(tmp_path)
        assert manifest2.is_complete

    def test_runner_failure_isolation(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, BenchmarkRunner

        def _failing_fn(spec, out_dir):
            if spec.config == "fail.json":
                raise ValueError("Intentional failure")
            return []

        specs = [
            BenchmarkSpec(detector="unicode", config="ok.json", lengths=(50,), samples=1, seed=42),
            BenchmarkSpec(detector="unicode", config="fail.json", lengths=(50,), samples=1, seed=43),
            BenchmarkSpec(detector="unicode", config="ok2.json", lengths=(50,), samples=1, seed=44),
        ]
        plan = BenchmarkPlan(schema_version="v1", name="failure-test", specs=tuple(specs))
        runner = BenchmarkRunner(plan=plan, experiment_fn=_failing_fn)
        manifest = runner.run(tmp_path)
        completed = len(manifest.completed_experiments)
        failed = len(manifest.failed_experiments)
        assert completed == 2  # ok and ok2
        assert failed == 1    # fail


# ---------------------------------------------------------------------------
# Comparison Report
# ---------------------------------------------------------------------------

class TestComparisonReport:
    def test_build_comparison(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="model-a", transform="identity", n=10, bl_detected=10, tr_detected=10),
            _make_result(detector="kgw", config="model-a", transform="lowercase", n=10, bl_detected=10, tr_detected=8, delta=-2.0),
            _make_result(detector="synthid", config="model-b", transform="identity", n=10, bl_detected=10, tr_detected=10),
        ]
        report = build_comparison_report(results, run_id="run-1", benchmark_name="test")
        assert report.schema_version == "provenance-benchmark-report-v1"
        assert len(report.rows) == 3
        assert len(report.by_model) == 2

    def test_comparison_by_transform(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="a", transform="identity", n=10, bl_detected=10, tr_detected=10),
            _make_result(detector="kgw", config="a", transform="lowercase", n=10, bl_detected=10, tr_detected=8),
            _make_result(detector="kgw", config="b", transform="identity", n=10, bl_detected=10, tr_detected=10),
            _make_result(detector="kgw", config="b", transform="lowercase", n=10, bl_detected=10, tr_detected=7),
        ]
        report = build_comparison_report(results)
        by_t = {r["transform"]: r for r in report.by_transform}
        assert "identity" in by_t
        assert "lowercase" in by_t

    def test_comparison_by_length(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="a", transform="identity", length=50, n=10, bl_detected=10, tr_detected=10),
            _make_result(detector="kgw", config="a", transform="identity", length=100, n=10, bl_detected=10, tr_detected=10),
        ]
        report = build_comparison_report(results)
        assert len(report.by_length) == 2

    def test_comparison_by_category(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="a", transform="lowercase", n=10, bl_detected=10, tr_detected=8),
            _make_result(detector="kgw", config="a", transform="unicode_nfc", n=10, bl_detected=10, tr_detected=10),
        ]
        category_map = {"lowercase": "casing", "unicode_nfc": "unicode"}
        report = build_comparison_report(results, category_map=category_map)
        assert len(report.by_category) == 2
        cats = {r["category"] for r in report.by_category}
        assert "casing" in cats
        assert "unicode" in cats

    def test_comparison_json_serializable(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [_make_result()]
        report = build_comparison_report(results, run_id="r1", benchmark_name="test")
        json_str = json.dumps(report.to_dict(), sort_keys=True)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == "provenance-benchmark-report-v1"
        assert parsed["run_id"] == "r1"

    def test_comparison_text_render(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="model-a", transform="identity"),
            _make_result(detector="kgw", config="model-a", transform="lowercase", bl_detected=10, tr_detected=8),
        ]
        report = build_comparison_report(results, run_id="r1", benchmark_name="test")
        text = report.render_text()
        assert "Cross-Model Benchmark Comparison Report" in text
        assert "kgw" in text
        assert "Limitations" in text
        assert "By Model" in text
        assert "By Transformation" in text


# ---------------------------------------------------------------------------
# Secret/text leakage prevention
# ---------------------------------------------------------------------------

class TestSecretLeakage:
    def test_no_raw_text_in_plan(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec
        specs = [BenchmarkSpec(detector="unicode", config="a.json", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        json_str = json.dumps(plan.to_dict())
        assert "secret" not in json_str.lower()
        assert "api_key" not in json_str.lower()
        assert "password" not in json_str.lower()

    def test_no_raw_text_in_manifest(self):
        from provenance.robustness.orchestration import BenchmarkPlan, BenchmarkSpec, create_run_manifest
        specs = [BenchmarkSpec(detector="unicode", config="a.json", lengths=(50,), samples=5, seed=42)]
        plan = BenchmarkPlan(schema_version="v1", name="test", specs=tuple(specs))
        manifest = create_run_manifest(plan)
        json_str = json.dumps(manifest.to_dict())
        assert "raw_text" not in json_str.lower()
        assert "watermark_key" not in json_str.lower()

    def test_no_raw_text_in_comparison(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [_make_result()]
        report = build_comparison_report(results)
        json_str = json.dumps(report.to_dict())
        assert len(json_str) < 5000  # report should be compact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_plan_argument_parsed(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "plan",
            "--plan", "benchmark_plan.json",
        ])
        assert args.plan == Path("benchmark_plan.json")
        assert args.dry_run is False
        assert args.resume is False
        assert args.force is False

    def test_dry_run_flag(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "plan",
            "--plan", "benchmark_plan.json",
            "--dry-run",
        ])
        assert args.dry_run is True

    def test_resume_flag(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "plan",
            "--plan", "benchmark_plan.json",
            "--resume",
        ])
        assert args.resume is True

    def test_force_flag(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "plan",
            "--plan", "benchmark_plan.json",
            "--force",
        ])
        assert args.force is True

    def test_dry_run_output(self, tmp_path):
        from provenance.cli import main
        from provenance.robustness.orchestration import BenchmarkPlan
        plan_path = tmp_path / "plan.json"
        plan = BenchmarkPlan.from_dict(_make_plan_dict(
            name="dry-test",
            specs=[
                _make_spec(detector="unicode", config="a.json"),
                _make_spec(detector="kgw", config="b.json"),
            ],
        ))
        plan.save(plan_path)
        ret = main(["benchmark", "plan", "--plan", str(plan_path), "--dry-run"])
        assert ret == 0

    def test_plan_validation_error(self, tmp_path):
        from provenance.cli import main
        from provenance.robustness.orchestration import BenchmarkPlan
        plan_path = tmp_path / "plan.json"
        plan = BenchmarkPlan.from_dict(_make_plan_dict(
            name="",
            specs=[_make_spec(detector="nonexistent")],
        ))
        plan.save(plan_path)
        ret = main(["benchmark", "plan", "--plan", str(plan_path)])
        assert ret == 1

    def test_plan_missing_file(self):
        from provenance.cli import main
        ret = main(["benchmark", "plan", "--plan", "/nonexistent/plan.json"])
        assert ret == 1


# ---------------------------------------------------------------------------
# Backward compatibility with Phase 5A–5D
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_phase5a_transforms_still_work(self):
        from provenance.robustness.transforms import get_all_transforms, get_transform
        transforms = get_all_transforms()
        assert len(transforms) >= 10
        t = get_transform("identity")
        assert t.name == "identity"

    def test_phase5a_evaluator_still_works(self):
        from provenance.detectors.registry import get_registry
        from provenance.robustness.evaluator import evaluate_robustness, compute_statistics
        records = evaluate_robustness("Hello \u00e9", get_registry().create("unicode"))
        stats = compute_statistics(records)
        assert "total_transforms" in stats
        assert "overall_detection_rate" in stats

    def test_phase5b_experiments_still_work(self):
        from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text
        from provenance.robustness.experiments import (
            WatermarkSample, evaluate_robustness_experiment,
            compute_robustness_report, ExperimentConfig,
        )
        from provenance.robustness.transforms import get_transform
        detector = KGWDetector.from_config_file("configs/kgw.example.json")
        text = generate_controlled_kgw_text(detector, token_count=120, watermarked=True)
        sample = WatermarkSample(
            text=text, token_count=120, sample_seed=42,
            prompt_id="p-0", length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        config = ExperimentConfig(
            detector_name="kgw", config_path="configs/kgw.example.json",
            lengths=(120,), samples_per_length=1, seed=42,
        )
        report = compute_robustness_report(records, config)
        assert report.baseline_detection_rate >= 0.0
        assert report.robustness_rate >= 0.0

    def test_phase5c_benchmark_still_works(self):
        from provenance.robustness.benchmark import (
            build_benchmark_result, aggregate_results,
            build_robustness_matrix, build_programmatic_report,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": True,
                     "original_score": 10.0, "transformed_score": 10.0,
                     "score_delta": 0.0, "detection_changed": False}
                    for _ in range(5)
                ], seed=1,
            ),
        ]
        agg = aggregate_results(results)
        matrix = build_robustness_matrix(agg)
        report = build_programmatic_report(results, aggregated=agg, matrix=matrix)
        assert report["schema_version"] == "provenance-robustness-v1"

    def test_phase5c_wilson_ci_still_works(self):
        from provenance.robustness.benchmark import wilson_interval, rate_estimate
        low, high = wilson_interval(5, 10)
        assert 0.0 <= low <= 0.5 <= high <= 1.0

    def test_phase5c_persistence_still_works(self, tmp_path):
        from provenance.robustness.benchmark import (
            build_benchmark_result, write_benchmark_results, read_benchmark_results,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="identity", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": True,
                     "original_score": 10.0, "transformed_score": 10.0,
                     "score_delta": 0.0, "detection_changed": False}
                    for _ in range(3)
                ], seed=1,
            ),
        ]
        path = tmp_path / "test.jsonl"
        write_benchmark_results(results, path)
        loaded = read_benchmark_results(path)
        assert len(loaded) == 1

    def test_phase5d_profiles_still_work(self):
        from provenance.robustness.profiles import get_profile, list_profile_names
        names = list_profile_names()
        assert "all_safe" in names
        p = get_profile("all_safe")
        assert len(p.transform_names) > 10

    def test_phase5d_advanced_transforms_still_work(self):
        from provenance.robustness.advanced_transforms import (
            get_all_advanced_transforms, get_categories, TransformCategory,
        )
        transforms = get_all_advanced_transforms()
        assert len(transforms) >= 20
        cats = get_categories()
        assert TransformCategory.FORMATTING in cats
        assert TransformCategory.LEXICAL in cats

    def test_registry_still_works(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "unicode" in reg.names()
        assert "kgw" in reg.names()
        assert "synthid" in reg.names()

    def test_cli_main_still_works(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        assert parser is not None


# ---------------------------------------------------------------------------
# Resume semantics (regression: failed experiments must be retried)
# ---------------------------------------------------------------------------

class TestResumeSemantics:
    def _plan(self, name="resume-test", seed=42):
        from provenance.robustness.orchestration import BenchmarkPlan
        return BenchmarkPlan.from_dict(_make_plan_dict(name, [_make_spec(seed=seed)]))

    def test_resume_retries_failed(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkRunner, ExperimentStatus
        calls = {"n": 0}

        def _flaky(spec, out_dir):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient boom")
            return []

        plan = self._plan()
        m1 = BenchmarkRunner(plan=plan, experiment_fn=_flaky).run(tmp_path)
        assert len(m1.failed_experiments) == 1

        m2 = BenchmarkRunner(plan=plan, resume=True, experiment_fn=_flaky).run(tmp_path)
        assert len(m2.completed_experiments) == 1
        assert len(m2.failed_experiments) == 0
        # Stale failure details are cleared on retry
        assert m2.experiments[0].error_type is None
        assert m2.experiments[0].error_message is None
        assert m2.experiments[0].status == ExperimentStatus.COMPLETED

    def test_resume_skips_completed(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkRunner
        calls = {"n": 0}

        def _counting(spec, out_dir):
            calls["n"] += 1
            return []

        plan = self._plan()
        BenchmarkRunner(plan=plan, experiment_fn=_counting).run(tmp_path)
        assert calls["n"] == 1
        BenchmarkRunner(plan=plan, resume=True, experiment_fn=_counting).run(tmp_path)
        assert calls["n"] == 1  # not re-run

    def test_resume_plan_mismatch_raises(self, tmp_path):
        from provenance.robustness.orchestration import BenchmarkRunner
        import pytest
        plan_a = self._plan(name="plan-a", seed=42)
        BenchmarkRunner(plan=plan_a).run(tmp_path)
        plan_b = self._plan(name="plan-b", seed=43)
        with pytest.raises(ValueError, match="different plan"):
            BenchmarkRunner(plan=plan_b, resume=True).run(tmp_path)

    def test_manifest_plan_id_roundtrip(self, tmp_path):
        from provenance.robustness.orchestration import RunManifest, create_run_manifest
        manifest = create_run_manifest(self._plan())
        assert manifest.plan_id is not None
        path = tmp_path / "manifest.json"
        manifest.save(path)
        loaded = RunManifest.from_file(path)
        assert loaded.plan_id == manifest.plan_id

    def test_legacy_manifest_without_plan_id_loads(self):
        from provenance.robustness.orchestration import RunManifest
        manifest = RunManifest(
            run_id="r", benchmark_version="v1", plan_name="p",
            started_at="t", experiments=[],
        )
        assert manifest.plan_id is None
        restored = RunManifest.from_dict(manifest.to_dict())
        assert restored.plan_id is None


# ---------------------------------------------------------------------------
# Comparison clamping (regression: inversion must not exceed 1.0)
# ---------------------------------------------------------------------------

class TestComparisonClamping:
    def test_inversion_clamped(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="a", transform="t",
                         n=10, bl_detected=2, tr_detected=5),
        ]
        report = build_comparison_report(results)
        assert report.rows[0].robustness_rate == 1.0
        assert report.rows[0].baseline_detected == 2
        assert report.rows[0].transformed_detected == 5
        by_model = {m["model_config"]: m for m in report.by_model}
        assert by_model["a"]["robustness_rate"] == 1.0

    def test_raw_counts_preserved(self):
        from provenance.robustness.orchestration import build_comparison_report
        results = [
            _make_result(detector="kgw", config="a", transform="t1",
                         n=10, bl_detected=10, tr_detected=7),
            _make_result(detector="kgw", config="a", transform="t2",
                         n=10, bl_detected=10, tr_detected=9),
        ]
        report = build_comparison_report(results)
        by_model = {m["model_config"]: m for m in report.by_model}
        assert by_model["a"]["robustness_rate"] == 0.8
        assert by_model["a"]["total_samples"] == 20
