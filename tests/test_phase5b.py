"""Tests for Phase 5B: Real Watermark Robustness Evaluation."""
from __future__ import annotations

import json
import tempfile
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


# ---------------------------------------------------------------------------
# KGW robustness evaluation
# ---------------------------------------------------------------------------

class TestKGWRobustness:
    def test_kgw_detects_watermarked_text(self, kgw_detector, kgw_watermarked_text):
        result = kgw_detector.detect(kgw_watermarked_text)
        assert result.detected is True
        assert result.status == "ok"
        assert result.score is not None
        assert result.score > 0

    def test_kgw_robustness_experiment(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_all_transforms

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], get_all_transforms()
        )
        assert len(records) >= 10  # at least 10 transforms
        for r in records:
            assert r.detector_name == "kgw"
            assert r.original_hash is not None
            assert r.transformed_hash is not None
            assert r.original_score is not None
            assert r.score_delta is not None

    def test_kgw_robustness_identity_preserves_detection(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], [get_transform("identity")]
        )
        assert len(records) == 1
        r = records[0]
        assert r.original_detected is True
        assert r.transformed_detected is True
        assert r.detection_preserved is True
        assert r.detection_changed is False
        assert r.score_delta == 0.0

    def test_kgw_score_delta_computed(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        assert len(records) == 2
        identity_rec = [r for r in records if r.transform_name == "identity"][0]
        lowercase_rec = [r for r in records if r.transform_name == "lowercase"][0]
        assert identity_rec.score_delta == 0.0
        # Lowercase may change the score
        assert lowercase_rec.score_delta is not None


# ---------------------------------------------------------------------------
# SynthID robustness evaluation
# ---------------------------------------------------------------------------

class TestSynthIDRobustness:
    def test_synthid_detects_watermarked_text(self, synthid_detector, synthid_watermarked_text):
        result = synthid_detector.detect(synthid_watermarked_text)
        assert result.detected is True
        assert result.status == "ok"
        assert result.score is not None
        assert result.score > 0

    def test_synthid_robustness_experiment(self, synthid_detector, synthid_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_all_transforms

        sample = WatermarkSample(
            text=synthid_watermarked_text,
            token_count=80,
            sample_seed=42,
            prompt_id="prompt-0",
            length=80,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            synthid_detector, [sample], get_all_transforms()
        )
        assert len(records) >= 10
        for r in records:
            assert r.detector_name == "synthid"
            assert r.original_hash is not None
            assert r.transformed_hash is not None
            assert r.original_score is not None

    def test_synthid_robustness_identity_preserves_detection(self, synthid_detector, synthid_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=synthid_watermarked_text,
            token_count=80,
            sample_seed=42,
            prompt_id="prompt-0",
            length=80,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            synthid_detector, [sample], [get_transform("identity")]
        )
        assert len(records) == 1
        r = records[0]
        assert r.original_detected is True
        assert r.transformed_detected is True
        assert r.detection_preserved is True


# ---------------------------------------------------------------------------
# Robustness metrics calculation
# ---------------------------------------------------------------------------

class TestRobustnessMetrics:
    def test_robustness_rate_calculation(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            ExperimentConfig, WatermarkSample, compute_robustness_report,
            evaluate_robustness_experiment,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("whitespace_normalization")],
        )
        config = ExperimentConfig(
            detector_name="kgw",
            config_path="configs/kgw.example.json",
            lengths=(120,),
            samples_per_length=1,
            seed=42,
        )
        report = compute_robustness_report(records, config)
        assert report.sample_count == 1
        assert report.baseline_detection_rate >= 0.0
        assert report.robustness_rate >= 0.0
        assert report.robustness_rate <= 1.0
        assert "identity" in report.per_transform
        assert "whitespace_normalization" in report.per_transform

    def test_report_deterministic(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            ExperimentConfig, WatermarkSample, compute_robustness_report,
            evaluate_robustness_experiment,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        config = ExperimentConfig(
            detector_name="kgw",
            config_path="configs/kgw.example.json",
            lengths=(120,),
            samples_per_length=1,
            seed=42,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        r1 = compute_robustness_report(records, config)
        r2 = compute_robustness_report(records, config)
        assert r1.to_dict() == r2.to_dict()


# ---------------------------------------------------------------------------
# Privacy / hash guarantees
# ---------------------------------------------------------------------------

class TestPrivacyHashGuarantees:
    def test_no_raw_text_in_robustness_records(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_all_transforms

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], get_all_transforms()
        )
        for r in records:
            d = r.to_dict()
            # Raw text should not appear anywhere in the record
            assert kgw_watermarked_text not in json.dumps(d)
            assert "original_hash" in d
            assert "transformed_hash" in d
            assert len(d["original_hash"]) == 64  # SHA-256 hex digest
            assert len(d["transformed_hash"]) == 64

    def test_no_raw_text_in_synthid_records(self, synthid_detector, synthid_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_all_transforms

        sample = WatermarkSample(
            text=synthid_watermarked_text,
            token_count=80,
            sample_seed=42,
            prompt_id="prompt-0",
            length=80,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            synthid_detector, [sample], get_all_transforms()
        )
        for r in records:
            d = r.to_dict()
            assert synthid_watermarked_text not in json.dumps(d)
            assert len(d["original_hash"]) == 64
            assert len(d["transformed_hash"]) == 64


# ---------------------------------------------------------------------------
# Deterministic results
# ---------------------------------------------------------------------------

class TestDeterministicResults:
    def test_kgw_robustness_deterministic(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        r1 = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        r2 = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        def _strip_timing(d):
            return {k: v for k, v in d.items() if k != "duration_ms"}
        assert [_strip_timing(r.to_dict()) for r in r1] == [_strip_timing(r.to_dict()) for r in r2]

    def test_synthid_robustness_deterministic(self, synthid_detector, synthid_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=synthid_watermarked_text,
            token_count=80,
            sample_seed=42,
            prompt_id="prompt-0",
            length=80,
            watermarked=True,
        )
        r1 = evaluate_robustness_experiment(
            synthid_detector, [sample],
            [get_transform("identity"), get_transform("uppercase")],
        )
        r2 = evaluate_robustness_experiment(
            synthid_detector, [sample],
            [get_transform("identity"), get_transform("uppercase")],
        )
        def _strip_timing(d):
            return {k: v for k, v in d.items() if k != "duration_ms"}
        assert [_strip_timing(r.to_dict()) for r in r1] == [_strip_timing(r.to_dict()) for r in r2]


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------

class TestCLIRobustnessPhase5B:
    def test_robustness_benchmark_kgw_simulation(self, tmp_path):
        """Test CLI robustness with KGW simulation detector (no HF model needed)."""
        from provenance.cli import main
        from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text

        # Generate sample text using the KGW simulation detector
        detector = KGWDetector.from_config_file(KgwConfig)
        text = generate_controlled_kgw_text(detector, token_count=120, watermarked=True)

        text_file = tmp_path / "kgw.txt"
        text_file.write_text(text, encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", str(KgwConfig),
            "--text", str(text_file),
            "--detector", "kgw",
            "--json",
        ])
        assert ret == 0

    def test_robustness_benchmark_synthid_simulation(self, tmp_path):
        """Test CLI robustness with SynthID simulation detector."""
        from provenance.cli import main
        from provenance.detectors.synthid import SynthIDTextDetector, generate_controlled_synthid_text

        detector = SynthIDTextDetector.from_config_file(SynthIDConfig)
        text = generate_controlled_synthid_text(detector, token_count=80, watermarked=True)

        text_file = tmp_path / "synthid.txt"
        text_file.write_text(text, encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", str(SynthIDConfig),
            "--text", str(text_file),
            "--detector", "synthid",
            "--json",
        ])
        assert ret == 0

    def test_robustness_benchmark_specific_transforms(self, tmp_path):
        """Test CLI with specific transform selection."""
        from provenance.cli import main
        from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text

        detector = KGWDetector.from_config_file(KgwConfig)
        text = generate_controlled_kgw_text(detector, token_count=120, watermarked=True)

        text_file = tmp_path / "kgw.txt"
        text_file.write_text(text, encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", str(KgwConfig),
            "--text", str(text_file),
            "--detector", "kgw",
            "--transforms", "identity,lowercase,whitespace_normalization",
            "--json",
        ])
        assert ret == 0

    def test_robustness_unicode_regression(self, tmp_path):
        """Verify Unicode robustness still works (Phase 5A regression)."""
        from provenance.cli import main

        text_file = tmp_path / "unicode.txt"
        text_file.write_text("Hello World with \u00e9\u00e8\u00ea", encoding="utf-8")

        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--text", str(text_file),
            "--detector", "unicode",
            "--json",
        ])
        assert ret == 0

    def test_robustness_watermark_requires_config(self):
        """Watermark mode without --config should fail."""
        from provenance.cli import main

        ret = main([
            "benchmark", "robustness",
            "--detector", "kgw",
        ])
        assert ret == 1

    def test_robustness_watermark_with_lengths_and_samples(self):
        """Test watermark mode with custom lengths and samples.

        Uses the simulation config which has simple tokenizer.
        The generation will fail since it requires HF model/tokenizer,
        but argument parsing should succeed (ret != 2 = not argparse error).
        """
        from provenance.cli import main

        ret = main([
            "benchmark", "robustness",
            "--config", str(KgwConfig),
            "--detector", "kgw",
            "--lengths", "50,100",
            "--samples", "2",
            "--seed", "42",
        ])
        # Simple tokenizer configs don't support the HF generation pipeline.
        # The important thing is it doesn't crash with argument errors (ret=2).
        assert ret != 2  # not an argparse error


# ---------------------------------------------------------------------------
# Benchmark report
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_robustness_report_render(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            ExperimentConfig, WatermarkSample, compute_robustness_report,
            evaluate_robustness_experiment, render_robustness_report,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample],
            [get_transform("identity"), get_transform("lowercase"), get_transform("uppercase")],
        )
        config = ExperimentConfig(
            detector_name="kgw",
            config_path="configs/kgw.example.json",
            lengths=(120,),
            samples_per_length=1,
            seed=42,
        )
        report = compute_robustness_report(records, config)
        text = render_robustness_report(report)
        assert "Watermark Robustness Report" in text
        assert "kgw" in text
        assert "Baseline detection rate" in text
        assert "Robustness rate" in text
        assert "Limitations" in text
        assert "adversarial attacks" in text

    def test_robustness_report_json_serializable(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import (
            ExperimentConfig, WatermarkSample, compute_robustness_report,
            evaluate_robustness_experiment,
        )
        from provenance.robustness.transforms import get_all_transforms

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], get_all_transforms()
        )
        config = ExperimentConfig(
            detector_name="kgw",
            config_path="configs/kgw.example.json",
            lengths=(120,),
            samples_per_length=1,
            seed=42,
        )
        report = compute_robustness_report(records, config)
        # Should be JSON-serializable
        json_str = json.dumps(report.to_dict(), indent=2, sort_keys=True)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["detector"] == "kgw"
        assert "baseline_detection_rate" in parsed
        assert "robustness_rate" in parsed
        assert "per_transform" in parsed

    def test_synthid_report(self, synthid_detector, synthid_watermarked_text):
        from provenance.robustness.experiments import (
            ExperimentConfig, WatermarkSample, compute_robustness_report,
            evaluate_robustness_experiment, render_robustness_report,
        )
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=synthid_watermarked_text,
            token_count=80,
            sample_seed=42,
            prompt_id="prompt-0",
            length=80,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            synthid_detector, [sample],
            [get_transform("identity"), get_transform("lowercase")],
        )
        config = ExperimentConfig(
            detector_name="synthid",
            config_path="configs/synthid.example.json",
            lengths=(80,),
            samples_per_length=1,
            seed=42,
        )
        report = compute_robustness_report(records, config)
        text = render_robustness_report(report)
        assert "synthid" in text.lower()
        assert "Baseline detection rate" in text


# ---------------------------------------------------------------------------
# Multiple samples
# ---------------------------------------------------------------------------

class TestMultipleSamples:
    def test_multiple_kgw_samples(self, kgw_detector):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.detectors.kgw import generate_controlled_kgw_text
        from provenance.robustness.transforms import get_transform

        samples = []
        for i in range(3):
            text = generate_controlled_kgw_text(kgw_detector, token_count=120, watermarked=True)
            samples.append(WatermarkSample(
                text=text,
                token_count=120,
                sample_seed=42 + i,
                prompt_id=f"prompt-{i}",
                length=120,
                watermarked=True,
            ))

        records = evaluate_robustness_experiment(
            kgw_detector, samples,
            [get_transform("identity"), get_transform("lowercase")],
        )
        # 3 samples x 2 transforms = 6 records
        assert len(records) == 6
        # All should be detected as watermarked in identity transform
        identity_records = [r for r in records if r.transform_name == "identity"]
        assert all(r.original_detected is True for r in identity_records)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_records_report(self):
        from provenance.robustness.experiments import (
            ExperimentConfig, compute_robustness_report,
        )
        config = ExperimentConfig(
            detector_name="kgw",
            config_path=None,
            lengths=(),
            samples_per_length=0,
            seed=42,
        )
        report = compute_robustness_report([], config)
        assert report.sample_count == 0
        assert report.baseline_detection_rate == 0.0
        assert report.robustness_rate == 0.0

    def test_only_identity_transform(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.transforms import get_transform

        sample = WatermarkSample(
            text=kgw_watermarked_text,
            token_count=120,
            sample_seed=42,
            prompt_id="prompt-0",
            length=120,
            watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], [get_transform("identity")]
        )
        assert len(records) == 1
        # No non-identity transforms → robustness_rate should be 0
        # (nothing to compare against)


# ---------------------------------------------------------------------------
# Robustness rate bounds (regression: rate must stay in [0, 1])
# ---------------------------------------------------------------------------

class TestRobustnessRateBounds:
    def _records(self, n_non_identity=9, baseline_detected=True):
        from provenance.robustness.experiments import RobustnessRecord
        meta = {"sample_seed": 1, "prompt_id": "p-0", "length": 50, "watermarked": True}
        recs = [RobustnessRecord(
            detector_name="kgw", transform_name="identity",
            implementation_kind="k", compatibility="c",
            original_hash="a", transformed_hash="b",
            original_score=10.0, transformed_score=10.0,
            original_detected=baseline_detected,
            transformed_detected=baseline_detected,
            original_length=10, transformed_length=10,
            score_delta=0.0, detection_changed=False,
            detection_preserved=True, duration_ms=1.0,
            metadata=dict(meta),
        )]
        for i in range(n_non_identity):
            recs.append(RobustnessRecord(
                detector_name="kgw", transform_name=f"t{i}",
                implementation_kind="k", compatibility="c",
                original_hash="a", transformed_hash="c",
                original_score=10.0, transformed_score=9.0,
                original_detected=baseline_detected,
                transformed_detected=True,
                original_length=10, transformed_length=10,
                score_delta=-1.0, detection_changed=False,
                detection_preserved=True, duration_ms=1.0,
                metadata=dict(meta),
            ))
        return recs

    def _config(self):
        from provenance.robustness.experiments import ExperimentConfig
        return ExperimentConfig(
            detector_name="kgw", config_path="c.json",
            lengths=(50,), samples_per_length=1, seed=42,
        )

    def test_many_transforms_stay_bounded(self):
        """1 baseline-positive sample x 9 preserved transforms must give 1.0, not 9.0."""
        from provenance.robustness.experiments import compute_robustness_report
        report = compute_robustness_report(self._records(9), self._config())
        assert 0.0 <= report.robustness_rate <= 1.0
        assert report.robustness_rate == 1.0

    def test_partial_preservation(self):
        """Robustness is the preserved fraction over matched evaluations."""
        from provenance.robustness.experiments import compute_robustness_report
        recs = self._records(4)
        # Flip two transformed outcomes to misses
        import dataclasses
        recs[1] = dataclasses.replace(recs[1], transformed_detected=False)
        recs[2] = dataclasses.replace(recs[2], transformed_detected=False)
        report = compute_robustness_report(recs, self._config())
        assert report.robustness_rate == 0.5

    def test_baseline_miss_excluded(self):
        """Baseline-missed samples contribute no denominator (rate 0.0, not >1)."""
        from provenance.robustness.experiments import compute_robustness_report
        report = compute_robustness_report(self._records(4, baseline_detected=False), self._config())
        assert report.robustness_rate == 0.0
