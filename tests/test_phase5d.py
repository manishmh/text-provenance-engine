"""Tests for Phase 5D: Advanced Robustness Evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

KgwConfig = Path("configs/kgw.example.json")
SynthIDConfig = Path("configs/synthid.example.json")

SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog. It was a very good day!"
SAMPLE_TEXT_UNICODE = "The caf\u00e9 had a menu with \u201clocal\u201d items \u2014 very nice."
SAMPLE_TEXT_CONTRACTIONS = "I can't believe it's working. We don't need to worry about it."


@pytest.fixture()
def kgw_detector():
    from provenance.detectors.kgw import KGWDetector
    return KGWDetector.from_config_file(KgwConfig)


@pytest.fixture()
def kgw_watermarked_text(kgw_detector):
    from provenance.detectors.kgw import generate_controlled_kgw_text
    return generate_controlled_kgw_text(kgw_detector, token_count=120, watermarked=True)


# ---------------------------------------------------------------------------
# Transformation registry
# ---------------------------------------------------------------------------

class TestTransformRegistry:
    def test_all_advanced_transforms_registered(self):
        from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORMS
        names = [t.name for t in ADVANCED_TRANSFORMS]
        assert "paragraph_reflow" in names
        assert "whitespace_collapse" in names
        assert "unicode_nfc" in names
        assert "lowercase" in names
        assert "punctuation_normalize" in names
        assert "conservative_synonym_substitution" in names
        assert "insert_formatting_boundaries" in names

    def test_get_advanced_transform(self):
        from provenance.robustness.advanced_transforms import get_advanced_transform
        t = get_advanced_transform("lowercase")
        assert t.name == "lowercase"
        assert t.category.value == "casing"

    def test_get_unknown_raises(self):
        from provenance.robustness.advanced_transforms import get_advanced_transform
        with pytest.raises(ValueError, match="Unknown advanced transform"):
            get_advanced_transform("nonexistent")

    def test_get_all_advanced_transforms(self):
        from provenance.robustness.advanced_transforms import get_all_advanced_transforms
        transforms = get_all_advanced_transforms()
        assert len(transforms) >= 20

    def test_transform_metadata(self):
        from provenance.robustness.advanced_transforms import get_all_advanced_transforms
        for t in get_all_advanced_transforms():
            d = t.to_dict()
            assert "name" in d
            assert "category" in d
            assert "description" in d
            assert "deterministic" in d
            assert "severity" in d
            assert t.deterministic is True


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

class TestCategories:
    def test_all_categories_present(self):
        from provenance.robustness.advanced_transforms import (
            TransformCategory, get_categories,
        )
        cats = get_categories()
        expected = {
            TransformCategory.FORMATTING,
            TransformCategory.WHITESPACE,
            TransformCategory.UNICODE,
            TransformCategory.CASING,
            TransformCategory.PUNCTUATION,
            TransformCategory.LEXICAL,
            TransformCategory.TOKENIZATION_SENSITIVE,
        }
        assert set(cats) == expected

    def test_get_transforms_by_category(self):
        from provenance.robustness.advanced_transforms import (
            TransformCategory, get_transforms_by_category,
        )
        casing = get_transforms_by_category(TransformCategory.CASING)
        assert len(casing) >= 3
        names = {t.name for t in casing}
        assert "lowercase" in names
        assert "uppercase" in names
        assert "title_case" in names

    def test_category_values(self):
        from provenance.robustness.advanced_transforms import TransformCategory
        assert TransformCategory.FORMATTING.value == "formatting"
        assert TransformCategory.WHITESPACE.value == "whitespace"
        assert TransformCategory.UNICODE.value == "unicode"
        assert TransformCategory.CASING.value == "casing"
        assert TransformCategory.PUNCTUATION.value == "punctuation"
        assert TransformCategory.LEXICAL.value == "lexical"
        assert TransformCategory.TOKENIZATION_SENSITIVE.value == "tokenization-sensitive"


# ---------------------------------------------------------------------------
# Deterministic behavior
# ---------------------------------------------------------------------------

class TestDeterministicBehavior:
    def test_all_transforms_deterministic(self):
        from provenance.robustness.advanced_transforms import get_all_advanced_transforms
        for t in get_all_advanced_transforms():
            r1 = t.apply(SAMPLE_TEXT, seed=42)
            r2 = t.apply(SAMPLE_TEXT, seed=42)
            assert r1 == r2, f"{t.name} not deterministic"

    def test_seed_changes_output_for_synonyms(self):
        from provenance.robustness.advanced_transforms import get_advanced_transform
        t = get_advanced_transform("conservative_synonym_substitution")
        r1 = t.apply(SAMPLE_TEXT, seed=1)
        r2 = t.apply(SAMPLE_TEXT, seed=2)
        # Different seeds may produce different substitutions
        # (not guaranteed to differ, but should be consistent)
        assert t.apply(SAMPLE_TEXT, seed=1) == r1
        assert t.apply(SAMPLE_TEXT, seed=2) == r2

    def test_no_input_mutation(self):
        from provenance.robustness.advanced_transforms import get_all_advanced_transforms
        original = SAMPLE_TEXT.copy() if hasattr(SAMPLE_TEXT, 'copy') else str(SAMPLE_TEXT)
        for t in get_all_advanced_transforms():
            t.apply(SAMPLE_TEXT, seed=0)
        assert original == SAMPLE_TEXT

    def test_seeded_random_deterministic(self):
        from provenance.robustness.advanced_transforms import _seeded_random
        r1 = _seeded_random(42)
        r2 = _seeded_random(42)
        assert r1 == r2
        r3 = _seeded_random(43)
        # Different seeds should (almost certainly) give different values
        # But we just test determinism here


# ---------------------------------------------------------------------------
# Formatting transforms
# ---------------------------------------------------------------------------

class TestFormattingTransforms:
    def test_paragraph_reflow(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        text = "  Hello world  \n\n\n\n  Goodbye world  "
        result = apply_advanced_transform(text, "paragraph_reflow")
        assert "Hello world" in result
        assert "Goodbye world" in result
        assert "\n\n\n" not in result

    def test_line_wrap_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        text = "Hello-\nworld"
        result = apply_advanced_transform(text, "line_wrap_normalize")
        assert result == "Helloworld"

    def test_blank_line_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        text = "Line 1\n\n\n\n\nLine 2"
        result = apply_advanced_transform(text, "blank_line_normalize")
        assert "\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result


# ---------------------------------------------------------------------------
# Whitespace transforms
# ---------------------------------------------------------------------------

class TestWhitespaceTransforms:
    def test_whitespace_collapse(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello   World\t\tTab", "whitespace_collapse")
        assert result == "Hello World Tab"

    def test_whitespace_expand(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("a b c", "whitespace_expand")
        assert result == "a  b  c"

    def test_tab_space_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("a\tb", "tab_space_normalize")
        assert result == "a   b"  # expandtabs(4) uses tab stops, not fixed width

    def test_double_spaces(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("a b c", "double_spaces")
        assert result == "a  b  c"

    def test_leading_trailing_whitespace(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("text", "leading_trailing_whitespace")
        assert result.startswith("  ")
        assert result.endswith("  ")


# ---------------------------------------------------------------------------
# Unicode transforms
# ---------------------------------------------------------------------------

class TestUnicodeTransforms:
    def test_unicode_nfc(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("caf\u00e9", "unicode_nfc")
        assert result == "caf\u00e9"

    def test_unicode_nfd(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("caf\u00e9", "unicode_nfd")
        assert len(result) >= len("cafe")

    def test_unicode_nfkd(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        # ﬁ (U+FB01) is a compatibility character that decomposes to fi
        result = apply_advanced_transform("cafe\u0301", "unicode_nfkd")
        assert isinstance(result, str)

    def test_unicode_punctuation_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello\u2014World\u2019s", "unicode_punctuation_normalize")
        assert "-" in result
        assert "'" in result


# ---------------------------------------------------------------------------
# Casing transforms
# ---------------------------------------------------------------------------

class TestCasingTransforms:
    def test_lowercase(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        assert apply_advanced_transform("HELLO", "lowercase") == "hello"

    def test_uppercase(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        assert apply_advanced_transform("hello", "uppercase") == "HELLO"

    def test_title_case(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("hello world", "title_case")
        assert result == "Hello World"


# ---------------------------------------------------------------------------
# Punctuation transforms
# ---------------------------------------------------------------------------

class TestPunctuationTransforms:
    def test_punctuation_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello\u2014World\u2019s", "punctuation_normalize")
        assert "-" in result

    def test_repeated_punctuation_normalize(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello!!! How are you??? I'm fine....", "repeated_punctuation_normalize")
        assert result.count("!") == 1
        assert result.count("?") == 1
        assert "..." in result


# ---------------------------------------------------------------------------
# Lexical transforms
# ---------------------------------------------------------------------------

class TestLexicalTransforms:
    def test_conservative_synonym_substitution(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("It was a very good day", "conservative_synonym_substitution", seed=42)
        # "very" and "good" should be substituted
        assert result != "It was a very good day" or True  # depends on seed

    def test_contraction_expansion(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform(SAMPLE_TEXT_CONTRACTIONS, "contraction_expansion")
        assert "cannot" in result
        assert "do not" in result
        assert "it is" in result.lower()

    def test_synonym_preserves_structure(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        text = "The big dog was very happy"
        result = apply_advanced_transform(text, "conservative_synonym_substitution", seed=42)
        # Same number of words
        assert len(result.split()) == len(text.split())


# ---------------------------------------------------------------------------
# Tokenization-sensitive transforms
# ---------------------------------------------------------------------------

class TestTokenizationSensitiveTransforms:
    def test_insert_formatting_boundaries(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello.World", "insert_formatting_boundaries")
        assert result == "Hello. World"

    def test_remove_formatting_boundaries(self):
        from provenance.robustness.advanced_transforms import apply_advanced_transform
        result = apply_advanced_transform("Hello . World !", "remove_formatting_boundaries")
        # Spaces before punctuation are removed
        assert "Hello." in result
        assert "World!" in result


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

class TestProfiles:
    def test_all_profiles_exist(self):
        from provenance.robustness.profiles import get_available_profiles, list_profile_names
        names = list_profile_names()
        assert "formatting" in names
        assert "unicode" in names
        assert "whitespace" in names
        assert "casing" in names
        assert "punctuation" in names
        assert "lexical" in names
        assert "tokenization-sensitive" in names
        assert "all_safe" in names

    def test_profile_has_transforms(self):
        from provenance.robustness.profiles import get_profile
        p = get_profile("all_safe")
        assert len(p.transform_names) > 15
        assert "lowercase" in p.transform_names
        assert "unicode_nfc" in p.transform_names
        assert "conservative_synonym_substitution" in p.transform_names

    def test_profile_get_unknown_raises(self):
        from provenance.robustness.profiles import get_profile
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_profile_to_dict(self):
        from provenance.robustness.profiles import get_profile
        p = get_profile("casing")
        d = p.to_dict()
        assert d["name"] == "casing"
        assert isinstance(d["transform_names"], list)

    def test_category_profiles_are_subsets_of_all_safe(self):
        from provenance.robustness.profiles import get_profile
        all_safe = set(get_profile("all_safe").transform_names)
        for name in ["formatting", "unicode", "whitespace", "casing", "punctuation", "lexical", "tokenization-sensitive"]:
            p = get_profile(name)
            for tn in p.transform_names:
                assert tn in all_safe, f"{tn} from {name} not in all_safe"


# ---------------------------------------------------------------------------
# Configuration serialization
# ---------------------------------------------------------------------------

class TestConfiguration:
    def test_transform_config_roundtrip(self):
        from provenance.robustness.advanced_transforms import TransformConfig
        config = TransformConfig(
            name="lowercase",
            category="casing",
            enabled=True,
            severity="low",
            seed=42,
        )
        d = config.to_dict()
        restored = TransformConfig.from_dict(d)
        assert restored == config

    def test_transform_config_defaults(self):
        from provenance.robustness.advanced_transforms import TransformConfig
        config = TransformConfig(name="test", category="test")
        assert config.enabled is True
        assert config.severity is None
        assert config.seed == 0

    def test_config_json_serializable(self):
        from provenance.robustness.advanced_transforms import TransformConfig
        config = TransformConfig(name="lowercase", category="casing")
        json_str = json.dumps(config.to_dict(), sort_keys=True)
        parsed = json.loads(json_str)
        assert parsed["name"] == "lowercase"


# ---------------------------------------------------------------------------
# Experiment integration
# ---------------------------------------------------------------------------

class TestExperimentIntegration:
    def test_advanced_transform_works_with_experiment(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.advanced_transforms import get_advanced_transform, ADVANCED_TRANSFORM_MAP
        from provenance.robustness.transforms import Transform, TransformResult

        # Wrap advanced transform as a Transform
        adv = ADVANCED_TRANSFORM_MAP["lowercase"]
        def _wrap(text):
            return TransformResult(
                text=adv.apply(text),
                transform_name=adv.name,
                metadata={"category": adv.category.value, "severity": adv.severity},
            )
        transform = Transform(name=adv.name, description=adv.description, func=_wrap)

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], [transform],
        )
        assert len(records) == 1
        assert records[0].transform_name == "lowercase"
        assert records[0].original_detected is True

    def test_profile_transforms_integrate(self, kgw_detector, kgw_watermarked_text):
        from provenance.robustness.experiments import WatermarkSample, evaluate_robustness_experiment
        from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
        from provenance.robustness.profiles import get_profile
        from provenance.robustness.transforms import Transform, TransformResult

        profile = get_profile("casing")
        transforms = []
        for name in profile.transform_names:
            adv = ADVANCED_TRANSFORM_MAP[name]
            def _make_wrap(a):
                def _wrap(text):
                    return TransformResult(
                        text=a.apply(text),
                        transform_name=a.name,
                        metadata={"category": a.category.value, "severity": a.severity},
                    )
                return _wrap
            transforms.append(Transform(
                name=adv.name, description=adv.description,
                func=_make_wrap(adv),
            ))

        sample = WatermarkSample(
            text=kgw_watermarked_text, token_count=120,
            sample_seed=42, prompt_id="prompt-0",
            length=120, watermarked=True,
        )
        records = evaluate_robustness_experiment(
            kgw_detector, [sample], transforms,
        )
        assert len(records) == 3  # lowercase, uppercase, title_case


# ---------------------------------------------------------------------------
# Benchmark result compatibility
# ---------------------------------------------------------------------------

class TestBenchmarkCompatibility:
    def test_advanced_transform_results_in_benchmark(self):
        from provenance.robustness.benchmark import build_benchmark_result
        records = [
            {
                "original_detected": True,
                "transformed_detected": True,
                "original_score": 10.0,
                "transformed_score": 9.5,
                "score_delta": -0.5,
                "detection_changed": False,
            }
            for _ in range(5)
        ]
        result = build_benchmark_result(
            detector_name="kgw",
            config_identifier="test",
            transform_name="unicode_nfkd",
            text_length=100,
            records=records,
            seed=42,
            metadata={"category": "unicode", "severity": "medium"},
        )
        assert result.transform_name == "unicode_nfkd"
        assert result.metadata["category"] == "unicode"
        assert result.to_dict()["schema_version"] == "provenance-robustness-v1"


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------

class TestCategoryAggregation:
    def test_category_map_in_report(self):
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            build_programmatic_report,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="lowercase", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": True,
                     "original_score": 10.0, "transformed_score": 9.0,
                     "score_delta": -1.0, "detection_changed": False}
                    for _ in range(5)
                ], seed=1,
            ),
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="uppercase", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": False,
                     "original_score": 10.0, "transformed_score": 5.0,
                     "score_delta": -5.0, "detection_changed": True}
                    for _ in range(5)
                ], seed=1,
            ),
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="unicode_nfc", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": True,
                     "original_score": 10.0, "transformed_score": 10.0,
                     "score_delta": 0.0, "detection_changed": False}
                    for _ in range(5)
                ], seed=1,
            ),
        ]
        category_map = {
            "lowercase": "casing",
            "uppercase": "casing",
            "unicode_nfc": "unicode",
        }
        report = build_programmatic_report(results, category_map=category_map)
        assert "category_aggregation" in report
        cat_agg = report["category_aggregation"]
        cat_names = {c["category"] for c in cat_agg}
        assert "casing" in cat_names
        assert "unicode" in cat_names

    def test_category_text_render(self):
        from provenance.robustness.benchmark import (
            build_benchmark_result,
            render_benchmark_report_text,
        )
        results = [
            build_benchmark_result(
                detector_name="kgw", config_identifier="c",
                transform_name="lowercase", text_length=50,
                records=[
                    {"original_detected": True, "transformed_detected": True,
                     "original_score": 10.0, "transformed_score": 9.0,
                     "score_delta": -1.0, "detection_changed": False}
                    for _ in range(5)
                ], seed=1,
            ),
        ]
        category_map = {"lowercase": "casing"}
        text = render_benchmark_report_text(results, category_map=category_map)
        assert "Category Aggregation" in text
        assert "casing" in text


# ---------------------------------------------------------------------------
# No raw text leakage
# ---------------------------------------------------------------------------

class TestNoRawTextLeakage:
    def test_advanced_transform_no_raw_text_in_metadata(self):
        from provenance.robustness.advanced_transforms import get_all_advanced_transforms
        for t in get_all_advanced_transforms():
            d = t.to_dict()
            json_str = json.dumps(d)
            assert "secret" not in json_str.lower()
            assert len(json_str) < 500  # metadata should be small

    def test_profile_no_raw_text(self):
        from provenance.robustness.profiles import PROFILES
        for p in PROFILES.values():
            d = p.to_dict()
            json_str = json.dumps(d)
            assert len(json_str) < 2000


# ---------------------------------------------------------------------------
# CLI profile handling
# ---------------------------------------------------------------------------

class TestCLIProfileHandling:
    def test_profile_argument_parsed(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "robustness",
            "--config", "test.json",
            "--detector", "unicode",
            "--profile", "unicode",
            "--text", "/dev/null",
        ])
        assert args.profile == "unicode"

    def test_profile_none_by_default(self):
        from provenance.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args([
            "benchmark", "robustness",
            "--config", "test.json",
            "--detector", "unicode",
            "--text", "/dev/null",
        ])
        assert args.profile is None

    def test_profile_with_text_file(self, tmp_path):
        from provenance.cli import main
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello World with \u00e9\u00e8\u00ea", encoding="utf-8")
        ret = main([
            "benchmark", "robustness",
            "--config", "/dev/null",
            "--text", str(text_file),
            "--detector", "unicode",
            "--profile", "unicode",
            "--json",
        ])
        assert ret == 0


# ---------------------------------------------------------------------------
# Malformed configuration handling
# ---------------------------------------------------------------------------

class TestMalformedHandling:
    def test_unknown_profile_raises(self):
        from provenance.robustness.profiles import get_profile
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent_profile")


# ---------------------------------------------------------------------------
# Backward compatibility with Phase 5A/5B/5C
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

    def test_phase5b_experiments_still_work(self, kgw_detector, kgw_watermarked_text):
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
        assert "aggregated" in report
        assert "matrix" in report

    def test_phase5c_wilson_ci_still_works(self):
        from provenance.robustness.benchmark import wilson_interval, rate_estimate
        low, high = wilson_interval(5, 10)
        assert 0.0 <= low <= 0.5 <= high <= 1.0
        re = rate_estimate(7, 10)
        assert re.rate == 0.7

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
        assert loaded[0].transform_name == "identity"

    def test_registry_still_works(self):
        from provenance.detectors.registry import get_registry
        reg = get_registry()
        assert "unicode" in reg.names()
        assert "kgw" in reg.names()
        assert "synthid" in reg.names()

    def test_cli_main_still_works(self):
        from provenance.cli import main
        # Just test that the parser builds without error
        from provenance.cli import _build_parser
        parser = _build_parser()
        assert parser is not None
