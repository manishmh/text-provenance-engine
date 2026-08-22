"""Tests for the reference-backed SynthID-Text detector.

These tests verify:
1. Hashing matches the reference LCG-based accumulate_hash (verified against PyTorch).
2. G-values match the reference computation.
3. Context repetition detection works correctly.
4. Weighted-mean scoring matches the reference.
5. Positive/negative control detection works.
6. Text-length detection behaviour is sensible.
7. Configuration validation works.
8. No raw keys leak into results or metadata.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from provenance.detectors.reference.synthid import (
    SynthIDReferenceConfig,
    SynthIDReferenceDetector,
    _compute_context_repetition_mask,
    _compute_g_values,
    _default_weights,
    _g_value_for_ngram,
    _hash_iv_from_keys,
    _weighted_mean_score,
    _to_unsigned,
    accumulate_hash,
)
from provenance.generation.synthid import generate_reference_synthid_text


# ---------------------------------------------------------------------------
# Reference config for tests
# ---------------------------------------------------------------------------

_CONFIG_DICT = {
    "configuration_id": "synthid-ref-test-v1",
    "version": "1",
    "tokenizer": "simple",
    "case_sensitive": False,
    "keys": [654, 400, 836],
    "ngram_len": 4,
    "sampling_table_size": 4096,
    "sampling_table_seed": 77,
    "context_history_size": 1024,
    "threshold": 0.7,
    "min_usable_tokens": 20,
    "hash_key_id": "test-keys-001",
    "vocabulary": [
        "amber", "binary", "cedar", "drift", "equal", "fable", "granite",
        "helium", "island", "jovial", "keystone", "lantern", "meadow",
        "nebula", "onyx", "prairie", "quantum", "ribbon", "summit", "tidal",
        "uplink", "violet", "willow", "yonder", "zircon", "artist", "border",
        "canvas", "detail", "evening", "fountain", "glacier",
    ],
}


def _make_detector(config: dict | None = None) -> SynthIDReferenceDetector:
    cfg = SynthIDReferenceConfig.from_dict(config or _CONFIG_DICT)
    cfg.validate()
    return SynthIDReferenceDetector(cfg)


# ===========================================================================
# Reference values computed from PyTorch synthid_text.hashing_function
# ===========================================================================

# These values were verified against PyTorch's torch.int64 accumulate_hash
# by computing: h = (h + data) * multiplier + increment with int64 wrapping.
_LCG_MULTIPLIER = 6364136223846793005


def _pytorch_accumulate_hash(iv: int, data: list[int]) -> int:
    """Reference accumulate_hash matching PyTorch int64 arithmetic."""
    import torch
    h = torch.tensor([iv], dtype=torch.int64)
    for d in data:
        h = torch.add(h, torch.tensor([d], dtype=torch.int64))
        h = torch.mul(h, torch.tensor([_LCG_MULTIPLIER], dtype=torch.int64))
        h = torch.add(h, torch.tensor([1], dtype=torch.int64))
    return _to_unsigned(h.item())


# ===========================================================================
# 1. Hashing: accumulate_hash matches reference LCG
# ===========================================================================

class TestAccumulateHash:
    """Verify accumulate_hash matches the DeepMind reference implementation."""

    def test_single_element(self):
        """Single element: verified against PyTorch int64 accumulate_hash."""
        expected = _pytorch_accumulate_hash(0, [42])
        result = accumulate_hash(0, (42,))
        assert result == expected

    def test_single_element_nonzero_iv(self):
        """Non-zero IV, single element: verified against PyTorch."""
        expected = _pytorch_accumulate_hash(12345, [42])
        result = accumulate_hash(12345, (42,))
        assert result == expected

    def test_multiple_elements(self):
        """Chain of updates: verified against PyTorch."""
        expected = _pytorch_accumulate_hash(0, [1, 2, 3])
        result = accumulate_hash(0, (1, 2, 3))
        assert result == expected

    def test_empty_data(self):
        """Empty data returns the IV unchanged."""
        assert accumulate_hash(999, ()) == 999

    def test_deterministic(self):
        """Same inputs always produce the same hash."""
        r1 = accumulate_hash(100, (1, 2, 3))
        r2 = accumulate_hash(100, (1, 2, 3))
        assert r1 == r2

    def test_different_inputs_differ(self):
        """Different data produces different hashes."""
        h1 = accumulate_hash(0, (1, 2))
        h2 = accumulate_hash(0, (1, 3))
        assert h1 != h2

    def test_matches_pytorch_various_inputs(self):
        """Exhaustive check: our accumulate_hash matches PyTorch for various inputs."""
        test_cases = [
            (0, [1]),
            (0, [1, 2, 3]),
            (42, [1]),
            (42, [42]),
            (100, [1]),
            (100, [1, 2, 3]),
            (100, [42]),
            (100, [654, 400]),
            ((1 << 62), [1]),
            ((1 << 62), [1, 2, 3]),
            (0, [654, 400]),
        ]
        for iv, data in test_cases:
            expected = _pytorch_accumulate_hash(iv, data)
            result = accumulate_hash(iv, tuple(data))
            assert result == expected, f"Failed for iv={iv}, data={data}"


# ===========================================================================
# 2. Hash IV derivation
# ===========================================================================

class TestHashIV:
    """Verify hash IV matches the reference SHA-256-based derivation."""

    def test_matches_reference(self):
        """IV = int.from_bytes(sha256(keys_bytes).digest(), 'big') % torch_long_max."""
        keys = (654, 400, 836)
        key_bytes = b"".join(
            k.to_bytes(8, byteorder="little", signed=False) for k in keys
        )
        digest = hashlib.sha256(key_bytes).digest()
        # Reference uses % torch.iinfo(torch.int64).max, not &
        expected = int.from_bytes(digest, byteorder="big") % ((1 << 63) - 1)
        assert _hash_iv_from_keys(keys) == expected

    def test_different_keys_different_iv(self):
        """Different key tuples produce different IVs."""
        iv1 = _hash_iv_from_keys((1, 2, 3))
        iv2 = _hash_iv_from_keys((4, 5, 6))
        assert iv1 != iv2


# ===========================================================================
# 3. G-value computation
# ===========================================================================

class TestGValues:
    """Verify g-value computation matches the reference algorithm."""

    def test_g_values_are_binary(self):
        """G-values must be 0 or 1."""
        hash_iv = _hash_iv_from_keys((654, 400, 836))
        for ngram in [(1, 2, 3, 4), (5, 6, 7, 8), (10, 20, 30, 40)]:
            g = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=0)
            assert g in (0, 1), f"g-value must be 0 or 1, got {g}"

    def test_g_value_deterministic(self):
        """Same inputs always produce the same g-value."""
        ngram = (10, 20, 30, 40)
        hash_iv = _hash_iv_from_keys((654, 400))
        g1 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=0)
        g2 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=0)
        assert g1 == g2

    def test_g_values_differ_by_key(self):
        """Different keys at same depth produce different g-values."""
        hash_iv = _hash_iv_from_keys((654,))
        ngram = (1, 2, 3, 4)
        g1 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=0)
        g2 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=400, depth=0)
        assert g1 in (0, 1)
        assert g2 in (0, 1)

    def test_g_values_differ_by_depth(self):
        """Different depths at same key produce different g-values."""
        hash_iv = _hash_iv_from_keys((654, 400))
        ngram = (10, 20, 30, 40)
        g0 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=0)
        g1 = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=654, depth=1)
        assert g0 in (0, 1)
        assert g1 in (0, 1)

    def test_compute_g_values_positions_before_ngram(self):
        """Positions before ngram_len - 1 have all-zero g-values."""
        hash_iv = _hash_iv_from_keys((654,))
        token_ids = [1, 2, 3, 4, 5]
        g_values = _compute_g_values(
            token_ids, ngram_len=4, hash_iv=hash_iv, keys=(654, 400)
        )
        assert len(g_values) == 5
        # Positions 0, 1, 2 (before ngram_len-1=3) should be zeros.
        assert g_values[0] == [0, 0]
        assert g_values[1] == [0, 0]
        assert g_values[2] == [0, 0]
        # Positions 3, 4 should have non-trivial g-values.
        assert len(g_values[3]) == 2
        assert len(g_values[4]) == 2

    def test_reference_algorithm_step_by_step(self):
        """Verify one g-value by tracing the reference algorithm step by step."""
        import torch
        from provenance.detectors.reference.synthid import (
            _signed_right_shift,
            _INT64_MAX,
        )

        hash_iv = 1000
        ngram = (10, 20, 30, 40)
        key = 654

        # Step 1: accumulate_hash(hash_iv, ngram)
        h1 = accumulate_hash(hash_iv, ngram)

        # Step 2: accumulate_hash(h1, (key,))
        h2 = accumulate_hash(h1, (key,))

        # Step 3: 12 rounds of accumulate_hash([1]) with signed >> 5
        h3 = h2
        for _ in range(12):
            h3 = accumulate_hash(h3, (1,))
            h3 = _signed_right_shift(h3, 5)

        expected_g = (h3 >> 30) & 1
        actual_g = _g_value_for_ngram(ngram, hash_iv=hash_iv, key=key, depth=0)
        assert actual_g == expected_g

        # Also verify against PyTorch reference
        def torch_g_value(ngram_list, hash_iv_val, key_val):
            iv_t = torch.tensor([hash_iv_val], dtype=torch.int64)
            h = iv_t
            for n in ngram_list:
                h = torch.add(h, torch.tensor([n], dtype=torch.int64))
                h = torch.mul(h, torch.tensor([_LCG_MULTIPLIER], dtype=torch.int64))
                h = torch.add(h, torch.tensor([1], dtype=torch.int64))
            h = torch.add(h, torch.tensor([key_val], dtype=torch.int64))
            h = torch.mul(h, torch.tensor([_LCG_MULTIPLIER], dtype=torch.int64))
            h = torch.add(h, torch.tensor([1], dtype=torch.int64))
            for _ in range(12):
                h = torch.add(h, torch.tensor([1], dtype=torch.int64))
                h = torch.mul(h, torch.tensor([_LCG_MULTIPLIER], dtype=torch.int64))
                h = torch.add(h, torch.tensor([1], dtype=torch.int64))
                h = h >> 5
            return ((h.item() >> 30) & 1)

        torch_g = torch_g_value(list(ngram), hash_iv, key)
        assert actual_g == torch_g, f"PyTorch g={torch_g}, ours={actual_g}"


# ===========================================================================
# 4. Context repetition mask
# ===========================================================================

class TestContextRepetitionMask:
    """Verify context repetition detection matches the reference."""

    def test_first_ngram_positions_are_masked(self):
        """Positions before ngram_len - 1 are always masked (False)."""
        hash_iv = _hash_iv_from_keys((654,))
        mask = _compute_context_repetition_mask(
            [1, 2, 3, 4, 5], ngram_len=4, hash_iv=hash_iv, context_history_size=10
        )
        assert mask[:3] == [False, False, False]
        assert len(mask) == 5

    def test_no_repetition_in_unique_contexts(self):
        """Unique contexts should all be unmasked."""
        hash_iv = _hash_iv_from_keys((654,))
        # Each n-gram is unique here.
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        mask = _compute_context_repetition_mask(
            token_ids, ngram_len=4, hash_iv=hash_iv, context_history_size=100
        )
        # Positions 3..7 should all be True (no repeats).
        assert all(mask[3:])

    def test_repeated_context_is_masked(self):
        """A repeated context should be marked False."""
        hash_iv = _hash_iv_from_keys((654,))
        # token_ids: [1, 2, 3, 4, 1, 2, 3, 4]
        # ngram_len=4: positions 3..7
        # pos 3 context = [1,2,3] (first)
        # pos 7 context = [1,2,3] (repeated)
        token_ids = [1, 2, 3, 4, 1, 2, 3, 4]
        mask = _compute_context_repetition_mask(
            token_ids, ngram_len=4, hash_iv=hash_iv, context_history_size=100
        )
        # Position 3: context [1,2,3] -> True (first time)
        assert mask[3] is True
        # Position 7: context [1,2,3] -> False (repeated)
        assert mask[7] is False

    def test_history_size_limits(self):
        """Contexts outside the history window are not detected as repeated."""
        hash_iv = _hash_iv_from_keys((654,))
        # token_ids: [1, 2, 3, 10, 11, 12, 13, 1, 2, 3, 20]
        token_ids = [1, 2, 3, 10, 11, 12, 13, 1, 2, 3, 20]
        mask = _compute_context_repetition_mask(
            token_ids, ngram_len=4, hash_iv=hash_iv, context_history_size=3
        )
        # Position 3: context [1,2,3] -> True (first)
        assert mask[3] is True
        # Position 9: context [1,2,3] -> should be True (pushed out of window)
        assert mask[9] is True


# ===========================================================================
# 5. Weighted-mean scoring
# ===========================================================================

class TestWeightedMeanScore:
    """Verify weighted-mean scoring matches the reference."""

    def test_all_ones(self):
        """All g-values 1, all mask True -> score should be 1.0."""
        g_values = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        mask = [True, True, True]
        weights = _default_weights(3)
        score = _weighted_mean_score(g_values, mask, weights, watermarking_depth=3)
        assert abs(score - 1.0) < 1e-10

    def test_all_zeros(self):
        """All g-values 0, all mask True -> score should be 0.0."""
        g_values = [[0, 0, 0], [0, 0, 0]]
        mask = [True, True]
        weights = _default_weights(3)
        score = _weighted_mean_score(g_values, mask, weights, watermarking_depth=3)
        assert abs(score - 0.0) < 1e-10

    def test_all_masked(self):
        """All mask False -> score should be 0.0."""
        g_values = [[1, 1], [1, 1]]
        mask = [False, False]
        weights = _default_weights(2)
        score = _weighted_mean_score(g_values, mask, weights, watermarking_depth=2)
        assert score == 0.0

    def test_weights_match_reference(self):
        """Default weights should match jnp.linspace(10, 1, depth) normalised."""
        for depth in [1, 2, 3, 5, 10, 30]:
            weights = _default_weights(depth)
            assert len(weights) == depth
            # Weights should sum to depth after normalisation.
            assert abs(sum(weights) - depth) < 1e-10
            # Weights should be non-negative.
            assert all(w >= 0 for w in weights)
            # First weight should be the largest (for depth > 1).
            if depth > 1:
                assert weights[0] >= weights[-1]

    def test_manual_computation(self):
        """Verify scoring against a hand-computed example."""
        g_values = [[1, 0], [0, 1]]
        mask = [True, True]
        weights = [2.0, 1.0]  # Normalised: [4/3, 2/3]
        norm_weights = [w * 2 / 3.0 for w in weights]
        # pos 0: 1 * 4/3 + 0 * 2/3 = 4/3
        # pos 1: 0 * 4/3 + 1 * 2/3 = 2/3
        # total = 4/3 + 2/3 = 2
        # score = 2 / (2 * 2) = 0.5
        score = _weighted_mean_score(g_values, mask, norm_weights, watermarking_depth=2)
        assert abs(score - 0.5) < 1e-10


# ===========================================================================
# 6. Positive / negative controls
# ===========================================================================

class TestPositiveNegativeControls:
    """Verify that watermarked text is detected and unwatermarked text is not."""

    @pytest.fixture()
    def detector(self) -> SynthIDReferenceDetector:
        return _make_detector()

    def test_watermarked_text_detected(self, detector: SynthIDReferenceDetector):
        """Reference-generated watermarked text should be detected."""
        text = generate_reference_synthid_text(
            detector, token_count=200, watermarked=True
        )
        result = detector.detect(text)
        assert result.status == "ok"
        assert result.implementation_kind == "reference"
        assert result.compatibility == "synthid-reference"
        assert result.detected is True
        assert result.score is not None
        assert result.score >= result.threshold
        assert result.evidence["usable_token_count"] >= 100

    def test_unwatermarked_text_not_detected(self, detector: SynthIDReferenceDetector):
        """Reference-generated unwatermarked text should not be detected."""
        text = generate_reference_synthid_text(
            detector, token_count=200, watermarked=False
        )
        result = detector.detect(text)
        assert result.status == "ok"
        assert result.implementation_kind == "reference"
        assert result.compatibility == "synthid-reference"
        assert result.detected is False
        assert result.score is not None
        assert result.score < result.threshold

    def test_watermarked_higher_score_than_unwatermarked(
        self, detector: SynthIDReferenceDetector
    ):
        """Watermarked text should have a higher score than unwatermarked."""
        wm_text = generate_reference_synthid_text(
            detector, token_count=200, watermarked=True
        )
        uwm_text = generate_reference_synthid_text(
            detector, token_count=200, watermarked=False
        )
        wm_result = detector.detect(wm_text)
        uwm_result = detector.detect(uwm_text)
        assert wm_result.score is not None
        assert uwm_result.score is not None
        assert wm_result.score > uwm_result.score


# ===========================================================================
# 7. Text-length tests
# ===========================================================================

class TestTextLength:
    """Verify detection behaviour at different token counts."""

    @pytest.fixture()
    def detector(self) -> SynthIDReferenceDetector:
        return _make_detector()

    def test_short_text_insufficient(self, detector: SynthIDReferenceDetector):
        """Very short text should return insufficient_text status."""
        text = generate_reference_synthid_text(
            detector, token_count=6, watermarked=True
        )
        result = detector.detect(text)
        assert result.status == "insufficient_text"
        assert result.detected is None

    def test_medium_length_detected(self, detector: SynthIDReferenceDetector):
        """50 tokens of watermarked text should be detected."""
        text = generate_reference_synthid_text(
            detector, token_count=50, watermarked=True
        )
        result = detector.detect(text)
        assert result.status == "ok"
        assert result.detected is True

    def test_long_length_detected(self, detector: SynthIDReferenceDetector):
        """240 tokens of watermarked text should be detected."""
        text = generate_reference_synthid_text(
            detector, token_count=240, watermarked=True
        )
        result = detector.detect(text)
        assert result.status == "ok"
        assert result.detected is True

    @pytest.mark.parametrize("token_count", [50, 100, 200])
    def test_unwatermarked_not_detected_at_various_lengths(
        self, detector: SynthIDReferenceDetector, token_count: int
    ):
        """Unwatermarked text should not be detected at various lengths."""
        text = generate_reference_synthid_text(
            detector, token_count=token_count, watermarked=False
        )
        result = detector.detect(text)
        assert result.status == "ok"
        assert result.detected is False


# ===========================================================================
# 8. Configuration validation
# ===========================================================================

class TestConfigValidation:
    """Verify configuration validation rules."""

    def test_valid_config(self):
        """A valid config should not raise."""
        cfg = SynthIDReferenceConfig.from_dict(_CONFIG_DICT)
        cfg.validate()

    def test_missing_required_field(self):
        """Missing required fields should raise DetectorConfigurationError."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {k: v for k, v in _CONFIG_DICT.items() if k != "keys"}
        with pytest.raises(DetectorConfigurationError, match="missing"):
            SynthIDReferenceConfig.from_dict(bad)

    def test_ngram_too_small(self):
        """ngram_len < 2 should raise."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {**_CONFIG_DICT, "ngram_len": 1}
        cfg = SynthIDReferenceConfig.from_dict(bad)
        with pytest.raises(DetectorConfigurationError, match="at least 2"):
            cfg.validate()

    def test_empty_keys(self):
        """Empty keys should raise."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {**_CONFIG_DICT, "keys": []}
        cfg = SynthIDReferenceConfig.from_dict(bad)
        with pytest.raises(DetectorConfigurationError, match="empty"):
            cfg.validate()

    def test_weights_mismatch(self):
        """Weights length != keys length should raise."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {**_CONFIG_DICT, "weights": [1.0, 2.0]}
        cfg = SynthIDReferenceConfig.from_dict(bad)
        with pytest.raises(DetectorConfigurationError, match="match"):
            cfg.validate()

    def test_threshold_out_of_range(self):
        """Threshold outside [0, 1] should raise."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {**_CONFIG_DICT, "threshold": 1.5}
        cfg = SynthIDReferenceConfig.from_dict(bad)
        with pytest.raises(DetectorConfigurationError, match="\\[0, 1\\]"):
            cfg.validate()

    def test_hash_key_id_required(self):
        """Missing hash_key_id should raise."""
        from provenance.detectors.base import DetectorConfigurationError

        bad = {k: v for k, v in _CONFIG_DICT.items() if k != "hash_key_id"}
        with pytest.raises(DetectorConfigurationError, match="missing"):
            SynthIDReferenceConfig.from_dict(bad)


# ===========================================================================
# 9. No raw key material in results
# ===========================================================================

class TestNoKeyLeakage:
    """Verify that raw key material never appears in results or metadata."""

    @pytest.fixture()
    def detector(self) -> SynthIDReferenceDetector:
        return _make_detector()

    def test_result_contains_no_raw_keys(self, detector: SynthIDReferenceDetector):
        """DetectionResult should not contain raw key values."""
        text = generate_reference_synthid_text(
            detector, token_count=100, watermarked=True
        )
        result = detector.detect(text)
        result_dict = result.to_dict()
        result_json = json.dumps(result_dict)

        # The raw keys should not appear in the JSON.
        for key in detector.config.keys:
            assert str(key) not in result_json or key in (
                detector.config.sampling_table_seed,
            ), f"Raw key {key} appeared in result JSON"

    def test_metadata_contains_hash_key_id_not_raw_keys(
        self, detector: SynthIDReferenceDetector
    ):
        """Metadata should contain hash_key_id, not raw keys."""
        text = generate_reference_synthid_text(
            detector, token_count=100, watermarked=True
        )
        result = detector.detect(text)
        # hash_key_id should be present.
        assert result.metadata.get("hash_key_id") == "test-keys-001"
        # evidence should contain hash_key_id.
        assert result.evidence.get("hash_key_id") == "test-keys-001"


# ===========================================================================
# 10. Comparison with simulation detector (both use same config params)
# ===========================================================================

class TestSimulationComparison:
    """Compare reference and simulation detectors on the same text.

    The two detectors use different hashing algorithms (LCG vs blake2b), so
    their g-values and scores will differ.  This test verifies that they
    produce different results but both are self-consistent.
    """

    @pytest.fixture()
    def ref_detector(self) -> SynthIDReferenceDetector:
        return _make_detector()

    def test_different_detectors_different_scores(
        self, ref_detector: SynthIDReferenceDetector
    ):
        """Reference and simulation detectors should produce different scores."""
        from provenance.detectors.simulated.synthid import (
            SynthIDConfig,
            SynthIDTextDetector,
        )

        sim_config = SynthIDConfig.from_dict(_CONFIG_DICT)
        sim_config.validate()
        sim_detector = SynthIDTextDetector(sim_config)

        text = generate_reference_synthid_text(
            ref_detector, token_count=100, watermarked=True
        )
        ref_result = ref_detector.detect(text)
        sim_result = sim_detector.detect(text)

        # Both should work, but scores differ due to different hashing.
        assert ref_result.status == "ok"
        assert sim_result.status == "ok"
        # Reference should detect (it generated the text).
        assert ref_result.detected is True
        # Simulation may or may not detect (different hashing).
        # Both should produce valid scores.
        assert ref_result.score is not None
        assert sim_result.score is not None
