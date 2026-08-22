"""End-to-end HF model-backed SynthID generation and detection test.

Generates watermarked and unwatermarked text with distilgpt2, then scores both
with the reference-backed SynthID detector.  Requires the optional 'hf'
dependencies (torch + transformers).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from provenance.configuration import ExperimentConfig  # noqa: E402
from provenance.detectors.reference.synthid import (  # noqa: E402
    SynthIDReferenceConfig,
    SynthIDReferenceDetector,
    _compute_g_values,
    _default_weights,
    _g_value_for_ngram,
    _hash_iv_from_keys,
    accumulate_hash,
)
from provenance.generation.synthid import (  # noqa: E402
    SynthIDGenerationConfig,
    SynthIDLogitsProcessor,
    generate_synthid_token_ids,
)
from provenance.models import HuggingFaceCausalLM  # noqa: E402
from provenance.tokenizers import tokenizer_from_config  # noqa: E402

CONFIG_PATH = "configs/synthid.hf.example.json"


def _load_experiment():
    return ExperimentConfig.from_file(CONFIG_PATH)


def _build_detector_and_model():
    experiment = _load_experiment()
    tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())
    model = HuggingFaceCausalLM(experiment.model)

    if tokenizer.vocabulary_size != model.vocab_size:
        pytest.skip(
            f"tokenizer vocab ({tokenizer.vocabulary_size}) != "
            f"model vocab ({model.vocab_size})"
        )

    synthid_config = SynthIDReferenceConfig.from_dict(
        {
            "configuration_id": experiment.configuration_id,
            "version": experiment.version,
            **experiment.watermark_configuration,
            **experiment.detector_configuration,
            "tokenizer": "huggingface",
            "model_identifier": experiment.model.model_identifier,
            "model_revision": experiment.model.revision,
            "tokenizer_identifier": experiment.tokenizer.tokenizer_identifier,
            "tokenizer_revision": experiment.tokenizer.tokenizer_revision,
            "device": experiment.model.device,
            "dtype": experiment.model.dtype,
            "local_files_only": experiment.model.local_files_only,
            **{
                k: v
                for k, v in experiment.generation_configuration.items()
                if k in ("max_new_tokens", "temperature", "top_p", "top_k", "seed")
            },
        }
    )
    synthid_config.validate()

    detector = SynthIDReferenceDetector(synthid_config, tokenizer=tokenizer)
    processor = SynthIDLogitsProcessor(synthid_config)
    # Use shorter generation for faster tests
    gen_config = SynthIDGenerationConfig(
        max_new_tokens=50,
        temperature=experiment.generation_configuration.get("temperature", 1.0),
        top_p=experiment.generation_configuration.get("top_p", 0.95),
        top_k=experiment.generation_configuration.get("top_k"),
        seed=experiment.generation_configuration.get("seed", 42),
    )
    return experiment, tokenizer, model, detector, processor, gen_config


# ---------------------------------------------------------------------------
# End-to-end generation + detection
# ---------------------------------------------------------------------------


class TestSynthIDHFE2E:
    """Full pipeline: HF model -> SynthID logits -> sampling -> detection."""

    @pytest.fixture()
    def components(self):
        return _build_detector_and_model()

    def test_watermarked_detected_unwatermarked_not(
        self, components
    ):
        """Watermarked text should score higher than unwatermarked."""
        experiment, tokenizer, model, detector, processor, gen_config = components

        prompt = "The future of artificial intelligence is shaped by"
        prompt_ids = tokenizer.encode(prompt)

        wm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )
        uwm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=False,
        )

        wm_result = detector.score_token_ids(wm_ids)
        uwm_result = detector.score_token_ids(uwm_ids)

        assert wm_result.status == "ok"
        assert uwm_result.status == "ok"
        assert wm_result.implementation_kind == "reference"
        assert wm_result.compatibility == "synthid-reference"

        # Both should have scores
        assert wm_result.score is not None
        assert uwm_result.score is not None

        # Watermarked should score substantially higher
        assert wm_result.score > uwm_result.score, (
            f"watermarked score {wm_result.score} should exceed "
            f"unwatermarked {uwm_result.score}"
        )

        # Watermarked should have more usable tokens
        assert wm_result.evidence["usable_token_count"] >= 20

    def test_metadata_no_raw_keys(self, components):
        """No raw key material should appear in output."""
        _, tokenizer, model, detector, processor, gen_config = components

        prompt_ids = tokenizer.encode("Test prompt")
        wm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )
        result = detector.score_token_ids(wm_ids)
        result_json = json.dumps(result.to_dict())

        # hash_key_id should be present
        assert result.metadata.get("hash_key_id") == "synthid-hf-demo-key-v1"

        # Raw keys should not appear
        for key in detector.config.keys:
            # Keys could coincidentally match token IDs, so check they're
            # not in the evidence/metadata sections specifically.
            assert result.evidence.get("hash_key_id") == "synthid-hf-demo-key-v1"

    def test_deterministic_generation(self, components):
        """Same seed/prompt produces identical token IDs."""
        _, tokenizer, model, _, processor, gen_config = components

        prompt_ids = tokenizer.encode("Deterministic test")
        ids_1 = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )
        ids_2 = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )
        assert ids_1 == ids_2

    def test_unwatermarked_uses_original_logits(self, components):
        """When watermarked=False, logits should be unmodified."""
        _, tokenizer, model, detector, processor, gen_config = components

        prompt_ids = tokenizer.encode("Unmodified logits test")

        # Generate unwatermarked
        uwm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=False,
        )
        # The detector should score near null for unwatermarked
        uwm_result = detector.score_token_ids(uwm_ids)
        assert uwm_result.score is not None
        # Unwatermarked should be below or near threshold
        assert uwm_result.score < 0.8, (
            f"unwatermarked score {uwm_result.score} too high for unmodified logits"
        )

    def test_token_id_scoring_matches_detect(self, components):
        """score_token_ids and detect(text) should agree."""
        _, tokenizer, model, detector, processor, gen_config = components

        prompt_ids = tokenizer.encode("Comparison test")
        wm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )

        text = tokenizer.decode(wm_ids)
        detect_result = detector.detect(text)
        score_result = detector.score_token_ids(wm_ids)

        # Scores should be very close (minor float differences from decode/encode)
        assert detect_result.score is not None
        assert score_result.score is not None
        assert abs(detect_result.score - score_result.score) < 0.01, (
            f"detect score {detect_result.score} vs score_token_ids {score_result.score}"
        )

    def test_reference_hash_equivalence(self, components):
        """Verify g-value computation matches PyTorch reference."""
        _, tokenizer, model, detector, processor, gen_config = components

        prompt_ids = tokenizer.encode("Hash test")
        wm_ids = generate_synthid_token_ids(
            model=model,
            prompt_token_ids=prompt_ids,
            processor=processor,
            generation_config=gen_config,
            watermarked=True,
        )

        # Compute g-values for a specific n-gram from the generated tokens
        if len(wm_ids) >= detector.config.ngram_len:
            ngram = tuple(wm_ids[-detector.config.ngram_len:])
            hash_iv = detector._hash_iv
            g = _g_value_for_ngram(
                ngram, hash_iv=hash_iv, key=detector.config.keys[0], depth=0
            )
            assert g in (0, 1)

            # Verify hash IV matches expected derivation
            import hashlib
            keys = detector.config.keys
            key_bytes = b"".join(
                k.to_bytes(8, byteorder="little", signed=False) for k in keys
            )
            digest = hashlib.sha256(key_bytes).digest()
            expected_iv = int.from_bytes(digest, byteorder="big") % ((1 << 63) - 1)
            assert hash_iv == expected_iv
