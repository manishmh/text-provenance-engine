import json
import subprocess
import sys

from provenance.configuration import ExperimentConfig
from provenance.detectors.reference.kgw import KGWReferenceConfig, KGWReferenceDetector, KGWReferenceScorer
from provenance.generation import KGWGenerationConfig, generate_kgw_token_ids
from provenance.models.local import DeterministicToyCausalLM
from provenance.tokenizers import tokenizer_from_config


CONFIG_PATH = "configs/kgw.reference.example.json"


def _reference_components():
    experiment = ExperimentConfig.from_file(CONFIG_PATH)
    tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())
    config = KGWReferenceConfig.from_dict(
        {
            "configuration_id": experiment.configuration_id,
            "version": experiment.version,
            **experiment.watermark_configuration,
            **experiment.detector_configuration,
        }
    )
    scorer = KGWReferenceScorer(config, vocab_size=tokenizer.vocabulary_size)
    model = DeterministicToyCausalLM(
        vocab_size=tokenizer.vocabulary_size,
        identifier=experiment.model.model_identifier,
    )
    generation = KGWGenerationConfig(**experiment.generation_configuration)
    return tokenizer, config, scorer, model, generation


def test_reference_kgw_model_backed_generation_positive_and_negative():
    tokenizer, config, scorer, model, generation = _reference_components()
    prompt_ids = tokenizer.encode("anchor beacon circuit delta")
    detector = KGWReferenceDetector(config, tokenizer)

    watermarked_ids = generate_kgw_token_ids(
        model=model,
        prompt_token_ids=prompt_ids,
        scorer=scorer,
        generation_config=generation,
        watermarked=True,
    )
    unwatermarked_ids = generate_kgw_token_ids(
        model=model,
        prompt_token_ids=prompt_ids,
        scorer=scorer,
        generation_config=KGWGenerationConfig(
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_k=generation.top_k,
            seed=generation.seed + 1,
        ),
        watermarked=False,
    )

    watermarked = detector.detect(tokenizer.decode(watermarked_ids))
    unwatermarked = detector.detect(tokenizer.decode(unwatermarked_ids))

    # Toy tokenizer -> the KGW algorithm running over controlled token IDs, which
    # must not be advertised as a real model/tokenizer-backed reference run.
    assert watermarked.implementation_kind == "reference-adapted"
    assert watermarked.compatibility == "controlled-local-only"
    assert watermarked.detected is True
    assert watermarked.evidence["p_value"] < 1e-6
    assert unwatermarked.detected is False
    assert unwatermarked.evidence["green_fraction"] < 0.4


def test_reference_kgw_generated_fixtures_are_detected():
    detector = KGWReferenceDetector.from_config_file(CONFIG_PATH)

    watermarked_text = open("data/generated/kgw/watermarked.txt", encoding="utf-8").read()
    unwatermarked_text = open("data/generated/kgw/unwatermarked.txt", encoding="utf-8").read()

    assert detector.detect(watermarked_text).detected is True
    assert detector.detect(unwatermarked_text).detected is False


def test_reference_kgw_cli_json_on_fixture():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "provenance",
            "analyze",
            "data/generated/kgw/watermarked.txt",
            "--detector",
            "kgw-reference",
            "--config",
            CONFIG_PATH,
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    result = payload["results"][0]
    assert result["implementation_kind"] == "reference-adapted"
    assert result["compatibility"] == "controlled-local-only"
    assert result["evidence"]["scheme"] == "kgw"
    assert result["detected"] is True
