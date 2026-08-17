#!/usr/bin/env python3
"""Generate a small controlled KGW reference validation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from provenance.configuration import ExperimentConfig  # noqa: E402
from provenance.detectors.reference.kgw import KGWReferenceConfig, KGWReferenceScorer  # noqa: E402
from provenance.generation import KGWGenerationConfig, generate_kgw_token_ids  # noqa: E402
from provenance.models.local import DeterministicToyCausalLM  # noqa: E402
from provenance.tokenizers import tokenizer_from_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled KGW reference samples")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/kgw"))
    parser.add_argument("--prompt", default="anchor beacon circuit delta")
    parser.add_argument("--prompt-id", default="kgw-reference-demo-prompt-1")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_file(args.config)
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
    prompt_token_ids = tokenizer.encode(args.prompt)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for watermarked, stem, seed_offset in [(True, "watermarked", 0), (False, "unwatermarked", 1)]:
        generation_for_sample = KGWGenerationConfig(
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_k=generation.top_k,
            seed=generation.seed + seed_offset,
        )
        generated_ids = generate_kgw_token_ids(
            model=model,
            prompt_token_ids=prompt_token_ids,
            scorer=scorer,
            generation_config=generation_for_sample,
            watermarked=watermarked,
        )
        text = tokenizer.decode(generated_ids)
        score = scorer.score_token_ids(generated_ids)
        text_path = args.out_dir / f"{stem}.txt"
        metadata_path = args.out_dir / f"{stem}.metadata.json"
        text_path.write_text(text, encoding="utf-8")
        metadata = {
            "watermarked": watermarked,
            "scheme": "kgw",
            "configuration_id": config.configuration_id,
            "configuration_version": config.version,
            "detector_reference_version": "kgw-python-left-v1",
            "model": model.identifier,
            "tokenizer": tokenizer.identifier,
            "prompt_id": args.prompt_id,
            "prompt": args.prompt,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_ids,
            "token_count": len(generated_ids),
            "generation_parameters": {
                "max_new_tokens": generation_for_sample.max_new_tokens,
                "temperature": generation_for_sample.temperature,
                "top_k": generation_for_sample.top_k,
                "seed": generation_for_sample.seed,
                "watermark_processor": "kgw-python-left-v1" if watermarked else None,
            },
            "detector_score": {
                "z_score": score.z_score,
                "p_value": score.p_value,
                "green_token_count": score.green_token_count,
                "scored_token_count": score.scored_token_count,
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
