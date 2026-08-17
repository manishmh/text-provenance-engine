#!/usr/bin/env python3
"""Generate genuine model/tokenizer-backed KGW samples.

Pipeline (no green tokens are ever selected directly):

    prompt -> HF tokenizer -> HF causal LM -> logits
           -> KGW logits processor -> sampling -> token IDs -> text

Requires the optional 'hf' dependencies (torch + transformers). The model and
tokenizer are downloaded from the Hugging Face Hub on first use; nothing is
downloaded at package installation time.
"""

from __future__ import annotations

import argparse
import hashlib
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
from provenance.models import HuggingFaceCausalLM  # noqa: E402
from provenance.tokenizers import tokenizer_from_config  # noqa: E402


def _hash_key_id(watermark: dict, hash_key: int) -> str:
    """Public identifier for the hash key. Never expose the raw secret key."""
    if watermark.get("hash_key_id"):
        return str(watermark["hash_key_id"])
    return "sha256:" + hashlib.sha256(str(hash_key).encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate model-backed KGW samples")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated/kgw/reference"))
    parser.add_argument(
        "--prompt",
        default="The history of lighthouses along the northern coast is a story of",
    )
    parser.add_argument("--prompt-id", default="kgw-hf-demo-prompt-1")
    args = parser.parse_args()

    experiment = ExperimentConfig.from_file(args.config)

    tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())
    model = HuggingFaceCausalLM(experiment.model)

    if tokenizer.vocabulary_size != model.vocab_size:
        raise SystemExit(
            "tokenizer vocabulary size "
            f"({tokenizer.vocabulary_size}) must equal model vocab size "
            f"({model.vocab_size}); the detector scores exact model token IDs."
        )

    kgw_config = KGWReferenceConfig.from_dict(
        {
            "configuration_id": experiment.configuration_id,
            "version": experiment.version,
            **experiment.watermark_configuration,
            **experiment.detector_configuration,
        }
    )
    scorer = KGWReferenceScorer(kgw_config, vocab_size=tokenizer.vocabulary_size)
    generation = KGWGenerationConfig(**experiment.generation_configuration)
    prompt_token_ids = tokenizer.encode(args.prompt)

    hash_key_id = _hash_key_id(experiment.watermark_configuration, kgw_config.hash_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for watermarked, stem in [(True, "watermarked"), (False, "unwatermarked")]:
        generated_ids = generate_kgw_token_ids(
            model=model,
            prompt_token_ids=prompt_token_ids,
            scorer=scorer,
            # Identical generation parameters for both samples: the ONLY
            # difference is whether the KGW logits processor is applied.
            generation_config=generation,
            watermarked=watermarked,
        )
        text = tokenizer.decode(generated_ids)
        score = scorer.score_token_ids(generated_ids)

        (args.out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        metadata = {
            "watermarked": watermarked,
            "scheme": "kgw",
            "variant": "kgw-python-left-v1",
            "configuration_id": kgw_config.configuration_id,
            "configuration_version": kgw_config.version,
            "detector_reference_version": "kgw-python-left-v1",
            "model_identifier": model.identifier,
            "model_revision": getattr(model, "resolved_revision", None),
            "tokenizer_identifier": tokenizer.identifier,
            "tokenizer_revision": getattr(tokenizer, "resolved_revision", None)
            or getattr(tokenizer, "revision", None),
            "vocab_size": tokenizer.vocabulary_size,
            "prompt_id": args.prompt_id,
            "prompt": args.prompt,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_ids,
            "token_count": len(generated_ids),
            "kgw_configuration": {
                "gamma": kgw_config.gamma,
                "delta": kgw_config.delta,
                "hash_key_id": hash_key_id,
                "seeding_scheme": f"left-hash/{kgw_config.f_scheme}",
                "prefix_length": kgw_config.prefix_length,
                "window_scheme": kgw_config.window_scheme,
                "ignore_repeated_ngrams": kgw_config.ignore_repeated_ngrams,
            },
            "generation_parameters": {
                "max_new_tokens": generation.max_new_tokens,
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "top_k": generation.top_k,
                "seed": generation.seed,
                "watermark_enabled": watermarked,
                "watermark_processor": "kgw-python-left-v1" if watermarked else None,
            },
            "detector_score": {
                "z_score": score.z_score,
                "p_value": score.p_value,
                "green_token_count": score.green_token_count,
                "scored_token_count": score.scored_token_count,
                "green_fraction": score.green_fraction,
            },
        }
        (args.out_dir / f"{stem}.metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[{stem}] tokens={len(generated_ids)} "
            f"green_fraction={score.green_fraction:.3f} "
            f"z={score.z_score:.3f} p={score.p_value:.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
