"""Model-backed KGW benchmark runner.

Generates watermarked and unwatermarked samples across several token lengths
with deterministic seeds, scores each with the MarkLLM-compatible KGW scorer
(``kgw-markllm-v1``), and produces :class:`EvaluationRecord` rows plus a report.

The generation core (:func:`generate_records`) takes an already-constructed
model/tokenizer/scorer, so it can be exercised with the offline
``DeterministicToyCausalLM`` fixture in tests without downloading a model. The
Hugging Face loading path lives in :func:`run_benchmark` and imports the optional
``hf`` dependencies lazily.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from provenance.benchmark.records import BENCHMARK_VERSION, EvaluationRecord, write_jsonl
from provenance.benchmark.report import DEFAULT_THRESHOLDS, build_report, render_text_report
from provenance.configuration import ExperimentConfig
from provenance.detectors.reference.kgw_markllm import (
    KGW_MARKLLM_VERSION,
    KGWMarkLLMConfig,
    KGWMarkLLMScorer,
    classify_backend,
)
from provenance.generation import KGWGenerationConfig, generate_kgw_token_ids
from provenance.generation.synthid import generate_synthid_token_ids

# Deterministic prompt pool. Sample i uses DEFAULT_PROMPTS[i % len(pool)], so a
# run with the same seed and sample count is fully reproducible. The pool is
# broad on purpose: at 50-100 samples per length, a handful of near-identical
# prompts would make samples correlated and inflate/deflate detection rates.
# These span history, science, fiction, journalism, technical, and everyday
# registers so continuations diverge.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "The history of lighthouses along the northern coast is a story of",
    "In the early years of the expedition, the crew kept a careful record of",
    "Researchers studying the migration of monarch butterflies have long",
    "The old railway station at the edge of town had been closed for",
    "When the committee finally published its report, the first thing readers",
    "To install the new irrigation controller, first make sure that",
    "She had never expected the letter to arrive so late, and when it finally",
    "The recipe calls for slowly folding the egg whites into the batter until",
    "Economists disagree about whether the recent shift in interest rates will",
    "On the surface of a distant moon, the rover paused at the edge of a",
    "Local officials announced on Tuesday that the bridge would remain closed",
    "The tutorial begins by explaining how memory is allocated when a program",
    "Every autumn the villagers gathered at the market square to trade",
    "According to the museum's new exhibit, medieval mapmakers believed that",
    "After the storm passed, the harbor was littered with debris and the",
    "The professor opened the lecture with a simple question about why",
)

_SEED_LENGTH_STRIDE = 100003  # prime, keeps per-length seed bands disjoint


class _LogitModel(Protocol):
    identifier: str
    vocab_size: int

    def next_token_logits(self, input_ids: list[int]) -> list[float]: ...


def derive_seed(base_seed: int, target_length: int, sample_index: int) -> int:
    """Reproducible per-sample seed. Both conditions at (length, index) share it.

    Using the same seed for the watermarked and unwatermarked sample at a given
    (length, index) means the only difference between the pair is whether the
    KGW logit bias is applied -- mirroring the generation script's design.
    """
    return base_seed + target_length * _SEED_LENGTH_STRIDE + sample_index


def hash_key_id_for(watermark: dict[str, Any], hash_key: int) -> str:
    """Public identifier for the secret key. Never expose the raw key."""
    if watermark.get("hash_key_id"):
        return str(watermark["hash_key_id"])
    return "sha256:" + hashlib.sha256(str(hash_key).encode("utf-8")).hexdigest()[:16]


def _model_revision(model: _LogitModel) -> str | None:
    return getattr(model, "resolved_revision", None) or getattr(model, "revision", None)


def _tokenizer_revision(tokenizer: Any) -> str | None:
    return getattr(tokenizer, "resolved_revision", None) or getattr(tokenizer, "revision", None)


@dataclass(frozen=True)
class BenchmarkSpec:
    lengths: tuple[int, ...]
    samples: int
    seed: int
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    prompts: tuple[str, ...] = DEFAULT_PROMPTS


def generate_records(
    *,
    model: _LogitModel,
    tokenizer: Any,
    scorer: KGWMarkLLMScorer,
    kgw_config: KGWMarkLLMConfig,
    experiment: ExperimentConfig,
    spec: BenchmarkSpec,
    hash_key_id: str,
    experiment_id: str,
    timestamp: str,
) -> list[EvaluationRecord]:
    """Core benchmark loop over lengths x samples x {watermarked, unwatermarked}.

    Model-agnostic: any object with ``vocab_size`` and ``next_token_logits`` works,
    so the offline toy model can drive it in tests.
    """
    implementation_kind, compatibility = classify_backend(tokenizer)
    seeding_scheme = f"{kgw_config.window_scheme}-hash/{kgw_config.f_scheme}"
    records: list[EvaluationRecord] = []

    for target_length in spec.lengths:
        for sample_index in range(spec.samples):
            prompt = spec.prompts[sample_index % len(spec.prompts)]
            prompt_id = f"prompt-{sample_index % len(spec.prompts)}"
            prompt_token_ids = tokenizer.encode(prompt)
            sample_seed = derive_seed(spec.seed, target_length, sample_index)
            generation = KGWGenerationConfig(
                max_new_tokens=target_length,
                temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
                top_p=experiment.generation_configuration.get("top_p"),
                top_k=experiment.generation_configuration.get("top_k"),
                seed=sample_seed,
            )
            for watermarked in (True, False):
                token_ids = generate_kgw_token_ids(
                    model=model,
                    prompt_token_ids=prompt_token_ids,
                    scorer=scorer,
                    generation_config=generation,
                    watermarked=watermarked,
                )
                score = scorer.score_token_ids(token_ids)
                detected = score.z_score >= kgw_config.z_threshold
                records.append(
                    EvaluationRecord(
                        experiment_id=experiment_id,
                        timestamp=timestamp,
                        benchmark_version=BENCHMARK_VERSION,
                        scheme="kgw",
                        variant=KGW_MARKLLM_VERSION,
                        configuration_id=kgw_config.configuration_id,
                        configuration_version=kgw_config.version,
                        implementation_kind=implementation_kind,
                        compatibility=compatibility,
                        model_identifier=getattr(model, "identifier", "unknown"),
                        model_revision=_model_revision(model),
                        tokenizer_identifier=tokenizer.identifier,
                        tokenizer_revision=_tokenizer_revision(tokenizer),
                        vocab_size=scorer.vocab_size,
                        gamma=kgw_config.gamma,
                        delta=kgw_config.delta,
                        prefix_length=kgw_config.prefix_length,
                        window_scheme=kgw_config.window_scheme,
                        seeding_scheme=seeding_scheme,
                        f_scheme=kgw_config.f_scheme,
                        hash_key_id=hash_key_id,
                        temperature=generation.temperature,
                        top_p=generation.top_p,
                        top_k=generation.top_k,
                        random_seed=sample_seed,
                        prompt_id=prompt_id,
                        target_length=target_length,
                        watermarked=watermarked,
                        token_count=score.token_count,
                        scored_token_count=score.scored_token_count,
                        detection_threshold=kgw_config.z_threshold,
                        detected=detected,
                        score=score.z_score,
                        green_token_count=score.green_token_count,
                        green_fraction=score.green_fraction,
                        expected_green_fraction=kgw_config.gamma,
                        z_score=score.z_score,
                        p_value=score.p_value,
                    )
                )
    return records


def reproducibility_metadata(
    *,
    kgw_config: KGWMarkLLMConfig,
    experiment: ExperimentConfig,
    scorer: KGWMarkLLMScorer,
    tokenizer: Any,
    model: _LogitModel,
    spec: BenchmarkSpec,
    hash_key_id: str,
    experiment_id: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "variant": KGW_MARKLLM_VERSION,
        "configuration_id": kgw_config.configuration_id,
        "configuration_version": kgw_config.version,
        "implementation_kind": classify_backend(tokenizer)[0],
        "compatibility": classify_backend(tokenizer)[1],
        "model_identifier": getattr(model, "identifier", "unknown"),
        "model_revision": _model_revision(model),
        "tokenizer_identifier": tokenizer.identifier,
        "tokenizer_revision": _tokenizer_revision(tokenizer),
        "vocab_size": scorer.vocab_size,
        "gamma": kgw_config.gamma,
        "delta": kgw_config.delta,
        "prefix_length": kgw_config.prefix_length,
        "window_scheme": kgw_config.window_scheme,
        "seeding_scheme": f"{kgw_config.window_scheme}-hash/{kgw_config.f_scheme}",
        "f_scheme": kgw_config.f_scheme,
        "hash_key_id": hash_key_id,
        "temperature": float(experiment.generation_configuration.get("temperature", 1.0)),
        "top_p": experiment.generation_configuration.get("top_p"),
        "top_k": experiment.generation_configuration.get("top_k"),
        "seed": spec.seed,
        "lengths": list(spec.lengths),
        "samples_per_length": spec.samples,
        "thresholds": list(spec.thresholds),
        "prompt_count": len(spec.prompts),
        "seed_derivation": "seed + length*100003 + sample_index (shared by both conditions)",
    }


def run_benchmark(
    *,
    config_path: str | Path,
    spec: BenchmarkSpec,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Load the HF model/tokenizer, run the benchmark, and persist outputs.

    Writes ``records.jsonl``, ``report.json`` and ``report.txt`` under ``out_dir``
    and returns the report dict. Requires the optional ``hf`` dependencies.

    The model/tokenizer are served from the process-global loading cache,
    so repeated benchmarks with the same configuration reuse them.
    """
    from provenance.loading import get_model, get_tokenizer

    experiment = ExperimentConfig.from_file(config_path)
    tokenizer = get_tokenizer(experiment.tokenizer.to_factory_dict())
    model = get_model(experiment.model)

    if tokenizer.vocabulary_size != model.vocab_size:
        raise SystemExit(
            f"tokenizer vocabulary size ({tokenizer.vocabulary_size}) must equal "
            f"model vocab size ({model.vocab_size}); the detector scores exact "
            "model token IDs."
        )

    kgw_config = KGWMarkLLMConfig.from_dict(
        {
            "configuration_id": experiment.configuration_id,
            "version": experiment.version,
            **experiment.watermark_configuration,
            **experiment.detector_configuration,
        }
    )
    scorer = KGWMarkLLMScorer(kgw_config, vocab_size=tokenizer.vocabulary_size)
    hash_key_id = hash_key_id_for(experiment.watermark_configuration, kgw_config.hash_key)

    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = f"{kgw_config.configuration_id}:{KGW_MARKLLM_VERSION}:{spec.seed}"

    records = generate_records(
        model=model,
        tokenizer=tokenizer,
        scorer=scorer,
        kgw_config=kgw_config,
        experiment=experiment,
        spec=spec,
        hash_key_id=hash_key_id,
        experiment_id=experiment_id,
        timestamp=timestamp,
    )

    repro = reproducibility_metadata(
        kgw_config=kgw_config,
        experiment=experiment,
        scorer=scorer,
        tokenizer=tokenizer,
        model=model,
        spec=spec,
        hash_key_id=hash_key_id,
        experiment_id=experiment_id,
        timestamp=timestamp,
    )
    report = build_report(records, thresholds=spec.thresholds, reproducibility=repro)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "records.jsonl")
    import json

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "report.txt").write_text(render_text_report(report), encoding="utf-8")
    return report


def generate_synthid_records(
    *,
    model: _LogitModel,
    tokenizer: Any,
    detector: Any,
    processor: Any,
    synthid_config: Any,
    experiment: ExperimentConfig,
    spec: BenchmarkSpec,
    hash_key_id: str,
    experiment_id: str,
    timestamp: str,
) -> list[EvaluationRecord]:
    """Core SynthID benchmark loop over lengths x samples x {watermarked, unwatermarked}."""
    records: list[EvaluationRecord] = []

    for target_length in spec.lengths:
        for sample_index in range(spec.samples):
            prompt = spec.prompts[sample_index % len(spec.prompts)]
            prompt_id = f"prompt-{sample_index % len(spec.prompts)}"
            prompt_token_ids = tokenizer.encode(prompt)
            sample_seed = derive_seed(spec.seed, target_length, sample_index)
            generation = KGWGenerationConfig(
                max_new_tokens=target_length,
                temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
                top_p=experiment.generation_configuration.get("top_p"),
                top_k=experiment.generation_configuration.get("top_k"),
                seed=sample_seed,
            )
            for watermarked in (True, False):
                token_ids = generate_synthid_token_ids(
                    model=model,
                    prompt_token_ids=prompt_token_ids,
                    processor=processor,
                    generation_config=generation,
                    watermarked=watermarked,
                )
                result = detector.score_token_ids(token_ids)
                score_val = result.score if result.score is not None else 0.0
                detected = score_val >= synthid_config.threshold
                records.append(
                    EvaluationRecord(
                        experiment_id=experiment_id,
                        timestamp=timestamp,
                        benchmark_version=BENCHMARK_VERSION,
                        scheme="synthid",
                        variant="synthid-reference-v1",
                        configuration_id=synthid_config.configuration_id,
                        configuration_version=synthid_config.version,
                        implementation_kind="reference",
                        compatibility="synthid-reference",
                        model_identifier=getattr(model, "identifier", "unknown"),
                        model_revision=_model_revision(model),
                        tokenizer_identifier=tokenizer.identifier,
                        tokenizer_revision=_tokenizer_revision(tokenizer),
                        vocab_size=getattr(model, "vocab_size", 0),
                        hash_key_id=hash_key_id,
                        ngram_len=synthid_config.ngram_len,
                        watermarking_depth=len(synthid_config.keys),
                        temperature=generation.temperature,
                        top_p=generation.top_p,
                        top_k=generation.top_k,
                        random_seed=sample_seed,
                        prompt_id=prompt_id,
                        target_length=target_length,
                        watermarked=watermarked,
                        token_count=len(token_ids),
                        scored_token_count=result.evidence.get("usable_token_count", len(token_ids)),
                        detection_threshold=synthid_config.threshold,
                        detected=detected,
                        score=score_val,
                    )
                )
    return records


def run_synthid_benchmark(
    *,
    config_path: str | Path,
    spec: BenchmarkSpec,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Load HF model, run SynthID benchmark, persist outputs.

    The model/tokenizer are served from the process-global loading cache,
    so repeated benchmarks with the same configuration reuse them.
    """
    from provenance.detectors.reference.synthid import (
        SynthIDReferenceConfig,
        SynthIDReferenceDetector,
    )
    from provenance.generation.synthid import (
        SynthIDLogitsProcessor,
        generate_synthid_token_ids,
    )
    from provenance.loading import get_model, get_tokenizer

    experiment = ExperimentConfig.from_file(config_path)
    tokenizer = get_tokenizer(experiment.tokenizer.to_factory_dict())
    model = get_model(experiment.model)

    if tokenizer.vocabulary_size != model.vocab_size:
        raise SystemExit(
            f"tokenizer vocabulary size ({tokenizer.vocabulary_size}) must equal "
            f"model vocab size ({model.vocab_size})"
        )

    synthid_config = SynthIDReferenceConfig.from_dict({
        "configuration_id": experiment.configuration_id,
        "version": experiment.version,
        **experiment.watermark_configuration,
        **experiment.detector_configuration,
        "tokenizer": "huggingface",
        "model_identifier": experiment.model.model_identifier,
        "model_revision": experiment.model.revision,
        "tokenizer_identifier": experiment.tokenizer.tokenizer_identifier,
        "tokenizer_revision": experiment.tokenizer.tokenizer_revision,
        **{k: v for k, v in experiment.generation_configuration.items()
           if k in ("max_new_tokens", "temperature", "top_p", "top_k", "seed")},
    })
    synthid_config.validate()

    detector = SynthIDReferenceDetector(synthid_config, tokenizer=tokenizer)
    processor = SynthIDLogitsProcessor(synthid_config)

    timestamp = datetime.now(timezone.utc).isoformat()
    experiment_id = f"{synthid_config.configuration_id}:synthid-reference-v1:{spec.seed}"

    records = generate_synthid_records(
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        processor=processor,
        synthid_config=synthid_config,
        experiment=experiment,
        spec=spec,
        hash_key_id=synthid_config.hash_key_id,
        experiment_id=experiment_id,
        timestamp=timestamp,
    )

    repro = {
        "benchmark_version": BENCHMARK_VERSION,
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "variant": "synthid-reference-v1",
        "scheme": "synthid",
        "configuration_id": synthid_config.configuration_id,
        "configuration_version": synthid_config.version,
        "implementation_kind": "reference",
        "compatibility": "synthid-reference",
        "model_identifier": getattr(model, "identifier", "unknown"),
        "model_revision": _model_revision(model),
        "tokenizer_identifier": tokenizer.identifier,
        "tokenizer_revision": _tokenizer_revision(tokenizer),
        "vocab_size": model.vocab_size,
        "hash_key_id": synthid_config.hash_key_id,
        "ngram_len": synthid_config.ngram_len,
        "watermarking_depth": len(synthid_config.keys),
        "temperature": float(experiment.generation_configuration.get("temperature", 1.0)),
        "top_p": experiment.generation_configuration.get("top_p"),
        "top_k": experiment.generation_configuration.get("top_k"),
        "seed": spec.seed,
        "lengths": list(spec.lengths),
        "samples_per_length": spec.samples,
        "thresholds": list(spec.thresholds),
        "prompt_count": len(spec.prompts),
        "seed_derivation": "seed + length*100003 + sample_index (shared by both conditions)",
    }

    from provenance.benchmark.report import build_report, render_text_report
    report = build_report(records, thresholds=spec.thresholds, reproducibility=repro)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "records.jsonl")
    import json
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "report.txt").write_text(render_text_report(report), encoding="utf-8")
    return report
