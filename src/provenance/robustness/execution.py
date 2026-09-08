"""Shared watermark benchmark execution pipeline (Phase 6C).

Single implementation of the generate → evaluate → persist pipeline used
by both the CLI (``benchmark robustness`` / ``benchmark plan``) and the
API benchmark-run worker. Moving it here guarantees the dashboard-triggered
runs execute byte-identical logic to the CLI.

Heavy imports stay inside functions so importing this module never requires
torch/transformers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def wrap_advanced_transform(adv_transform: Any, seed: int = 0) -> Any:
    """Wrap an AdvancedTransform as a Transform-compatible object."""
    from provenance.robustness.transforms import Transform, TransformResult

    def _apply(text: str) -> TransformResult:
        result = adv_transform.apply(text, seed)
        return TransformResult(
            text=result,
            transform_name=adv_transform.name,
            metadata={
                "category": adv_transform.category.value,
                "severity": adv_transform.severity,
                "changed": result != text,
            },
        )

    return Transform(
        name=adv_transform.name,
        description=adv_transform.description,
        func=_apply,
    )


def write_watermark_benchmark_results(
    records: Any,
    *,
    detector_name: str,
    config_identifier: str,
    seed: int,
    out_dir: str | Path,
) -> list:
    """Persist Phase 5C benchmark results grouped by (transform, length).

    Returns the list of ``RobustnessBenchmarkResult`` objects written to
    ``<out_dir>/benchmark_results.jsonl``.
    """
    from collections import defaultdict

    from provenance.robustness.benchmark import build_benchmark_result, write_benchmark_results

    groups: dict[tuple[str, int | None], list] = defaultdict(list)
    for r in records:
        groups[(r.transform_name, r.metadata.get("length"))].append(r)

    benchmark_results = []
    for (tname, length), trecords in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        benchmark_results.append(build_benchmark_result(
            detector_name=detector_name,
            config_identifier=config_identifier,
            transform_name=tname,
            text_length=length,
            records=[r.to_dict() for r in trecords],
            seed=seed,
        ))
    write_benchmark_results(benchmark_results, Path(out_dir) / "benchmark_results.jsonl")
    return benchmark_results


def resolve_experiment_transforms(spec: Any) -> list:
    """Resolve the transform list for a ``BenchmarkSpec``.

    Precedence: explicit ``transforms`` names, then ``profile``, then all
    Phase 5A baseline transforms. Profile and explicit names may refer to
    either the Phase 5A baseline namespace or the Phase 5D advanced
    namespace; advanced names take precedence on collision (the overlapping
    transforms are behaviorally equivalent).
    """
    from provenance.robustness.transforms import get_all_transforms, get_transform

    if spec.transforms:
        names = list(spec.transforms)
    elif spec.profile:
        from provenance.robustness.profiles import get_profile
        names = list(get_profile(spec.profile).transform_names)
    else:
        return get_all_transforms()

    from provenance.robustness.advanced_transforms import ADVANCED_TRANSFORM_MAP
    transforms = []
    for name in names:
        if name in ADVANCED_TRANSFORM_MAP:
            transforms.append(wrap_advanced_transform(ADVANCED_TRANSFORM_MAP[name], spec.seed))
        else:
            transforms.append(get_transform(name))
    return transforms


def generate_watermark_samples(
    *,
    detector_name: str,
    config_path: str,
    lengths: tuple[int, ...],
    samples_per_length: int,
    seed: int,
    quiet: bool = False,
) -> list | None:
    """Generate watermarked samples using the existing generation pipeline.

    The model/tokenizer load once per call (served from the process-global
    loading cache). Returns None if the config format doesn't support
    generation (e.g. simulation configs with simple tokenizer).
    """
    from provenance.benchmark.runner import derive_seed, DEFAULT_PROMPTS
    from provenance.robustness.experiments import WatermarkSample
    from provenance.configuration import ExperimentConfig as ExpConfig
    from provenance.loading import get_model, get_tokenizer

    try:
        experiment = ExpConfig.from_file(config_path)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if not quiet:
            print(
                f"Error: Config file {config_path} does not match the expected format for watermark generation.\n"
                f"Watermark generation requires an HF experiment config (with 'model' and 'tokenizer' sections).\n"
                f"Simulation configs (simple tokenizer) cannot generate watermarked samples.\n"
                f"Use --text <file> for text-file mode, or provide an HF experiment config.\n"
                f"Details: {exc}",
                flush=True,
            )
        return None
    # Load once per call (served from the process-global cache, so repeated
    # specs sharing a config reuse the model instead of reloading per sample).
    tokenizer = get_tokenizer(experiment.tokenizer.to_factory_dict())
    model = get_model(experiment.model)

    samples = []
    for length in lengths:
        for i in range(samples_per_length):
            prompt = DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]
            sample_seed = derive_seed(seed, length, i)

            if detector_name in ("kgw", "kgw-reference"):
                text = generate_kgw_sample(
                    config_path=config_path,
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    length=length,
                    seed=sample_seed,
                    watermarked=True,
                )
            elif detector_name in ("synthid", "synthid-reference"):
                text = generate_synthid_sample(
                    config_path=config_path,
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    length=length,
                    seed=sample_seed,
                    watermarked=True,
                )
            else:
                raise ValueError(f"Cannot generate samples for detector: {detector_name}")

            samples.append(WatermarkSample(
                text=text,
                token_count=len(tokenizer.encode(text)),
                sample_seed=sample_seed,
                prompt_id=f"prompt-{i % len(DEFAULT_PROMPTS)}",
                length=length,
                watermarked=True,
            ))

    return samples


def generate_kgw_sample(
    *,
    config_path: str,
    tokenizer: Any,
    prompt: str,
    length: int,
    seed: int,
    watermarked: bool,
    model: Any = None,
) -> str:
    """Generate a KGW watermarked text sample.

    When ``model`` is provided it is reused as-is (the caller is expected
    to serve it from the loading cache); otherwise it is loaded fresh.
    """
    from provenance.detectors.reference.kgw_markllm import (
        KGWMarkLLMConfig,
        KGWMarkLLMScorer,
        classify_backend,
    )
    from provenance.generation import KGWGenerationConfig, generate_kgw_token_ids
    from provenance.configuration import ExperimentConfig as ExpConfig

    experiment = ExpConfig.from_file(config_path)
    if model is None:
        from provenance.loading import get_model
        model = get_model(experiment.model)

    kgw_config = KGWMarkLLMConfig.from_dict({
        "configuration_id": experiment.configuration_id,
        "version": experiment.version,
        **experiment.watermark_configuration,
        **experiment.detector_configuration,
    })
    scorer = KGWMarkLLMScorer(kgw_config, vocab_size=tokenizer.vocabulary_size)
    prompt_token_ids = tokenizer.encode(prompt)
    gen_config = KGWGenerationConfig(
        max_new_tokens=length,
        temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
        top_p=experiment.generation_configuration.get("top_p"),
        top_k=experiment.generation_configuration.get("top_k"),
        seed=seed,
    )
    token_ids = generate_kgw_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        scorer=scorer,
        generation_config=gen_config,
        watermarked=watermarked,
    )
    return tokenizer.decode(token_ids)


def generate_synthid_sample(
    *,
    config_path: str,
    tokenizer: Any,
    prompt: str,
    length: int,
    seed: int,
    watermarked: bool,
    model: Any = None,
) -> str:
    """Generate a SynthID watermarked text sample.

    When ``model`` is provided it is reused as-is (the caller is expected
    to serve it from the loading cache); otherwise it is loaded fresh.
    """
    from provenance.detectors.reference.synthid import (
        SynthIDReferenceConfig,
        SynthIDReferenceDetector,
    )
    from provenance.generation import SynthIDGenerationConfig, SynthIDLogitsProcessor, generate_synthid_token_ids
    from provenance.configuration import ExperimentConfig as ExpConfig

    experiment = ExpConfig.from_file(config_path)
    if model is None:
        from provenance.loading import get_model
        model = get_model(experiment.model)

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

    processor = SynthIDLogitsProcessor(synthid_config)
    prompt_token_ids = tokenizer.encode(prompt)
    gen_config = SynthIDGenerationConfig(
        max_new_tokens=length,
        temperature=float(experiment.generation_configuration.get("temperature", 1.0)),
        top_p=experiment.generation_configuration.get("top_p"),
        top_k=experiment.generation_configuration.get("top_k"),
        seed=seed,
    )
    token_ids = generate_synthid_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        processor=processor,
        generation_config=gen_config,
        watermarked=watermarked,
    )
    return tokenizer.decode(token_ids)


def _is_hf_experiment_config(config_path: str) -> bool:
    """Best-effort check whether a config looks like an HF experiment config."""
    try:
        import json as _json
        payload = _json.loads(Path(config_path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and "model" in payload and "tokenizer" in payload


def run_plan_experiment(spec: Any, exp_out_dir: str | Path) -> list[dict]:
    """Execute one ``BenchmarkSpec``: generate → evaluate → persist.

    Runs the real watermark pipeline and returns the result dicts. Any
    exception propagates to the caller (the plan runner records it as a
    FAILED experiment; the API worker records a failed run).
    """
    from provenance.detectors.registry import get_registry
    from provenance.robustness.experiments import evaluate_robustness_experiment

    detector_name = spec.detector
    if detector_name == "unicode":
        raise ValueError(
            "unicode detector does not support watermark sample generation; "
            "evaluate it with `benchmark robustness --text <file> --detector unicode` instead"
        )
    if detector_name not in ("kgw", "kgw-reference", "synthid", "synthid-reference"):
        raise ValueError(f"Cannot generate samples for detector: {detector_name}")
    if detector_name in ("kgw", "synthid") and _is_hf_experiment_config(spec.config):
        # Fail fast with guidance: the simulation detector loader would
        # otherwise reject the HF config with a confusing message during
        # evaluation, after samples were already generated.
        raise ValueError(
            f"Detector {detector_name!r} is a simulation detector and cannot evaluate "
            f"HF experiment configs; use '{detector_name}-reference' instead"
        )

    transforms = resolve_experiment_transforms(spec)
    detector = get_registry().create(detector_name, config_path=spec.config)

    wm_samples = generate_watermark_samples(
        detector_name=detector_name,
        config_path=spec.config,
        lengths=spec.lengths,
        samples_per_length=spec.samples,
        seed=spec.seed,
    )
    if wm_samples is None:
        raise ValueError(
            f"Config {spec.config} does not support watermark sample generation "
            "(requires an HF experiment config with model/tokenizer sections)"
        )

    records = evaluate_robustness_experiment(detector, wm_samples, transforms)
    results = write_watermark_benchmark_results(
        records,
        detector_name=detector_name,
        config_identifier=spec.config,
        seed=spec.seed,
        out_dir=exp_out_dir,
    )
    return [r.to_dict() for r in results]
