"""Offline model-backed benchmark test using the deterministic toy LM.

Exercises the full generate -> score -> record -> aggregate -> serialize path
without downloading a model. Requires torch because the MarkLLM-compatible scorer
uses ``torch.randperm``; model-backed HF benchmarks live outside this test.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from provenance.benchmark.records import EvaluationRecord, read_jsonl, write_jsonl
from provenance.benchmark.report import build_report
from provenance.benchmark.runner import BenchmarkSpec, generate_records, hash_key_id_for
from provenance.configuration import ExperimentConfig
from provenance.detectors.reference.kgw_markllm import KGWMarkLLMConfig, KGWMarkLLMScorer
from provenance.models.local import DeterministicToyCausalLM
from provenance.tokenizers import tokenizer_from_config

CONFIG_PATH = "configs/kgw.reference.example.json"


def _components():
    experiment = ExperimentConfig.from_file(CONFIG_PATH)
    tokenizer = tokenizer_from_config(experiment.tokenizer.to_factory_dict())
    kgw_config = KGWMarkLLMConfig.from_dict(
        {
            "configuration_id": experiment.configuration_id,
            "version": experiment.version,
            **experiment.watermark_configuration,
            **experiment.detector_configuration,
        }
    )
    scorer = KGWMarkLLMScorer(kgw_config, vocab_size=tokenizer.vocabulary_size)
    model = DeterministicToyCausalLM(
        vocab_size=tokenizer.vocabulary_size,
        identifier=experiment.model.model_identifier,
    )
    return experiment, tokenizer, kgw_config, scorer, model


def _run(spec):
    experiment, tokenizer, kgw_config, scorer, model = _components()
    hash_key_id = hash_key_id_for(experiment.watermark_configuration, kgw_config.hash_key)
    return generate_records(
        model=model,
        tokenizer=tokenizer,
        scorer=scorer,
        kgw_config=kgw_config,
        experiment=experiment,
        spec=spec,
        hash_key_id=hash_key_id,
        experiment_id="toy-benchmark",
        timestamp="2026-08-19T00:00:00+00:00",
    )


def test_toy_benchmark_shapes_and_separation():
    spec = BenchmarkSpec(lengths=(40, 80), samples=2, seed=7, thresholds=(2.0, 3.0, 4.0))
    records = _run(spec)

    # lengths x samples x {watermarked, unwatermarked}
    assert len(records) == 2 * 2 * 2
    assert sorted({r.target_length for r in records}) == [40, 80]
    assert all(r.variant == "kgw-markllm-v1" for r in records)
    # Toy tokenizer -> controlled-local-only, not a real model-backed reference run.
    assert all(r.implementation_kind == "reference-adapted" for r in records)
    assert all(r.compatibility == "controlled-local-only" for r in records)

    wm = [r for r in records if r.watermarked]
    un = [r for r in records if not r.watermarked]
    mean_wm_z = sum(r.z_score for r in wm) / len(wm)
    mean_un_z = sum(r.z_score for r in un) / len(un)
    assert mean_wm_z > mean_un_z
    assert mean_wm_z > 3.0  # delta=4.0 produces a strong watermark on the toy LM
    # Watermarked samples carry a higher green fraction than any unwatermarked one.
    assert min(r.green_fraction for r in wm) > max(r.green_fraction for r in un)


def test_toy_benchmark_secret_key_never_stored():
    records = _run(BenchmarkSpec(lengths=(40,), samples=1, seed=1))
    for record in records:
        assert record.hash_key_id  # a public identifier is present
        assert not hasattr(record, "hash_key")
    # 15485863 is the raw secret key from the config; it must not appear anywhere.
    assert "15485863" not in json.dumps([r.to_dict() for r in records])


def test_toy_benchmark_determinism():
    spec = BenchmarkSpec(lengths=(40, 80), samples=2, seed=7)
    first = _run(spec)
    second = _run(spec)
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_toy_benchmark_serialization_and_report(tmp_path):
    spec = BenchmarkSpec(lengths=(40, 80), samples=2, seed=7, thresholds=(2.0, 3.0, 4.0))
    records = _run(spec)

    path = write_jsonl(records, tmp_path / "records.jsonl")
    assert read_jsonl(path) == records

    report = build_report(records, thresholds=spec.thresholds)
    assert report["sample_counts"]["total"] == len(records)
    assert [b["target_length"] for b in report["by_length"]] == [40, 80]
    # Every threshold point carries matched watermarked/unwatermarked counts.
    for point in report["overall"]["thresholds"]:
        assert point["n_watermarked"] == 4
        assert point["n_unwatermarked"] == 4
