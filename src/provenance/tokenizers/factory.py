"""Tokenizer factory helpers."""

from __future__ import annotations

from typing import Any

from provenance.tokenizers.base import ProvenanceTokenizer
from provenance.tokenizers.huggingface import HuggingFaceTokenizer
from provenance.tokenizers.simple import SimpleVocabularyTokenizer


def tokenizer_from_config(config: dict[str, Any]) -> ProvenanceTokenizer:
    tokenizer_type = str(config.get("type", "simple-vocabulary"))
    if tokenizer_type in {"simple", "simple-vocabulary", "regex"}:
        return SimpleVocabularyTokenizer(
            vocabulary=tuple(config["vocabulary"]),
            tokenizer_identifier=str(config.get("tokenizer_identifier", "simple-vocabulary")),
            model_identifier=config.get("model_identifier"),
            case_sensitive=bool(config.get("case_sensitive", False)),
            namespace=str(config.get("namespace", "simple-token")),
        )
    if tokenizer_type in {"hf", "huggingface", "transformers"}:
        return HuggingFaceTokenizer(
            tokenizer_identifier=str(config["tokenizer_identifier"]),
            model_identifier=config.get("model_identifier"),
            revision=config.get("tokenizer_revision") or config.get("revision"),
            local_files_only=bool(config.get("local_files_only", False)),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
        )
    raise ValueError(f"unsupported tokenizer type: {tokenizer_type}")
