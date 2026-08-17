"""Configuration structures for model/tokenizer-backed experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    model_identifier: str
    tokenizer_identifier: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        return cls(
            model_identifier=str(data.get("model_identifier", data.get("model", ""))),
            tokenizer_identifier=data.get("tokenizer_identifier"),
            revision=data.get("revision") or data.get("model_revision"),
            tokenizer_revision=data.get("tokenizer_revision"),
            device=str(data.get("device", "cpu")),
            dtype=str(data.get("dtype", "float32")),
            local_files_only=bool(data.get("local_files_only", False)),
        )


@dataclass(frozen=True)
class TokenizerConfig:
    type: str
    tokenizer_identifier: str
    tokenizer_revision: str | None = None
    model_identifier: str | None = None
    vocabulary: tuple[str, ...] = ()
    case_sensitive: bool = False
    local_files_only: bool = False
    trust_remote_code: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerConfig":
        return cls(
            type=str(data.get("type", "simple-vocabulary")),
            tokenizer_identifier=str(data.get("tokenizer_identifier", data.get("identifier", ""))),
            tokenizer_revision=data.get("tokenizer_revision") or data.get("revision"),
            model_identifier=data.get("model_identifier"),
            vocabulary=tuple(str(token) for token in data.get("vocabulary", ())),
            case_sensitive=bool(data.get("case_sensitive", False)),
            local_files_only=bool(data.get("local_files_only", False)),
            trust_remote_code=bool(data.get("trust_remote_code", False)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_factory_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tokenizer_identifier": self.tokenizer_identifier,
            "tokenizer_revision": self.tokenizer_revision,
            "model_identifier": self.model_identifier,
            "vocabulary": list(self.vocabulary),
            "case_sensitive": self.case_sensitive,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    configuration_id: str
    version: str
    model: ModelConfig
    tokenizer: TokenizerConfig
    watermark_configuration: dict[str, Any]
    detector_configuration: dict[str, Any]
    generation_configuration: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            configuration_id=str(data["configuration_id"]),
            version=str(data.get("version", "1")),
            model=ModelConfig.from_dict(data.get("model", {})),
            tokenizer=TokenizerConfig.from_dict(data["tokenizer"]),
            watermark_configuration=dict(data.get("watermark", data.get("watermark_configuration", {}))),
            detector_configuration=dict(data.get("detector", data.get("detector_configuration", {}))),
            generation_configuration=dict(data.get("generation", data.get("generation_configuration", {}))),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
