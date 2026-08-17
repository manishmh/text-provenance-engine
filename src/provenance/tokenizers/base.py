"""Common tokenizer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProvenanceTokenizer(ABC):
    """Tokenizer abstraction that preserves exact token IDs."""

    identifier: str
    model_identifier: str | None
    #: Backend family. Detectors use this to classify how "real" a run is.
    #: "simple" -> controlled local vocabulary; "huggingface" -> real model tokenizer.
    backend: str = "simple"
    #: Tokenizer/model revision where applicable (e.g. a Hugging Face git revision).
    revision: str | None = None

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs used by a detector."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text."""

    @property
    @abstractmethod
    def vocabulary_size(self) -> int:
        """Number of token IDs in the tokenizer vocabulary."""
