"""Simple tokenizer for controlled local simulations and offline fixtures."""

from __future__ import annotations

from provenance.preprocessing.text import simple_tokenize, stable_hash_int
from provenance.tokenizers.base import ProvenanceTokenizer


class SimpleVocabularyTokenizer(ProvenanceTokenizer):
    """Regex tokenizer with an explicit local vocabulary.

    This tokenizer is not model-compatible. It exists for controlled simulations
    and offline fixtures where the exact token ID mapping is recorded.
    """

    backend = "simple"
    revision = None

    def __init__(
        self,
        *,
        vocabulary: tuple[str, ...] | list[str],
        tokenizer_identifier: str = "simple-vocabulary",
        model_identifier: str | None = None,
        case_sensitive: bool = False,
        namespace: str = "simple-token",
    ):
        if len(vocabulary) < 2:
            raise ValueError("simple tokenizer vocabulary must contain at least two tokens")
        self.vocabulary = tuple(str(token) for token in vocabulary)
        self.identifier = tokenizer_identifier
        self.model_identifier = model_identifier
        self.case_sensitive = case_sensitive
        self.namespace = namespace
        self._vocab_index = {
            self._normalize_token(token): index for index, token in enumerate(self.vocabulary)
        }

    def encode(self, text: str) -> list[int]:
        return [self.token_id(token) for token in simple_tokenize(text, case_sensitive=self.case_sensitive)]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.vocabulary[token_id % self.vocabulary_size] for token_id in token_ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def token_id(self, token: str) -> int:
        normalized = self._normalize_token(token)
        if normalized in self._vocab_index:
            return self._vocab_index[normalized]
        return stable_hash_int([self.namespace, self.identifier, normalized]) % self.vocabulary_size

    def _normalize_token(self, token: str) -> str:
        return token if self.case_sensitive else token.lower()
