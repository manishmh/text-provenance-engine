"""Hugging Face tokenizer adapter.

Transformers is imported only when this adapter is instantiated.
"""

from __future__ import annotations

from provenance.tokenizers.base import ProvenanceTokenizer


class HuggingFaceTokenizer(ProvenanceTokenizer):
    """Adapter for `transformers.AutoTokenizer`.

    This is a real, model-compatible tokenizer: token IDs match the identifiers
    that the corresponding causal LM was trained on.
    """

    backend = "huggingface"

    def __init__(
        self,
        *,
        tokenizer_identifier: str,
        model_identifier: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Hugging Face tokenizer support requires the optional 'hf' dependencies"
            ) from exc

        self.identifier = tokenizer_identifier
        self.model_identifier = model_identifier
        self.revision = revision
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_identifier,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        # Resolved git commit hash of the downloaded snapshot, when transformers
        # records it. Falls back to the requested revision label.
        self.resolved_revision = (
            getattr(self._tokenizer, "_commit_hash", None) or revision or "main"
        )

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocabulary_size(self) -> int:
        return len(self._tokenizer.get_vocab())

    @property
    def raw_tokenizer(self):
        return self._tokenizer
