"""Text validation and lightweight tokenization."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a str")


def simple_tokenize(text: str, *, case_sensitive: bool = True) -> list[str]:
    """Tokenize text with a deterministic regex tokenizer.

    This is not a model tokenizer. KGW and SynthID detectors using this tokenizer
    are only meaningful for samples generated with the same explicit config.
    """

    if not case_sensitive:
        text = text.lower()
    return [
        token
        for token in TOKEN_RE.findall(text)
        if any(not unicodedata.category(char).startswith("C") for char in token)
    ]


def stable_hash_int(parts: Iterable[object], *, digest_size: int = 8) -> int:
    data = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=digest_size).digest(), "big")


def stable_hash_float(parts: Iterable[object]) -> float:
    return stable_hash_int(parts) / float(1 << 64)


def basic_text_statistics(text: str) -> dict[str, object]:
    tokens = simple_tokenize(text)
    return {
        "character_count": len(text),
        "byte_count_utf8": len(text.encode("utf-8")),
        "line_count": text.count("\n") + 1 if text else 0,
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "normalization": {
            "is_nfc": unicodedata.is_normalized("NFC", text),
            "is_nfkc": unicodedata.is_normalized("NFKC", text),
        },
    }
