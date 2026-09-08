"""Robustness transformations for evaluating detector robustness.

Each transformation:
- accepts text and returns transformed text
- has a stable name
- is deterministic
- exposes metadata
- never modifies the original input in-place
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class TransformResult:
    """Result of applying a transformation."""

    text: str
    transform_name: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "transform_name": self.transform_name,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Transform:
    """A named, deterministic text transformation."""

    name: str
    description: str
    func: Callable[[str], TransformResult]

    def apply(self, text: str) -> TransformResult:
        return self.func(text)

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}


# ── Built-in transformations ───────────────────────────────────────────


def _identity(text: str) -> TransformResult:
    return TransformResult(text=text, transform_name="identity")


def _whitespace_normalization(text: str) -> TransformResult:
    """Normalize whitespace: collapse runs of spaces/tabs, strip lines."""
    result = re.sub(r"[ \t]+", " ", text)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()
    return TransformResult(
        text=result,
        transform_name="whitespace_normalization",
        metadata={"original_length": len(text), "transformed_length": len(result)},
    )


def _unicode_normalization_nfc(text: str) -> TransformResult:
    """Apply Unicode NFC normalization."""
    result = unicodedata.normalize("NFC", text)
    return TransformResult(
        text=result,
        transform_name="unicode_normalization_nfc",
        metadata={"form": "NFC", "changed": result != text},
    )


def _unicode_normalization_nfd(text: str) -> TransformResult:
    """Apply Unicode NFD normalization."""
    result = unicodedata.normalize("NFD", text)
    return TransformResult(
        text=result,
        transform_name="unicode_normalization_nfd",
        metadata={"form": "NFD", "changed": result != text},
    )


def _lowercase(text: str) -> TransformResult:
    """Convert to lowercase."""
    result = text.lower()
    return TransformResult(
        text=result,
        transform_name="lowercase",
        metadata={"changed": result != text},
    )


def _uppercase(text: str) -> TransformResult:
    """Convert to uppercase."""
    result = text.upper()
    return TransformResult(
        text=result,
        transform_name="uppercase",
        metadata={"changed": result != text},
    )


def _punctuation_normalization(text: str) -> TransformResult:
    """Normalize common punctuation variants to ASCII equivalents."""
    replacements = {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # ellipsis
        "\u00A0": " ",  # non-breaking space
        "\u200B": "",  # zero-width space
        "\u200C": "",  # zero-width non-joiner
        "\u200D": "",  # zero-width joiner
        "\uFEFF": "",  # BOM / zero-width no-break space
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return TransformResult(
        text=result,
        transform_name="punctuation_normalization",
        metadata={"changed": result != text},
    )


def _strip_blank_lines(text: str) -> TransformResult:
    """Remove blank lines while preserving single newlines."""
    lines = text.split("\n")
    result_lines = [line for line in lines if line.strip()]
    result = "\n".join(result_lines)
    return TransformResult(
        text=result,
        transform_name="strip_blank_lines",
        metadata={"lines_removed": len(lines) - len(result_lines)},
    )


def _add_leading_trailing_whitespace(text: str) -> TransformResult:
    """Add leading and trailing whitespace."""
    result = "  " + text + "  "
    return TransformResult(
        text=result,
        transform_name="add_leading_trailing_whitespace",
        metadata={"added_chars": 4},
    )


def _double_spaces(text: str) -> TransformResult:
    """Replace single spaces with double spaces."""
    result = text.replace(" ", "  ")
    return TransformResult(
        text=result,
        transform_name="double_spaces",
        metadata={"changed": result != text},
    )


# ── Registry ───────────────────────────────────────────────────────────

ALL_TRANSFORMS: list[Transform] = [
    Transform(name="identity", description="No transformation (baseline)", func=_identity),
    Transform(name="whitespace_normalization", description="Collapse whitespace runs", func=_whitespace_normalization),
    Transform(name="unicode_normalization_nfc", description="Unicode NFC normalization", func=_unicode_normalization_nfc),
    Transform(name="unicode_normalization_nfd", description="Unicode NFD normalization", func=_unicode_normalization_nfd),
    Transform(name="lowercase", description="Convert to lowercase", func=_lowercase),
    Transform(name="uppercase", description="Convert to uppercase", func=_uppercase),
    Transform(name="punctuation_normalization", description="Normalize Unicode punctuation to ASCII", func=_punctuation_normalization),
    Transform(name="strip_blank_lines", description="Remove blank lines", func=_strip_blank_lines),
    Transform(name="add_leading_trailing_whitespace", description="Add leading/trailing spaces", func=_add_leading_trailing_whitespace),
    Transform(name="double_spaces", description="Double all spaces", func=_double_spaces),
]

TRANSFORM_MAP: dict[str, Transform] = {t.name: t for t in ALL_TRANSFORMS}


def get_transform(name: str) -> Transform:
    """Get a transform by name."""
    if name not in TRANSFORM_MAP:
        raise ValueError(f"Unknown transform: {name!r}. Available: {list(TRANSFORM_MAP.keys())}")
    return TRANSFORM_MAP[name]


def get_all_transforms() -> list[Transform]:
    """Get all registered transforms."""
    return list(ALL_TRANSFORMS)


def apply_transform(text: str, transform_name: str) -> TransformResult:
    """Apply a named transform to text."""
    return get_transform(transform_name).apply(text)
