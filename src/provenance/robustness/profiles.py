"""Predefined transformation profiles for robustness evaluation.

Profiles group transformations by category or purpose. They expand into
a deterministic list of transforms. A profile must only SELECT
transformations — it must not perform adaptive optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Profile:
    """A named, deterministic list of transformation names."""
    name: str
    description: str
    transform_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "transform_names": list(self.transform_names),
        }


# ── Predefined profiles ──────────────────────────────────────────────

PROFILES: dict[str, Profile] = {
    "formatting": Profile(
        name="formatting",
        description="Paragraph reflow, line wrapping, blank line normalization",
        transform_names=(
            "paragraph_reflow",
            "line_wrap_normalize",
            "blank_line_normalize",
        ),
    ),
    "unicode": Profile(
        name="unicode",
        description="Unicode normalization forms and punctuation normalization",
        transform_names=(
            "unicode_nfc",
            "unicode_nfd",
            "unicode_nfkd",
            "unicode_punctuation_normalize",
        ),
    ),
    "whitespace": Profile(
        name="whitespace",
        description="Whitespace collapse, expansion, and normalization",
        transform_names=(
            "whitespace_collapse",
            "whitespace_expand",
            "tab_space_normalize",
            "double_spaces",
        ),
    ),
    "casing": Profile(
        name="casing",
        description="Case transformations (lowercase, uppercase, title case)",
        transform_names=(
            "lowercase",
            "uppercase",
            "title_case",
        ),
    ),
    "punctuation": Profile(
        name="punctuation",
        description="Punctuation normalization and repeated punctuation handling",
        transform_names=(
            "punctuation_normalize",
            "repeated_punctuation_normalize",
        ),
    ),
    "lexical": Profile(
        name="lexical",
        description="Conservative synonym substitution and contraction expansion",
        transform_names=(
            "conservative_synonym_substitution",
            "contraction_expansion",
        ),
    ),
    "tokenization-sensitive": Profile(
        name="tokenization-sensitive",
        description="Transformations that affect tokenization boundaries",
        transform_names=(
            "insert_formatting_boundaries",
            "remove_formatting_boundaries",
        ),
    ),
    "all_safe": Profile(
        name="all_safe",
        description="All safe, deterministic transformations",
        transform_names=(
            # Formatting
            "paragraph_reflow",
            "line_wrap_normalize",
            "blank_line_normalize",
            # Whitespace
            "whitespace_collapse",
            "whitespace_expand",
            "tab_space_normalize",
            "double_spaces",
            # Unicode
            "unicode_nfc",
            "unicode_nfd",
            "unicode_nfkd",
            "unicode_punctuation_normalize",
            # Casing
            "lowercase",
            "uppercase",
            "title_case",
            # Punctuation
            "punctuation_normalize",
            "repeated_punctuation_normalize",
            # Lexical
            "conservative_synonym_substitution",
            "contraction_expansion",
            # Tokenization-sensitive
            "insert_formatting_boundaries",
            "remove_formatting_boundaries",
        ),
    ),
}


def get_profile(name: str) -> Profile:
    """Get a profile by name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name!r}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def get_available_profiles() -> dict[str, Profile]:
    """Get all available profiles."""
    return dict(PROFILES)


def list_profile_names() -> list[str]:
    """Get all profile names."""
    return list(PROFILES.keys())
