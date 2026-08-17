"""Unicode artifact inspection.

This module adapts the text-only inspection categories from
watermarks-remover's `service/scripts/text_unicode.py` into a detection-only
API. It reports suspicious characters and does not alter input text.
"""

from __future__ import annotations

import unicodedata

from provenance.schemas import UnicodeFinding

ZERO_WIDTH_CODEPOINTS = {
    0x034F: "Combining grapheme joiner can carry invisible formatting.",
    0x061C: "Arabic letter mark affects bidirectional layout.",
    0x180E: "Deprecated Mongolian vowel separator is invisible formatting.",
    0x200B: "Zero-width space can carry invisible text-level marks.",
    0x200C: "Zero-width non-joiner can carry invisible text-level marks.",
    0x200D: "Zero-width joiner can carry invisible text-level marks.",
    0x2060: "Word joiner is invisible formatting.",
    0xFEFF: "Zero-width no-break space / byte order mark inside text.",
}

BIDI_CONTROLS = {
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
}

UNUSUAL_WHITESPACE = {
    0x00A0,
    0x1680,
    0x2000,
    0x2001,
    0x2002,
    0x2003,
    0x2004,
    0x2005,
    0x2006,
    0x2007,
    0x2008,
    0x2009,
    0x200A,
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
}

SOFT_HYPHEN = 0x00AD
TAG_CHAR_RANGES = ((0xE0000, 0xE007F),)
VARIATION_SELECTOR_RANGES = ((0xFE00, 0xFE0F), (0xE0100, 0xE01EF))
PRIVATE_USE_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def _is_allowed_control(char: str) -> bool:
    return char in {"\n", "\r", "\t"}


def _unicode_name(char: str) -> str:
    return unicodedata.name(char, "<unassigned>")


def _codepoint(char: str) -> str:
    return f"U+{ord(char):04X}"


def _finding_for(char: str, position: int) -> UnicodeFinding | None:
    cp = ord(char)
    general_category = unicodedata.category(char)

    if cp in BIDI_CONTROLS:
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="bidi_control",
            severity="high",
            reason="Bidirectional control character can reorder displayed text.",
            character=char,
        )

    if cp in ZERO_WIDTH_CODEPOINTS:
        severity = "high" if cp in {0x061C, 0x180E, 0xFEFF} else "medium"
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="zero_width",
            severity=severity,
            reason=ZERO_WIDTH_CODEPOINTS[cp],
            character=char,
        )

    if _in_ranges(cp, TAG_CHAR_RANGES):
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="tag_character",
            severity="high",
            reason="Unicode tag character is invisible and can encode hidden data.",
            character=char,
        )

    if cp == SOFT_HYPHEN:
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="format_control",
            severity="medium",
            reason="Soft hyphen is usually invisible and can be used as a carrier.",
            character=char,
        )

    if cp in UNUSUAL_WHITESPACE:
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="unusual_whitespace",
            severity="low",
            reason="Whitespace is visually similar to a regular space but distinct.",
            character=char,
        )

    if _in_ranges(cp, VARIATION_SELECTOR_RANGES):
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="variation_selector",
            severity="low",
            reason="Variation selector is invisible formatting and may be meaningful only in context.",
            character=char,
        )

    if _in_ranges(cp, PRIVATE_USE_RANGES):
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="private_use",
            severity="high",
            reason="Private-use character has no public Unicode semantics.",
            character=char,
        )

    if general_category == "Cf":
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="format_control",
            severity="medium",
            reason="Unicode format control character may be visually hidden.",
            character=char,
        )

    if general_category.startswith("C") and not _is_allowed_control(char):
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="control",
            severity="high",
            reason="Non-printing control character appears in text.",
            character=char,
        )

    if "FULLWIDTH" in _unicode_name(char) or "HALFWIDTH" in _unicode_name(char):
        return UnicodeFinding(
            codepoint=_codepoint(char),
            name=_unicode_name(char),
            position=position,
            category="confusable",
            severity="low",
            reason="Compatibility-width character may be visually confusable.",
            character=char,
        )

    return None


def analyze_unicode(text: str) -> list[UnicodeFinding]:
    """Return one finding per suspicious Unicode character occurrence."""

    findings: list[UnicodeFinding] = []
    for position, char in enumerate(text):
        finding = _finding_for(char, position)
        if finding is not None:
            findings.append(finding)
    return findings
