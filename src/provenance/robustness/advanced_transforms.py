"""Advanced robustness transformations with structured taxonomy.

Phase 5D extends the Phase 5A transformation framework with a broader set
of realistic, controlled text perturbations organized by category.

Every transformation is:
- Deterministic and reproducible from a seed/configuration
- Bounded and controlled (no adversarial optimization)
- Never uses external LLMs or online APIs
- Designed for measurement, not evasion

Safety boundary: These transforms measure robustness under controlled
perturbations. They do NOT implement adversarial attacks, watermark
removal, gradient-based evasion, or automated paraphrasing.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ── Transform categories ──────────────────────────────────────────────

class TransformCategory(str, Enum):
    """Categories for classifying text transformations."""
    FORMATTING = "formatting"
    WHITESPACE = "whitespace"
    UNICODE = "unicode"
    CASING = "casing"
    PUNCTUATION = "punctuation"
    LEXICAL = "lexical"
    TOKENIZATION_SENSITIVE = "tokenization-sensitive"


# ── Advanced Transform dataclass ──────────────────────────────────────

@dataclass(frozen=True)
class AdvancedTransform:
    """A named, deterministic text transformation with rich metadata."""
    name: str
    category: TransformCategory
    description: str
    func: Callable[[str, int], str]
    deterministic: bool = True
    severity: str = "low"  # low, medium, high — relative text modification magnitude

    def apply(self, text: str, seed: int = 0) -> str:
        """Apply this transformation to text. Seed ensures reproducibility."""
        return self.func(text, seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "deterministic": self.deterministic,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for a single transformation."""
    name: str
    category: str
    enabled: bool = True
    severity: str | None = None
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "enabled": self.enabled,
            "severity": self.severity,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformConfig:
        return cls(
            name=data["name"],
            category=data["category"],
            enabled=data.get("enabled", True),
            severity=data.get("severity"),
            seed=data.get("seed", 0),
        )


@dataclass(frozen=True)
class TransformProfile:
    """A named collection of transformations."""
    name: str
    description: str
    transform_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "transform_names": list(self.transform_names),
        }


# ── Seeded RNG for deterministic behavior ─────────────────────────────

def _seeded_random(seed: int) -> float:
    """Deterministic pseudo-random float in [0, 1) from seed."""
    h = hashlib.sha256(str(seed).encode()).hexdigest()
    return int(h[:8], 16) / (16**8)


# ── Formatting transforms ─────────────────────────────────────────────

def _paragraph_reflow(text: str, seed: int = 0) -> str:
    """Reflow paragraphs: collapse multiple blank lines into single, normalize indentation."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    # Normalize each paragraph: strip leading/trailing whitespace
    cleaned = ['\n'.join(line.strip() for line in p.split('\n')).strip() for p in paragraphs]
    return '\n\n'.join(p for p in cleaned if p)


def _line_wrap_normalize(text: str, seed: int = 0) -> str:
    """Normalize line wrapping: join broken lines within paragraphs."""
    lines = text.split('\n')
    result = []
    current: list[str] = []
    hyphenated = False
    for line in lines:
        stripped = line.strip()
        if stripped == '':
            if current:
                result.append(''.join(current))
                current = []
                hyphenated = False
            result.append('')
        elif stripped.endswith('-'):
            # Hyphenated line break — join without space
            current.append(stripped[:-1])
            hyphenated = True
        else:
            if current and not hyphenated:
                current.append(' ')
            current.append(stripped)
            hyphenated = False
    if current:
        result.append(''.join(current))
    return '\n'.join(result).strip()


def _blank_line_normalize(text: str, seed: int = 0) -> str:
    """Normalize blank lines: collapse runs of 3+ newlines to exactly 2."""
    return re.sub(r'\n{3,}', '\n\n', text).strip()


# ── Whitespace transforms ─────────────────────────────────────────────

def _whitespace_collapse(text: str, seed: int = 0) -> str:
    """Collapse runs of spaces/tabs to single space, preserving newlines."""
    result = re.sub(r'[ \t]+', ' ', text)
    return result


def _whitespace_expand(text: str, seed: int = 0) -> str:
    """Expand single spaces to double spaces."""
    return text.replace(' ', '  ')


def _tab_space_normalize(text: str, seed: int = 0) -> str:
    """Convert tabs to spaces (4-space width)."""
    return text.expandtabs(4)


def _leading_trailing_whitespace(text: str, seed: int = 0) -> str:
    """Add leading and trailing whitespace."""
    return '  ' + text + '  '


def _double_spaces(text: str, seed: int = 0) -> str:
    """Replace single spaces with double spaces."""
    return text.replace(' ', '  ')


# ── Unicode transforms ────────────────────────────────────────────────

def _unicode_nfc(text: str, seed: int = 0) -> str:
    """Apply Unicode NFC normalization."""
    return unicodedata.normalize('NFC', text)


def _unicode_nfd(text: str, seed: int = 0) -> str:
    """Apply Unicode NFD normalization."""
    return unicodedata.normalize('NFD', text)


def _unicode_nfkd(text: str, seed: int = 0) -> str:
    """Apply Unicode NFKD (compatibility decomposition) normalization.

    Decomposes compatibility characters (e.g., ﬁ → fi, ² → 2).
    Safe for benchmarking — does not modify semantic content.
    """
    return unicodedata.normalize('NFKD', text)


def _unicode_punctuation_normalize(text: str, seed: int = 0) -> str:
    """Normalize Unicode punctuation to ASCII equivalents."""
    replacements = {
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201C': '"',   # left double quote
        '\u201D': '"',   # right double quote
        '\u2013': '-',   # en dash
        '\u2014': '-',   # em dash
        '\u2026': '...',  # ellipsis
        '\u00A0': ' ',   # non-breaking space
        '\u200B': '',    # zero-width space
        '\u200C': '',    # zero-width non-joiner
        '\u200D': '',    # zero-width joiner
        '\uFEFF': '',    # BOM
    }
    result = text
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


# ── Casing transforms ─────────────────────────────────────────────────

def _lowercase(text: str, seed: int = 0) -> str:
    """Convert to lowercase."""
    return text.lower()


def _uppercase(text: str, seed: int = 0) -> str:
    """Convert to uppercase."""
    return text.upper()


def _title_case(text: str, seed: int = 0) -> str:
    """Convert to title case. Only capitalizes first letter of each word."""
    return text.title()


# ── Punctuation transforms ────────────────────────────────────────────

def _punctuation_normalize(text: str, seed: int = 0) -> str:
    """Normalize Unicode punctuation to ASCII equivalents."""
    return _unicode_punctuation_normalize(text, seed)


def _repeated_punctuation_normalize(text: str, seed: int = 0) -> str:
    """Normalize repeated punctuation (e.g., !!! → !, ... → ...)."""
    result = re.sub(r'!{2,}', '!', text)
    result = re.sub(r'\?{2,}', '?', result)
    result = re.sub(r'\.{4,}', '...', result)
    return result


# ── Lexical transforms ────────────────────────────────────────────────

# Deterministic synonym mapping (conservative, no semantic change)
_SYNONYM_MAP: dict[str, list[str]] = {
    'very': ['quite', 'rather'],
    'big': ['large', 'great'],
    'small': ['little', 'tiny'],
    'good': ['fine', 'decent'],
    'bad': ['poor', 'weak'],
    'fast': ['quick', 'rapid'],
    'slow': ['gradual', 'steady'],
    'happy': ['glad', 'pleased'],
    'sad': ['somber', 'grim'],
    'start': ['begin', 'commence'],
    'end': ['finish', 'conclude'],
    'help': ['assist', 'aid'],
    'use': ['employ', 'utilize'],
    'make': ['create', 'produce'],
    'get': ['obtain', 'receive'],
    'give': ['provide', 'offer'],
    'take': ['accept', 'receive'],
    'see': ['observe', 'notice'],
    'know': ['understand', 'comprehend'],
    'think': ['believe', 'consider'],
    'say': ['state', 'mention'],
    'tell': ['inform', 'advise'],
    'ask': ['inquire', 'question'],
    'try': ['attempt', 'endeavor'],
    'need': ['require', 'demand'],
    'want': ['desire', 'wish'],
    'like': ['enjoy', 'appreciate'],
    'show': ['demonstrate', 'display'],
    'seem': ['appear', 'look'],
    'find': ['discover', 'locate'],
    'come': ['arrive', 'approach'],
    'go': ['travel', 'proceed'],
    'put': ['place', 'position'],
    'keep': ['retain', 'maintain'],
    'let': ['allow', 'permit'],
    'begin': ['start', 'commence'],
    'feel': ['sense', 'experience'],
    'bring': ['carry', 'deliver'],
    'happen': ['occur', 'transpire'],
    'write': ['compose', 'draft'],
    'sit': ['settle', 'rest'],
    'stand': ['remain', 'endure'],
    'lose': ['misplace', 'forfeit'],
    'pay': ['compensate', 'remit'],
    'meet': ['encounter', 'greet'],
    'include': ['contain', 'encompass'],
    'continue': ['persist', 'proceed'],
    'set': ['establish', 'fix'],
    'learn': ['study', 'acquire'],
    'change': ['alter', 'modify'],
    'lead': ['direct', 'guide'],
    'understand': ['comprehend', 'grasp'],
    'watch': ['observe', 'monitor'],
    'follow': ['pursue', 'track'],
    'stop': ['cease', 'halt'],
    'speak': ['talk', 'converse'],
    'read': ['peruse', 'examine'],
    'spend': ['expend', 'allocate'],
    'grow': ['expand', 'increase'],
    'open': ['unseal', 'reveal'],
    'walk': ['stroll', 'traverse'],
    'win': ['prevail', 'triumph'],
    'offer': ['propose', 'present'],
    'remember': ['recall', 'recollect'],
    'consider': ['ponder', 'contemplate'],
    'appear': ['emerge', 'surface'],
    'buy': ['purchase', 'acquire'],
    'serve': ['attend', 'cater'],
    'die': ['perish', 'cease'],
    'send': ['dispatch', 'transmit'],
    'build': ['construct', 'erect'],
    'stay': ['remain', 'linger'],
    'fall': ['drop', 'descend'],
    'cut': ['trim', 'slice'],
    'reach': ['attain', 'arrive'],
    'kill': ['eliminate', 'destroy'],
    'remain': ['persist', 'endure'],
    'suggest': ['propose', 'recommend'],
    'raise': ['elevate', 'lift'],
    'pass': ['proceed', 'traverse'],
    'sell': ['trade', 'vend'],
    'require': ['demand', 'necessitate'],
    'report': ['state', 'announce'],
    'decide': ['determine', 'resolve'],
    'pull': ['draw', 'tug'],
}


def _conservative_synonym_substitution(text: str, seed: int = 0) -> str:
    """Conservative synonym substitution using deterministic local mapping.

    Only substitutes words that appear in the synonym map. Uses seed
    to deterministically choose among alternatives. Preserves case of
    the original word.
    """
    words = text.split(' ')
    result = []
    for i, word in enumerate(words):
        # Strip punctuation for lookup
        stripped = re.sub(r'[^\w]', '', word).lower()
        if stripped in _SYNONYM_MAP:
            alternatives = _SYNONYM_MAP[stripped]
            # Deterministic choice based on seed and position
            idx = int(_seeded_random(seed + i * 1000) * len(alternatives))
            replacement = alternatives[idx]
            # Preserve case
            if word[0].isupper():
                replacement = replacement.capitalize()
            # Preserve trailing punctuation
            trailing = word[len(stripped):]
            result.append(replacement + trailing)
        else:
            result.append(word)
    return ' '.join(result)


# Contraction mappings
_CONTRACTION_EXPANSIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "mustn't": "must not",
    "let's": "let us",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "there's": "there is",
    "here's": "here is",
    "where's": "where is",
    "when's": "when is",
    "how's": "how is",
    "i'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
}


def _contraction_expansion(text: str, seed: int = 0) -> str:
    """Expand contractions using a deterministic mapping.

    Preserves the original word boundaries and punctuation.
    """
    result = text
    for contraction, expansion in _CONTRACTION_EXPANSIONS.items():
        # Case-insensitive replacement, preserve original case where possible
        pattern = re.compile(re.escape(contraction), re.IGNORECASE)
        def _replace(match):
            orig = match.group(0)
            # Capitalize first letter if original was capitalized
            if orig[0].isupper():
                return expansion[0].upper() + expansion[1:]
            return expansion
        result = pattern.sub(_replace, result)
    return result


# ── Tokenization-sensitive transforms ─────────────────────────────────

def _insert_formatting_boundaries(text: str, seed: int = 0) -> str:
    """Insert harmless formatting boundaries (extra spaces around punctuation).

    This simulates minor formatting noise that may affect tokenization.
    """
    # Add space before certain punctuation
    result = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)
    return result


def _remove_formatting_boundaries(text: str, seed: int = 0) -> str:
    """Remove spaces around punctuation (aggressive minification)."""
    result = re.sub(r'\s+([.,;:!?])', r'\1', text)
    return result


# ── Registry ──────────────────────────────────────────────────────────

ADVANCED_TRANSFORMS: list[AdvancedTransform] = [
    # Formatting
    AdvancedTransform(
        name="paragraph_reflow",
        category=TransformCategory.FORMATTING,
        description="Reflow paragraphs: collapse blank lines, normalize indentation",
        func=_paragraph_reflow,
        severity="medium",
    ),
    AdvancedTransform(
        name="line_wrap_normalize",
        category=TransformCategory.FORMATTING,
        description="Normalize line wrapping within paragraphs",
        func=_line_wrap_normalize,
        severity="low",
    ),
    AdvancedTransform(
        name="blank_line_normalize",
        category=TransformCategory.FORMATTING,
        description="Collapse runs of blank lines to single blank line",
        func=_blank_line_normalize,
        severity="low",
    ),

    # Whitespace
    AdvancedTransform(
        name="whitespace_collapse",
        category=TransformCategory.WHITESPACE,
        description="Collapse runs of spaces/tabs to single space",
        func=_whitespace_collapse,
        severity="low",
    ),
    AdvancedTransform(
        name="whitespace_expand",
        category=TransformCategory.WHITESPACE,
        description="Expand single spaces to double spaces",
        func=_whitespace_expand,
        severity="low",
    ),
    AdvancedTransform(
        name="tab_space_normalize",
        category=TransformCategory.WHITESPACE,
        description="Convert tabs to 4-space width",
        func=_tab_space_normalize,
        severity="low",
    ),
    AdvancedTransform(
        name="leading_trailing_whitespace",
        category=TransformCategory.WHITESPACE,
        description="Add leading and trailing whitespace",
        func=_leading_trailing_whitespace,
        severity="low",
    ),
    AdvancedTransform(
        name="double_spaces",
        category=TransformCategory.WHITESPACE,
        description="Replace single spaces with double spaces",
        func=_double_spaces,
        severity="low",
    ),

    # Unicode
    AdvancedTransform(
        name="unicode_nfc",
        category=TransformCategory.UNICODE,
        description="Unicode NFC normalization",
        func=_unicode_nfc,
        severity="low",
    ),
    AdvancedTransform(
        name="unicode_nfd",
        category=TransformCategory.UNICODE,
        description="Unicode NFD normalization",
        func=_unicode_nfd,
        severity="low",
    ),
    AdvancedTransform(
        name="unicode_nfkd",
        category=TransformCategory.UNICODE,
        description="Unicode NFKD compatibility decomposition",
        func=_unicode_nfkd,
        severity="medium",
    ),
    AdvancedTransform(
        name="unicode_punctuation_normalize",
        category=TransformCategory.UNICODE,
        description="Normalize Unicode punctuation to ASCII equivalents",
        func=_unicode_punctuation_normalize,
        severity="low",
    ),

    # Casing
    AdvancedTransform(
        name="lowercase",
        category=TransformCategory.CASING,
        description="Convert to lowercase",
        func=_lowercase,
        severity="low",
    ),
    AdvancedTransform(
        name="uppercase",
        category=TransformCategory.CASING,
        description="Convert to uppercase",
        func=_uppercase,
        severity="medium",
    ),
    AdvancedTransform(
        name="title_case",
        category=TransformCategory.CASING,
        description="Convert to title case",
        func=_title_case,
        severity="medium",
    ),

    # Punctuation
    AdvancedTransform(
        name="punctuation_normalize",
        category=TransformCategory.PUNCTUATION,
        description="Normalize Unicode punctuation to ASCII equivalents",
        func=_punctuation_normalize,
        severity="low",
    ),
    AdvancedTransform(
        name="repeated_punctuation_normalize",
        category=TransformCategory.PUNCTUATION,
        description="Normalize repeated punctuation marks",
        func=_repeated_punctuation_normalize,
        severity="low",
    ),

    # Lexical
    AdvancedTransform(
        name="conservative_synonym_substitution",
        category=TransformCategory.LEXICAL,
        description="Conservative synonym substitution using deterministic local mapping",
        func=_conservative_synonym_substitution,
        severity="medium",
    ),
    AdvancedTransform(
        name="contraction_expansion",
        category=TransformCategory.LEXICAL,
        description="Expand contractions using deterministic mapping",
        func=_contraction_expansion,
        severity="low",
    ),

    # Tokenization-sensitive
    AdvancedTransform(
        name="insert_formatting_boundaries",
        category=TransformCategory.TOKENIZATION_SENSITIVE,
        description="Insert spaces around punctuation",
        func=_insert_formatting_boundaries,
        severity="low",
    ),
    AdvancedTransform(
        name="remove_formatting_boundaries",
        category=TransformCategory.TOKENIZATION_SENSITIVE,
        description="Remove spaces around punctuation",
        func=_remove_formatting_boundaries,
        severity="low",
    ),
]

ADVANCED_TRANSFORM_MAP: dict[str, AdvancedTransform] = {
    t.name: t for t in ADVANCED_TRANSFORMS
}


def get_advanced_transform(name: str) -> AdvancedTransform:
    """Get an advanced transform by name."""
    if name not in ADVANCED_TRANSFORM_MAP:
        raise ValueError(f"Unknown advanced transform: {name!r}. Available: {list(ADVANCED_TRANSFORM_MAP.keys())}")
    return ADVANCED_TRANSFORM_MAP[name]


def get_all_advanced_transforms() -> list[AdvancedTransform]:
    """Get all registered advanced transforms."""
    return list(ADVANCED_TRANSFORMS)


def get_transforms_by_category(category: TransformCategory) -> list[AdvancedTransform]:
    """Get all transforms in a category."""
    return [t for t in ADVANCED_TRANSFORMS if t.category == category]


def get_categories() -> list[TransformCategory]:
    """Get all available categories."""
    seen: set[TransformCategory] = set()
    result: list[TransformCategory] = []
    for t in ADVANCED_TRANSFORMS:
        if t.category not in seen:
            result.append(t.category)
            seen.add(t.category)
    return result


def apply_advanced_transform(text: str, transform_name: str, seed: int = 0) -> str:
    """Apply a named advanced transform to text."""
    return get_advanced_transform(transform_name).apply(text, seed)
