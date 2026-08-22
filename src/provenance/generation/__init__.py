"""Controlled sample generation utilities."""

from provenance.generation.kgw import (
    KGWGenerationConfig,
    KGWLogitsProcessor,
    generate_kgw_token_ids,
    sample_from_logits,
)
from provenance.generation.synthid import (
    SynthIDGenerationConfig,
    SynthIDLogitsProcessor,
    generate_reference_synthid_text,
    generate_synthid_token_ids,
)

__all__ = ["KGWGenerationConfig", "KGWLogitsProcessor", "generate_kgw_token_ids", "sample_from_logits", "SynthIDGenerationConfig", "SynthIDLogitsProcessor", "generate_reference_synthid_text", "generate_synthid_token_ids"]
