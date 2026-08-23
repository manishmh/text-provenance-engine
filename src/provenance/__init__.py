"""Text Provenance Engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from provenance.engine import ProvenanceEngine


def __getattr__(name: str) -> type:  # pragma: no cover
    if name == "ProvenanceEngine":
        from provenance.engine import ProvenanceEngine
        return ProvenanceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProvenanceEngine"]
