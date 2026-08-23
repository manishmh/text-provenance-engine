"""Source-tree shim for running without installation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "provenance"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

if TYPE_CHECKING:
    from provenance.engine import ProvenanceEngine  # noqa: E402


def __getattr__(name: str) -> type:  # pragma: no cover
    if name == "ProvenanceEngine":
        from provenance.engine import ProvenanceEngine
        return ProvenanceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ProvenanceEngine"]
