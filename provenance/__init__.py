"""Source-tree shim for running without installation."""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "provenance"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))

from provenance.engine import ProvenanceEngine  # noqa: E402

__all__ = ["ProvenanceEngine"]
