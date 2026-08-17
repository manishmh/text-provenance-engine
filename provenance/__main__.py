"""Source-tree shim for `python -m provenance`."""

from provenance.cli import main


raise SystemExit(main())
