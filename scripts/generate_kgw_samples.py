#!/usr/bin/env python3
"""Generate controlled KGW simulation positive and negative samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from provenance.detectors.kgw import KGWDetector, generate_controlled_kgw_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled KGW simulation samples")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--model", default="local-simple-tokenizer")
    parser.add_argument("--prompt-id", default="demo")
    args = parser.parse_args()

    detector = KGWDetector.from_config_file(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for watermarked, stem in [(True, "watermarked"), (False, "unwatermarked")]:
        text = generate_controlled_kgw_text(
            detector,
            token_count=args.tokens,
            watermarked=watermarked,
        )
        text_path = args.out_dir / f"{stem}.txt"
        metadata_path = args.out_dir / f"{stem}.metadata.json"
        text_path.write_text(text, encoding="utf-8")
        metadata = {
            "watermarked": watermarked,
            "scheme": "kgw",
            "implementation_kind": "simulation",
            "compatibility": "controlled-local-only",
            "configuration_id": detector.config.configuration_id,
            "configuration_version": detector.config.version,
            "model": args.model,
            "prompt_id": args.prompt_id,
            "token_count": args.tokens,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
