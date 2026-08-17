#!/usr/bin/env python3
"""Generate controlled SynthID-Text simulation positive and negative samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from provenance.detectors.synthid import (  # noqa: E402
    SynthIDTextDetector,
    generate_controlled_synthid_text,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate controlled SynthID-Text simulation samples")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=180)
    parser.add_argument("--model", default="local-simple-tokenizer")
    parser.add_argument("--prompt-id", default="demo")
    args = parser.parse_args()

    detector = SynthIDTextDetector.from_config_file(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for watermarked, stem in [(True, "watermarked"), (False, "unwatermarked")]:
        text = generate_controlled_synthid_text(
            detector,
            token_count=args.tokens,
            watermarked=watermarked,
        )
        text_path = args.out_dir / f"{stem}.txt"
        metadata_path = args.out_dir / f"{stem}.metadata.json"
        text_path.write_text(text, encoding="utf-8")
        metadata = {
            "watermarked": watermarked,
            "scheme": "synthid",
            "implementation_kind": "simulation",
            "compatibility": "controlled-local-only",
            "configuration_id": detector.config.configuration_id,
            "configuration_version": detector.config.version,
            "model": args.model,
            "prompt_id": args.prompt_id,
            "token_count": args.tokens,
            "detector": "weighted_mean",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
