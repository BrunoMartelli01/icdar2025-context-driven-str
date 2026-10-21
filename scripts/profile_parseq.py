#!/usr/bin/env python3
"""Run the qualified APPROXIMATE PARSeq adapter on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.parseq import (
    PARSEQ_GIT_SOURCE,
    PARSEQ_SOURCE_TYPES,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument(
        "--runtime-source-type", choices=PARSEQ_SOURCE_TYPES,
        default=PARSEQ_GIT_SOURCE,
    )
    parser.add_argument("--runtime-source-archive", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--role", choices=("T1_full_image", "T3_localized_crop"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
        runtime_source=args.runtime_source,
        checkpoint=args.checkpoint,
        device=args.device,
        runtime_source_type=args.runtime_source_type,
        runtime_source_archive=args.runtime_source_archive,
    ))
    payload = recognizer.recognize(image, input_role=args.role).to_dict()
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
