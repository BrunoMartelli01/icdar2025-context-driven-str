"""Prepare CTW1500, ICDAR2015, and Total-Text segmentation datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from segmentation.scene_text_preparation import prepare_scene_text_segmentation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    report = prepare_scene_text_segmentation(args.download_root, args.output_root)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
