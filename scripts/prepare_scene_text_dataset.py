#!/usr/bin/env python3
"""Build or verify a checksum-frozen scene-text dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recognition.scene_text_manifest import (
    build_scene_text_manifest,
    load_scene_text_manifest,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--dataset", choices=("icdar2015", "total_text", "ctw1500"), required=True)
    build.add_argument("--split", choices=("train", "test"), required=True)
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--images", type=Path, required=True)
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--allow-partial",
        action="store_true",
        help="diagnostic fixtures only; official qualification must omit this flag",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build":
        result = build_scene_text_manifest(
            dataset=args.dataset,
            split=args.split,
            data_root=args.data_root,
            image_directory=args.images,
            annotation_directory=args.annotations,
            output=args.output,
            enforce_official_count=not args.allow_partial,
        )
        report = {
            "status": "manifest_built",
            "dataset": result["dataset"],
            "split": result["split"],
            "sample_count": result["sample_count"],
            "instance_count": result["instance_count"],
            "care_instance_count": result["care_instance_count"],
            "manifest": str(args.output.resolve()),
            "manifest_sha256": sha256_file(args.output.resolve()),
            "table_5_claim": False,
        }
    else:
        _, report = load_scene_text_manifest(args.manifest.resolve(), args.data_root.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
