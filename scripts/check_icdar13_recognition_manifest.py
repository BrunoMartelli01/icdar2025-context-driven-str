#!/usr/bin/env python3
"""Verify the frozen ICDAR13 recognition population without defining a metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recognition.icdar13_annotations import load_icdar13_recognition_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, report = load_icdar13_recognition_samples(args.manifest, args.data_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
