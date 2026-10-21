#!/usr/bin/env python3
"""Emit explicit non-accuracy alignment diagnostics from existing trace outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq_deepsolo_alignment import diagnose_parseq_deepsolo_alignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples, _ = load_icdar13_recognition_samples(args.manifest, args.data_root)
    report = diagnose_parseq_deepsolo_alignment(args.index, samples)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite alignment output: {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=output.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
