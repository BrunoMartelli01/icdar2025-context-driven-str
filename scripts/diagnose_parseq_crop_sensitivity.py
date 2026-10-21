#!/usr/bin/env python3
"""Run a non-accuracy PARSeq crop/GT sensitivity grid from frozen trace masks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_SOURCE_TYPES,
    ParseqApproxConfig,
    ParseqApproxRecognizer,
)
from recognition.parseq_crop_sensitivity import (
    build_variant_grid,
    diagnose_parseq_crop_sensitivity,
)


def _integers(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument(
        "--parseq-source-type",
        choices=PARSEQ_SOURCE_TYPES,
        default=PARSEQ_ARCHIVE_SOURCE,
    )
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-areas", type=_integers, default=(50, 150, 300))
    parser.add_argument("--paddings", type=_integers, default=(0, 6, 12, 24))
    parser.add_argument("--connectivities", type=_integers, default=(4, 8))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite sensitivity output: {}".format(output))
    samples, dataset = load_icdar13_recognition_samples(args.manifest, args.data_root)
    variants = build_variant_grid(
        args.min_areas, args.paddings, args.connectivities
    )
    recognizer = ParseqApproxRecognizer(
        ParseqApproxConfig(
            runtime_source=args.parseq_source,
            checkpoint=args.parseq_checkpoint,
            device=args.device,
            runtime_source_type=args.parseq_source_type,
            runtime_source_archive=args.parseq_source_archive,
        )
    )
    try:
        report = diagnose_parseq_crop_sensitivity(
            index_path=args.index,
            samples=samples,
            variants=variants,
            recognize_crop=lambda crop: recognizer.recognize(
                crop, input_role="T3_localized_crop"
            ),
            parseq_provenance={
                **dict(recognizer._provenance),
                "device": args.device,
                "dataset_manifest_sha256": dataset["manifest_sha256"],
            },
        )
    finally:
        recognizer.release()
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
