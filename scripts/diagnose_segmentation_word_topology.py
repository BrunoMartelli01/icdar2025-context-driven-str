#!/usr/bin/env python3
"""Run first-ten native segmentation topology and conditional OCR diagnostics."""

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
from recognition.segmentation_word_topology import (
    diagnose_segmentation_word_topology,
    write_per_word_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-type", choices=PARSEQ_SOURCE_TYPES, default=PARSEQ_ARCHIVE_SOURCE)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    csv_output = args.csv_output.resolve()
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite topology diagnostic output")
    samples, dataset_report = load_icdar13_recognition_samples(args.manifest, args.data_root)
    if args.sample_count != 10:
        raise ValueError("This diagnostic is frozen to the first 10 samples")
    samples = samples[:10]
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
        report = diagnose_segmentation_word_topology(
            index_path=args.index,
            samples=samples,
            segmentation_manifest_path=args.segmentation_manifest,
            data_root=args.data_root,
            segmentation_checkpoint=args.segmentation_checkpoint,
            parseq_provenance={
                **dict(recognizer._provenance),
                "device": args.device,
                "dataset_manifest_sha256": dataset_report["manifest_sha256"],
            },
            recognize_crop=lambda crop: recognizer.recognize(crop, input_role="T3_localized_crop"),
        )
    finally:
        recognizer.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    csv_temporary = csv_output.parent / ("." + csv_output.name + ".tmp")
    write_per_word_csv(report, csv_temporary)
    os.replace(csv_temporary, csv_output)
    print(output)
    print(csv_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
