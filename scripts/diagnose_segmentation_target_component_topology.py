#!/usr/bin/env python3
"""Write the fixed first-ten target/prediction component topology diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.segmentation_target_component_topology import (
    diagnose_target_component_topology,
    write_target_component_per_word_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--segmentation-manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--segmentation-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    csv_output = args.csv_output.resolve()
    if output.exists() or csv_output.exists():
        raise FileExistsError("Refusing to overwrite target-component topology output")
    samples, _ = load_icdar13_recognition_samples(args.manifest, args.data_root)
    report = diagnose_target_component_topology(
        index_path=args.index,
        samples=samples[:10],
        segmentation_manifest_path=args.segmentation_manifest,
        data_root=args.data_root,
        segmentation_checkpoint=args.segmentation_checkpoint,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    csv_temporary = csv_output.parent / ("." + csv_output.name + ".tmp")
    write_target_component_per_word_csv(report, csv_temporary)
    os.replace(csv_temporary, csv_output)
    print(output)
    print(csv_output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
