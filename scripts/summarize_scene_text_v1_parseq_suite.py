#!/usr/bin/env python3
"""Collect the six frozen scene-text V1+PARSeq TEST diagnostics."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile


RUNS = tuple(
    f"{dataset}-{model}"
    for dataset in ("ctw1500", "icdar2015", "total_text")
    for model in ("specialized", "combined")
)


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results.json"
    if output.exists():
        print(output)
        print(_sha(output))
        return 0
    reports = {}
    sources = {}
    for name in RUNS:
        path = root / name / "results.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing suite result: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        reports[name] = report["summary"]
        sources[name] = {"path": str(path), "sha256": _sha(path)}
    payload = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "official-test V1+PARSeq diagnostic; not paper Table 5",
        "reports": reports,
        "sources": sources,
    }
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=root, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    compact = {
        name: {
            "gt": value["geometry"]["gt_instances"],
            "geometry_iou50": value["geometry"]["iou_ge_0_50_one_to_one"],
            "parseq_exact_iou50": (
                None
                if value["recognition"] is None
                else value["recognition"]["exact_casefold_spatial_one_to_one"]
            ),
        }
        for name, value in reports.items()
    }
    print(json.dumps(compact, indent=2), flush=True)
    print(f"JSON {output} SHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
