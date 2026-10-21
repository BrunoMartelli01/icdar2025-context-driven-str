#!/usr/bin/env python3
"""Validate the frozen isolated runtime without importing it into the main process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepsolo.candidates import (  # noqa: E402
    GIT_CHECKOUT_SOURCE_TYPE,
    GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
    QUALIFIED_CHECKPOINT_SHA256,
    sha256_file,
    verify_source_identity,
)


EXPECTED = {
    "pip": "23.3.2",
    "setuptools": "59.5.0",
    "wheel": "0.37.1",
    "torch": "1.9.0+cu111",
    "torchvision": "0.10.0+cu111",
    "detectron2": "0.6+cu111",
    "numpy": "1.20.0",
    "opencv-python": "4.5.5.62",
}
DEEPSOLO_REVISION = "dbadae995035246bad3376c7a44c015c69e9b313"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-type",
        choices=(GIT_CHECKOUT_SOURCE_TYPE, GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE),
        default=GIT_CHECKOUT_SOURCE_TYPE,
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--source-commit-marker", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-archive-sha256")
    args = parser.parse_args()
    identity = verify_source_identity(
        args.source_root,
        source_type=args.source_type,
        declared_revision=args.source_revision,
        source_commit_marker=args.source_commit_marker,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
    )
    checkpoint_sha256 = sha256_file(args.checkpoint.resolve())
    versions = {name: importlib.metadata.version(name) for name in EXPECTED}
    problems = []
    if sys.version_info[:2] != (3, 8):
        problems.append("python expected major/minor 3.8, got {}".format(platform.python_version()))
    for name, expected in EXPECTED.items():
        if versions[name] != expected:
            problems.append("{} expected {}, got {}".format(name, expected, versions[name]))
    if identity["source_revision"] != DEEPSOLO_REVISION:
        problems.append(
            "DeepSolo revision mismatch: {}".format(identity["source_revision"])
        )
    if checkpoint_sha256 != QUALIFIED_CHECKPOINT_SHA256:
        problems.append("DeepSolo checkpoint SHA256 mismatch")
    import torch
    import detectron2
    import adet._C

    if torch.version.cuda != "11.1":
        problems.append("torch CUDA runtime expected 11.1, got {}".format(torch.version.cuda))
    if not torch.cuda.is_available():
        problems.append("CUDA is unavailable in context-str-deepsolo")

    payload = {
        "success": not problems,
        "environment": "context-str-deepsolo",
        "python": platform.python_version(),
        "packages": versions,
        "deepsolo_revision": identity["source_revision"],
        "source_type": identity["source_type"],
        "source_revision": identity["source_revision"],
        "source_revision_evidence": identity["source_revision_evidence"],
        "source_archive_sha256": identity["source_archive_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "problems": problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
