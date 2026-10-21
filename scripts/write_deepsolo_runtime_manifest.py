#!/usr/bin/env python3
"""Create an untracked runtime manifest from explicit candidate/checkpoint pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deepsolo.candidates import (
    CANDIDATES, DEEPSOLO_SOURCE_REVISION, GIT_CHECKOUT_SOURCE_TYPE,
    GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE, inspect_candidate_source, sha256_file,
    verify_source_identity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--source-type",
        choices=(GIT_CHECKOUT_SOURCE_TYPE, GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE),
        default=GIT_CHECKOUT_SOURCE_TYPE,
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--source-commit-marker", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--source-archive-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-checkpoint", action="append", required=True,
        metavar="ARTIFACT_ID=CHECKPOINT_PATH",
    )
    parser.add_argument("--runtime-kind", choices=("host_subprocess",))
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--runtime-environment-report", type=Path)
    args = parser.parse_args()
    identity = verify_source_identity(
        args.source_root,
        source_type=args.source_type,
        declared_revision=args.source_revision,
        source_commit_marker=args.source_commit_marker,
        source_archive=args.source_archive,
        source_archive_sha256=args.source_archive_sha256,
    )
    resolved = {}
    for value in args.candidate_checkpoint:
        artifact_id, separator, raw_path = value.partition("=")
        if not separator or artifact_id not in CANDIDATES:
            parser.error("--candidate-checkpoint requires a known ARTIFACT_ID=CHECKPOINT_PATH")
        checkpoint = Path(raw_path).resolve()
        if not checkpoint.is_file():
            parser.error("checkpoint does not exist: {}".format(checkpoint))
        inspect_candidate_source(
            args.source_root,
            artifact_id,
            verified_source_identity=identity,
        )
        resolved[artifact_id] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    runtime = {}
    if args.runtime_kind is not None:
        if args.runtime_python is None or args.runtime_environment_report is None:
            parser.error(
                "--runtime-kind requires --runtime-python and --runtime-environment-report"
            )
        # Preserve the venv entry-point path in provenance.  Resolving its
        # symlink records the base interpreter and would bypass the isolated
        # environment if that recorded path were executed later.
        runtime_python = args.runtime_python.absolute()
        environment_report = args.runtime_environment_report.resolve()
        if not runtime_python.is_file():
            parser.error("runtime Python does not exist: {}".format(runtime_python))
        if not environment_report.is_file():
            parser.error("runtime environment report does not exist: {}".format(environment_report))
        report_payload = json.loads(environment_report.read_text(encoding="utf-8"))
        if report_payload.get("success") is not True:
            parser.error("runtime environment report is not successful")
        if (
            report_payload.get("source_type") != identity["source_type"]
            or report_payload.get("source_revision") != identity["source_revision"]
            or report_payload.get("source_archive_sha256")
            != identity["source_archive_sha256"]
        ):
            parser.error("runtime environment report source identity does not match")
        checkpoint_digests = {
            entry["checkpoint_sha256"] for entry in resolved.values()
        }
        if report_payload.get("checkpoint_sha256") not in checkpoint_digests:
            parser.error("runtime environment report checkpoint identity does not match")
        runtime = {
            "kind": args.runtime_kind,
            "artifact_id": "ART-DEEPSOLO-PAPER-HOST-RUNTIME-CANDIDATE",
            "classification": "HARDWARE_DEVIATION",
            "python_executable": str(runtime_python),
            "environment_report": str(environment_report),
            "environment_report_sha256": hashlib.sha256(
                environment_report.read_bytes()
            ).hexdigest(),
        }
    payload = {
        "schema_version": 1,
        "source_root": str(args.source_root.resolve()),
        "source_revision": DEEPSOLO_SOURCE_REVISION,
        "source_type": identity["source_type"],
        "source_revision_evidence": identity["source_revision_evidence"],
        "source_archive": identity["source_archive"],
        "source_archive_sha256": identity["source_archive_sha256"],
        "candidates": resolved,
        "runtime": runtime,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
