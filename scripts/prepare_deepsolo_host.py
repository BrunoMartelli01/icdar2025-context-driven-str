#!/usr/bin/env python3
"""Create an isolated, non-Docker DeepSolo host runtime from frozen inputs.

Run this module from ``context-str-main``. It never installs packages into the
main environment and never downloads or selects a model checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from deepsolo.candidates import (
    DEEPSOLO_SOURCE_REVISION,
    GIT_CHECKOUT_SOURCE_TYPE,
    GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
    QUALIFIED_CANDIDATE_ID,
    QUALIFIED_CHECKPOINT_SHA256,
    inspect_candidate_source,
    sha256_file,
    verify_source_identity,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "environments" / "deepsolo" / "environment.yml"
MAIN_VENV = (ROOT / ".venvs" / "context-str-main").resolve()
EXPECTED_CONFIG_SHA256 = "0da913239e4b3f11ae1bad19e0b104996f7c3f3607aa2a7f4b1f52e3838b702d"
BUILD_BOOTSTRAP_REQUIREMENTS = (
    "setuptools==59.5.0",
    "wheel==0.37.1",
    "numpy==1.20.0",
)
LEGACY_SOURCE_BUILD_REQUIREMENT = "scikit-image==0.15.0"


def _run(command: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
    printable = " ".join(shlex.quote(value) for value in command)
    print("+ {}".format(printable), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def _load_install_spec(path: Path) -> Tuple[str, List[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    dependencies = payload.get("dependencies") if isinstance(payload, dict) else None
    if not isinstance(dependencies, list):
        raise ValueError("DeepSolo environment spec has no dependency list")
    pip_version: Optional[str] = None
    pip_requirements: Optional[List[str]] = None
    for entry in dependencies:
        if isinstance(entry, str) and entry.startswith("pip="):
            pip_version = entry.partition("=")[2]
        if isinstance(entry, dict) and "pip" in entry:
            values = entry["pip"]
            if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
                raise ValueError("DeepSolo pip dependency list is invalid")
            pip_requirements = []
            for item in values:
                pip_requirements.extend(shlex.split(item))
    if not pip_version or not pip_requirements:
        raise ValueError("DeepSolo environment spec must pin pip and pip requirements")
    return pip_version, pip_requirements


def _partition_install_requirements(
    arguments: List[str],
) -> Tuple[List[str], List[str]]:
    """Return pip index arguments and package requirements from the frozen spec."""
    index_arguments: List[str] = []
    requirements: List[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--find-links":
            if index + 1 >= len(arguments):
                raise ValueError("--find-links in DeepSolo spec requires a value")
            index_arguments.extend(arguments[index:index + 2])
            index += 2
            continue
        if value.startswith("--"):
            raise ValueError("Unsupported pip option in DeepSolo spec: {}".format(value))
        requirements.append(value)
        index += 1
    required = set(BUILD_BOOTSTRAP_REQUIREMENTS)
    required.add(LEGACY_SOURCE_BUILD_REQUIREMENT)
    missing = sorted(required.difference(requirements))
    if missing:
        raise ValueError(
            "DeepSolo spec is missing required legacy build pins: {}".format(
                ", ".join(missing)
            )
        )
    return index_arguments, requirements


def _validate_paths(venv: Path, source: Path, checkpoint: Path, report: Path) -> None:
    if venv == MAIN_VENV or MAIN_VENV in venv.parents:
        raise ValueError("Refusing to modify context-str-main or a path below it")
    if "deepsolo" not in venv.name.lower():
        raise ValueError("The isolated venv basename must contain 'deepsolo'")
    if not source.is_dir():
        raise FileNotFoundError("Frozen DeepSolo source is missing: {}".format(source))
    if not checkpoint.is_file():
        raise FileNotFoundError("Frozen DeepSolo checkpoint is missing: {}".format(checkpoint))
    if ROOT == source or ROOT in source.parents:
        raise ValueError("DeepSolo upstream source must remain outside this repository")
    if ROOT == checkpoint or ROOT in checkpoint.parents:
        raise ValueError("DeepSolo checkpoint must remain outside this repository")
    if ROOT == report or ROOT in report.parents:
        raise ValueError("Runtime qualification output must remain outside this repository")


def _validate_frozen_inputs(
    source: Path,
    checkpoint: Path,
    *,
    source_type: str = GIT_CHECKOUT_SOURCE_TYPE,
    declared_revision: Optional[str] = None,
    source_commit_marker: Optional[Path] = None,
    source_archive: Optional[Path] = None,
    source_archive_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    identity = verify_source_identity(
        source,
        source_type=source_type,
        declared_revision=declared_revision,
        source_commit_marker=source_commit_marker,
        source_archive=source_archive,
        source_archive_sha256=source_archive_sha256,
    )
    inspected = inspect_candidate_source(
        source,
        QUALIFIED_CANDIDATE_ID,
        verified_source_identity=identity,
    )
    if inspected["source_revision"] != DEEPSOLO_SOURCE_REVISION:
        raise RuntimeError("DeepSolo source revision mismatch")
    if inspected["config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("DeepSolo config SHA256 mismatch")
    digest = sha256_file(checkpoint)
    if digest != QUALIFIED_CHECKPOINT_SHA256:
        raise RuntimeError("DeepSolo checkpoint SHA256 mismatch")
    return {
        **identity,
        "config_sha256": inspected["config_sha256"],
        "checkpoint_sha256": digest,
    }


def _build_environment(cuda_home: Path) -> Dict[str, str]:
    environment = dict(os.environ)
    environment.update({
        "CUDA_HOME": str(cuda_home),
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "TORCH_CUDA_ARCH_LIST": "8.6",
        "MAX_JOBS": "2",
        "FORCE_CUDA": "1",
        "PATH": "{}:{}".format(cuda_home / "bin", environment.get("PATH", "")),
        "LD_LIBRARY_PATH": "{}:{}".format(
            cuda_home / "lib64", environment.get("LD_LIBRARY_PATH", "")
        ),
    })
    return environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path, default=Path("/usr/bin/python3.8"))
    parser.add_argument("--venv", type=Path, required=True)
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-11.1"))
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Optional complete local wheel directory; enables pip --no-index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_python = args.base_python.resolve()
    venv = args.venv.resolve()
    source = args.source_root.resolve()
    source_commit_marker = (
        args.source_commit_marker.resolve()
        if args.source_commit_marker is not None else None
    )
    source_archive = (
        args.source_archive.resolve() if args.source_archive is not None else None
    )
    checkpoint = args.checkpoint.resolve()
    report = args.report.resolve()
    spec = args.spec.resolve()
    cuda_home = args.cuda_home.resolve()
    _validate_paths(venv, source, checkpoint, report)
    identities = _validate_frozen_inputs(
        source,
        checkpoint,
        source_type=args.source_type,
        declared_revision=args.source_revision,
        source_commit_marker=source_commit_marker,
        source_archive=source_archive,
        source_archive_sha256=args.source_archive_sha256,
    )
    if not base_python.is_file():
        raise FileNotFoundError("Python 3.8 executable is missing: {}".format(base_python))
    version = subprocess.run(
        [str(base_python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not version.startswith("3.8."):
        raise RuntimeError("DeepSolo host runtime requires Python 3.8, got {}".format(version))
    nvcc = cuda_home / "bin" / "nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError("CUDA 11.1 nvcc is missing: {}".format(nvcc))
    pip_version, raw_requirements = _load_install_spec(spec)
    index_arguments, requirements = _partition_install_requirements(raw_requirements)
    if args.wheelhouse is not None:
        wheelhouse = args.wheelhouse.resolve()
        if not wheelhouse.is_dir():
            raise FileNotFoundError("Wheelhouse is missing: {}".format(wheelhouse))
        index_arguments = ["--no-index", "--find-links", str(wheelhouse)]

    if not venv.exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        _run([str(base_python), "-m", "venv", str(venv)])
    elif not (venv / "pyvenv.cfg").is_file():
        raise RuntimeError("Existing target is not a Python venv: {}".format(venv))
    python = venv / "bin" / "python"
    build_environment = _build_environment(cuda_home)
    _run(
        [str(python), "-m", "pip", "install"]
        + index_arguments
        + ["pip=={}".format(pip_version)]
    )
    _run(
        [str(python), "-m", "pip", "install"]
        + index_arguments
        + list(BUILD_BOOTSTRAP_REQUIREMENTS),
        env=build_environment,
    )
    _run(
        [str(python), "-m", "pip", "install"]
        + index_arguments
        + ["--no-build-isolation", "--no-deps", LEGACY_SOURCE_BUILD_REQUIREMENT],
        env=build_environment,
    )
    _run(
        [str(python), "-m", "pip", "install"] + index_arguments + requirements,
        env=build_environment,
    )
    _run(
        [str(python), "-m", "pip", "install", "--no-build-isolation", "--editable", str(source)],
        env=build_environment,
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    source_identity_arguments = [
        "--source-type", identities["source_type"],
        "--source-revision", identities["source_revision"],
    ]
    if identities["source_type"] == GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE:
        source_identity_arguments.extend([
            "--source-commit-marker",
            identities["source_revision_evidence"]["commit_marker_path"],
            "--source-archive", identities["source_archive"],
            "--source-archive-sha256", identities["source_archive_sha256"],
        ])
    _run([
        str(python),
        str(ROOT / "scripts" / "check_deepsolo_environment.py"),
        "--source-root", str(source),
        "--checkpoint", str(checkpoint),
        "--output", str(report),
    ] + source_identity_arguments, env=build_environment)
    frozen = _run(
        [str(python), "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        env=build_environment,
    ).stdout
    freeze_path = report.with_name(report.stem + ".pip-freeze.txt")
    freeze_path.write_text(frozen, encoding="utf-8")
    metadata_path = report.with_name(report.stem + ".setup.json")
    metadata_path.write_text(
        json.dumps({
            "schema_version": 1,
            "classification": "HARDWARE_DEVIATION",
            "description": "unprivileged non-Docker DeepSolo subprocess runtime",
            "python": version,
            "venv": str(venv),
            "cuda_home": str(cuda_home),
            "spec": str(spec),
            "frozen_inputs": identities,
            "source_type": identities["source_type"],
            "source_revision": identities["source_revision"],
            "source_revision_evidence": identities["source_revision_evidence"],
            "source_archive_sha256": identities["source_archive_sha256"],
            "checkpoint_sha256": identities["checkpoint_sha256"],
            "qualification_report": str(report),
            "pip_freeze": str(freeze_path),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
