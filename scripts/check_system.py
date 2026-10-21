#!/usr/bin/env python3
"""Report physical-host prerequisites for the reproduction environments.

This script intentionally has no third-party dependencies.  In particular, the
CUDA value printed by ``nvidia-smi`` is reported as driver compatibility and is
not treated as evidence that a CUDA Toolkit or ``nvcc`` is installed.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _meminfo() -> dict[str, int | None]:
    values: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"([^:]+):\s+(\d+)\s+kB", line)
            if match:
                values[match.group(1)] = int(match.group(2)) * 1024
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
    }


def _cpu_model() -> str | None:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _python_executables() -> list[dict[str, str]]:
    candidates: set[Path] = set()
    for name in ("python", "python3", "python3.8", "python3.9", "python3.10", "python3.11", "python3.12", "python3.13"):
        resolved = shutil.which(name)
        if resolved:
            candidates.add(Path(resolved).resolve())
    for directory in (Path("/usr/bin"), Path("/usr/local/bin")):
        if directory.is_dir():
            candidates.update(path.resolve() for path in directory.glob("python3.*") if path.is_file())

    found: list[dict[str, str]] = []
    for executable in sorted(candidates, key=str):
        version = _run([str(executable), "--version"])
        if version:
            found.append({"path": str(executable), "version": version})
    return found


def _nvidia() -> dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "nvidia_smi_available": smi is not None,
        "nvidia_smi_path": smi,
        "driver_cuda_compatibility": None,
        "gpus": [],
    }
    if not smi:
        return result

    header = _run([smi]) or ""
    match = re.search(r"CUDA Version:\s*([0-9.]+)", header)
    if match:
        result["driver_cuda_compatibility"] = match.group(1)

    query = _run(
        [
            smi,
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if query:
        for line in query.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                result["gpus"].append(
                    {
                        "model": fields[0],
                        "total_vram_mib": int(fields[1]),
                        "driver_version": fields[2],
                    }
                )
    return result


def _cuda_toolkit() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    conventional_roots = [Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))]
    existing_roots = sorted({str(path.resolve()) for path in conventional_roots if path.exists()})
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if "cuda" in entry.lower()]
    nvcc_version = _run([nvcc, "--version"]) if nvcc else None

    # Toolkit evidence requires nvcc or a conventional toolkit root.  The
    # nvidia-smi compatibility number is deliberately not consulted here.
    return {
        "nvcc_available": nvcc is not None,
        "nvcc_path": nvcc,
        "nvcc_version": nvcc_version,
        "cuda_home": cuda_home,
        "cuda_path_entries": path_entries,
        "conventional_toolkit_roots": existing_roots,
        "system_cuda_toolkit_detected": bool(nvcc or existing_roots),
    }


def collect() -> dict[str, Any]:
    os_release = _os_release()
    disk = shutil.disk_usage(Path.cwd())
    return {
        "operating_system": {
            "distribution": os_release.get("NAME"),
            "version": os_release.get("VERSION_ID"),
            "pretty_name": os_release.get("PRETTY_NAME"),
            "kernel": platform.release(),
        },
        "cpu": {"model": _cpu_model(), "architecture": platform.machine()},
        "memory": _meminfo(),
        "disk_for_current_filesystem": {
            "path": str(Path.cwd()),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "python_executables": _python_executables(),
        "git_version": _run(["git", "--version"]),
        "nvidia": _nvidia(),
        "cuda_toolkit": _cuda_toolkit(),
        "interpretation": {
            "nvidia_smi_cuda_field": "Maximum CUDA runtime compatibility exposed by the installed driver; not proof of an installed CUDA Toolkit.",
            "system_cuda_toolkit_rule": "Detected only from nvcc or a conventional CUDA toolkit root.",
        },
    }


def main() -> int:
    print(json.dumps(collect(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
