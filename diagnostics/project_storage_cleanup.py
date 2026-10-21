"""Audited removal of regenerable project caches; no model/results deletion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("caption", "deepsolo", "diagnostics", "localize", "models",
               "pipeline", "recognition", "scoring", "scripts", "segmentation",
               "tests", "configs", "provenance", "docs", "manifests", "papers",
               "artifacts/diagnostics")
REPORT_DIR = REPO / "artifacts/diagnostics/storage-audit-20260914"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def disk_bytes(path):
    return int(subprocess.check_output(["du", "-s", "-B1", str(path)], text=True).split()[0])


def inventory():
    hashes = {}
    for directory in SOURCE_DIRS:
        root = REPO / directory
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "cache")
                       and not (Path(base) / d).is_symlink()
                       and Path(base) / d != REPORT_DIR]
            for name in files:
                path = Path(base) / name
                if path.is_symlink() or path.suffix in (".pyc", ".pyo"):
                    continue
                hashes[str(path.relative_to(REPO))] = sha(path)
    # Cache hard links may disappear; installed environment metadata must not change.
    for name in ("context-str-main", "context-str-main-py311", "context-str-deepsolo-host",
                 "context-str-lavis", "context-str-captioners"):
        root = REPO / ".venvs" / name
        for path in [root / "pyvenv.cfg", *root.glob("lib/python*/site-packages/*.dist-info/METADATA")]:
            if path.is_file():
                hashes[str(path.relative_to(REPO))] = sha(path)
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if not args.execute:
        ap.error("An explicit --execute is required")
    cache = REPO / ".venvs/uv-cache"
    if cache.is_symlink() or cache.resolve() != cache or not cache.is_dir():
        raise RuntimeError("Unexpected cache target; refusing cleanup")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before = disk_bytes(REPO)
    frozen = inventory()
    report = {"scope": str(REPO), "before_bytes": before,
              "cache_bytes_with_shared_links": disk_bytes(cache),
              "removed_empty_files": ["cat", "cd", "confronto"],
              "removed_bytecode": [], "status": "STARTED",
              "preserved": ["all environments", "model checkpoints", "datasets", "scientific results"]}
    def save():
        output = REPORT_DIR / "cleanup.json"
        part = output.with_suffix(".json.part")
        part.write_text(json.dumps(report, indent=2) + "\n")
        part.replace(output)
    save()
    print("[cleanup] clearing only uv's package cache; installed hard links stay valid", flush=True)
    subprocess.run([str(REPO / ".venvs/uv-0.12.5/bin/uv"), "cache", "clean",
                    "--cache-dir", str(cache)], check=True)
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=REPO, text=True).splitlines())
    for directory in SOURCE_DIRS:
        root = REPO / directory
        for base, dirs, files in os.walk(root, topdown=False, followlinks=False):
            parent = Path(base)
            if parent.name != "__pycache__" or parent.is_symlink():
                continue
            if not parent.resolve().is_relative_to(REPO):
                raise RuntimeError("Unexpected bytecode path")
            for name in files:
                path = parent / name
                relative = str(path.relative_to(REPO))
                if path.suffix in (".pyc", ".pyo") and not path.is_symlink() and relative not in tracked:
                    path.unlink()
                    report["removed_bytecode"].append(relative)
            if not any(parent.iterdir()):
                parent.rmdir()
    after_inventory = inventory()
    if frozen != after_inventory:
        report["status"] = "FAILED_SOURCE_OR_ENVIRONMENT_INTEGRITY"
        save()
        raise RuntimeError("A protected source, result or environment metadata changed during cleanup")
    report.update(status="COMPLETE", after_bytes=disk_bytes(REPO),
                  protected_files_verified=len(frozen), protected_hashes_unchanged=True)
    report["reclaimed_bytes"] = before - report["after_bytes"]
    save()
    print(f"[cleanup] reclaimed {report['reclaimed_bytes'] / 1024**3:.3f} GiB", flush=True)
    print(f"JSON {REPORT_DIR / 'cleanup.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
