"""Check or export a source-only collaboration tree (Python 3.8+, Git).

The inventory subcommand additionally uses the main environment's PyYAML.
No command installs packages, downloads assets, runs a model, or modifies Git's
index/remotes/history. Export destinations must be new and outside the source.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {
    ".github", "Figures", "caption", "cli", "configs", "deepsolo", "diagnostics",
    "docker", "docs", "environments", "localize", "manifests", "models", "papers",
    "pipeline", "provenance", "recognition", "research", "scoring", "scripts",
    "segmentation", "tests",
}
ROOT_FILES = {
    ".gitattributes", ".gitignore", ".rgignore", "AGENTS.md", "README.md",
    "CONTRIBUTING.md", "requirements.txt", "LICENSE", "CITATION.cff",
    "COLLABORATION_EXPORT.json",
}
REQUIRED = (
    "AGENTS.md", "README.md", "CONTRIBUTING.md", ".gitignore",
    "docs/GETTING_STARTED.md", "docs/REPOSITORY_MAP.md", "docs/EXTERNAL_ASSETS.md",
    "provenance/EXTERNAL_ASSETS.json", "research/experiments.json",
    "docs/AUDIT_INDEX.json", "docs/AUDIT_INDEX.md",
)
FORBIDDEN_DIRS = {
    ".git", ".aws", ".ssh", ".venvs", ".venv", "venv", "__pycache__",
    ".pytest_cache", ".cache", "node_modules", "build", "dist",
}
FORBIDDEN_ROOTS = {"artifacts", "datasets", "data", "runs", "outputs", "weights",
                   "checkpoints", "cache", "external", "DeepSolo"}
FORBIDDEN_SUFFIXES = {
    ".pdf", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".so",
    ".npy", ".npz", ".pkl", ".pickle", ".h5", ".hdf5", ".zip", ".tar",
    ".gz", ".tgz", ".part", ".pyc", ".pem", ".key",
}
TEXT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".sh", ".txt", ".in", ".example", ".cff"}
MAX_SOURCE_BYTES = 2 * 1024 * 1024  # Collaboration policy, not GitHub's limit.
SECRET_PATTERNS = (
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)


class SourceError(ValueError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args, required=True):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if required and result.returncode:
        raise SourceError("Git failed: " + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def visible_files(root):
    raw = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    return sorted({p.decode("utf-8") for p in raw.split(b"\0") if p})


def in_source_allowlist(relative):
    parts = Path(relative).parts
    return bool(parts) and (relative in ROOT_FILES or parts[0] in SOURCE_DIRS)


def forbidden_reason(relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return "path escapes repository"
    if (path.parts and path.parts[0] in FORBIDDEN_ROOTS) or any(
        part in FORBIDDEN_DIRS for part in path.parts[:-1]
    ):
        return "external/generated/credential directory"
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return "external artifact or secret file type"
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return "populated local environment file"
    if any(path.name.endswith(suffix) for suffix in (".local.json", ".local.yaml", ".local.yml")):
        return "local path configuration"
    return None


def check_source(root):
    files = visible_files(root)
    errors = []
    total = 0
    for name in files:
        path = root / name
        reason = forbidden_reason(name)
        if reason:
            errors.append(name + ": " + reason)
            continue
        if not in_source_allowlist(name):
            errors.append(name + ": outside the documented source allowlist")
            continue
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            errors.append(name + ": symlinks are not exported")
            continue
        if not path.is_file():
            errors.append(name + ": missing/non-regular Git-visible file")
            continue
        size = path.stat().st_size
        total += size
        if size > MAX_SOURCE_BYTES:
            errors.append(name + ": exceeds 2 MiB source policy; use external provenance")
            continue
        raw = path.read_bytes()
        if path.suffix.lower() == ".png" and (
            name.startswith("Figures/") or name.startswith("tests/fixtures/")
        ):
            if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(name + ": invalid declared PNG fixture/illustration")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and name not in ROOT_FILES and path.name != "Dockerfile":
            errors.append(name + ": unsupported source format")
            continue
        if any(pattern.search(raw) for pattern in SECRET_PATTERNS):
            # Do not disclose the matched credential in output.
            errors.append(name + ": possible credential/private key; inspect locally")
            continue
        try:
            decoded = raw.decode("utf-8")
            if "\0" in decoded:
                raise ValueError("NUL in source text")
            if path.suffix == ".json":
                json.loads(decoded)
        except (UnicodeError, ValueError) as exc:
            errors.append(name + ": invalid text/JSON (" + type(exc).__name__ + ")")
    for name in REQUIRED:
        if name not in files:
            errors.append(name + ": required onboarding source is missing or ignored")
    if errors:
        raise SourceError("Source validation failed:\n" + "\n".join(errors))
    return files, total


def export_source(root, destination):
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise SourceError("Export destination must be outside the source repository")
    if destination.exists():
        raise SourceError("Export destination already exists; choose a new directory")
    files, total = check_source(root)
    files = [name for name in files if name != "COLLABORATION_EXPORT.json"]
    destination.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in files:
        original, target = root / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = original.read_bytes()
        source_hashes[name] = digest(raw)
        target.write_bytes(raw)
        target.chmod(0o755 if original.stat().st_mode & 0o111 else 0o644)
    environment = dict(os.environ)
    environment.pop("CONTEXT_STR_RUNS_ROOT", None)
    environment.pop("CONTEXT_STR_ARTIFACTS_ROOT", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    catalog = subprocess.run(
        [sys.executable, "-m", "research", "catalog"], cwd=str(destination),
        env=environment, capture_output=True, text=True,
    )
    if catalog.returncode:
        raise SourceError("Export retained for inspection; catalog failed: " + catalog.stderr.strip())
    records = []
    for name in files:
        target = destination / name
        raw = target.read_bytes()
        records.append({"path": name, "bytes": len(raw), "sha256": digest(raw),
                        "source_sha256": source_hashes[name],
                        "executable": bool(target.stat().st_mode & 0o111)})
    manifest = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Initial source-only export; describes export time, not later commits",
        "source_state": "Includes pre-existing uncommitted local research source; only reported focused tests are qualified",
        "source_git_base": git(root, "rev-parse", "--verify", "HEAD", required=False).decode().strip() or None,
        "upstream_repository": "https://github.com/Ritabrata04/icdar2025-context-driven-str",
        "source_file_count": len(records),
        "exported_bytes": sum(item["bytes"] for item in records),
        "regenerated_views": ["docs/AUDIT_INDEX.json", "docs/AUDIT_INDEX.md"],
        "excluded_asset_inventory": "provenance/EXTERNAL_ASSETS.json",
        "scientific_validation": "No training/inference; no result or artifact promoted",
        "files": records,
    }
    (destination / "COLLABORATION_EXPORT.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"destination": str(destination), "files": len(records), "source_bytes": total,
            "exported_bytes": manifest["exported_bytes"], "catalog": catalog.stdout.strip()}


def artifact_families(root):
    families = {}
    base = root / "artifacts"
    if not base.is_dir():
        return []
    for directory, names, filenames in os.walk(str(base), followlinks=False):
        names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        for filename in filenames:
            path = Path(directory) / filename
            if path.is_symlink():
                continue
            parts = path.relative_to(base).parts
            depth = 2 if parts[0] in {"diagnostics", "training", "exports"} else 1
            key = "artifacts/" + "/".join(parts[:min(depth, len(parts) - 1)])
            row = families.setdefault(key, {"path": key, "files": 0, "bytes": 0,
                "recovery": "maintainer run pack or new producer run; exact replay unresolved without its inputs",
                "procedure": "docs/EXTERNAL_ASSETS.md#project-checkpoints-and-generated-experiment-outputs"})
            row["files"] += 1
            row["bytes"] += path.stat().st_size
    return [families[key] for key in sorted(families)]


def write_inventory(root, output):
    try:
        import yaml
    except ImportError as exc:
        raise SourceError("Inventory generation needs PyYAML from the existing main environment") from exc
    source_path = root / "provenance/SOURCES.yaml"
    artifact_path = root / "provenance/ARTIFACTS.yaml"
    registry_path = root / "research/experiments.json"
    sources = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    artifacts = yaml.safe_load(artifact_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = set(visible_files(root))
    comparisons = {}
    for group in registry["groups"]:
        source = group["source"]
        key = (source["root"], source["path"], source["sha256"])
        row = comparisons.setdefault(key, {**source, "groups": [],
            "included_in_source_export": (source["root"] == "repository"
                and source["path"] in selected and in_source_allowlist(source["path"])
                and forbidden_reason(source["path"]) is None),
            "recovery": "docs/EXTERNAL_ASSETS.md#restore-saved-comparisons"})
        row["groups"].append(group["id"])
    payload = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Dated workspace inventory; local files are not a public archive",
        "scientific_status": "Original classifications/statuses copied without promotion",
        "canonical_inputs": {str(p.relative_to(root)): digest(p.read_bytes())
                             for p in (source_path, artifact_path, registry_path)},
        "sources": sources,
        "artifacts": artifacts,
        "comparison_inputs": [comparisons[key] for key in sorted(comparisons)],
        "excluded_workspace_families": artifact_families(root),
        "other_exclusions": {
            "environments": "docs/GETTING_STARTED.md#2-create-an-isolated-main-environment-when-needed",
            "datasets_weights_caches": "docs/EXTERNAL_ASSETS.md",
            "external_host_roots": "docs/HOST_OPERATIONS.md",
            "historical_external_evidence": "docs/history/WORKSPACE_AUDIT_INDEX.json",
            "credentials": "Each contributor supplies their own; never share or archive",
        },
        "known_recovery_gaps": [
            "No public historical run/checkpoint archive recorded",
            "Original M6 Docker base construction history incomplete",
            "Unresolved strict artifact identities/protocols remain blockers",
            "Temporary historical paths may no longer exist; no dummy reconstruction",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"output": str(output), "comparison_inputs": len(comparisons),
            "excluded_families": len(payload["excluded_workspace_families"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Validate Git-visible source; no installs/models")
    export = commands.add_parser("export", help="Write a NEW source-only copy outside this repository")
    export.add_argument("--destination", type=Path, required=True)
    inventory = commands.add_parser("inventory", help="Record external assets using existing PyYAML")
    inventory.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = args.repo_root.resolve()
        if args.command == "check":
            files, size = check_source(root)
            result = {"status": "PASS", "source_files": len(files), "bytes": size,
                      "scope": "source export policy; no scientific qualification"}
        elif args.command == "export":
            result = export_source(root, args.destination)
        else:
            result = write_inventory(root, args.output.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (SourceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
