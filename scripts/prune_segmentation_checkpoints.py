"""Keep only the manifest-selected checkpoint in completed segmentation runs.

This deliberately does not touch incomplete/qualification runs.  The default is
read-only; --apply writes an audit before unlinking each explicitly inventoried
epoch checkpoint and its checksum sidecar.  Historical per-epoch evaluation
reports remain, but deleted intermediate weights cannot be used to rerun them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml


EPOCH_FILE = re.compile(r"epoch-\d{4}\.pt")
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_regular_local(path: Path, directory: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
        raise ValueError(f"Unsafe or missing checkpoint file: {path}")


def _read_manifest(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or symlinked run manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Malformed run manifest: {path}")
    return value


def _is_completed(manifest: dict, run_dir: Path) -> bool:
    if manifest.get("status") != "SUCCESS":
        return False
    if manifest.get("training_complete") is True:
        return True
    if manifest.get("training_complete") is False:
        return False
    # Legacy author-protocol runs predate the training_complete field. Verify
    # their resolved config and full epoch count before treating them as done.
    if manifest.get("success") is not True:
        raise ValueError(f"SUCCESS run lacks completion evidence: {run_dir}")
    config_record = manifest.get("resolved_config", {})
    if not isinstance(config_record, dict) or config_record.get("path") != "resolved_config.yaml":
        raise ValueError(f"Legacy run lacks resolved-config provenance: {run_dir}")
    expected_sha = config_record.get("sha256")
    if not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
        raise ValueError(f"Legacy run lacks resolved-config SHA256: {run_dir}")
    config_path = run_dir / "resolved_config.yaml"
    _assert_regular_local(config_path, run_dir)
    if sha256_file(config_path) != expected_sha:
        raise ValueError(f"Legacy resolved config SHA256 mismatch: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records = manifest.get("epochs")
    configured_epochs = config.get("training", {}).get("epochs") if isinstance(config, dict) else None
    if (
        not isinstance(configured_epochs, int)
        or not isinstance(records, list)
        or len(records) != configured_epochs
        or not records
        or records[-1].get("epoch") != configured_epochs
    ):
        raise ValueError(f"Legacy SUCCESS run does not contain every configured epoch: {run_dir}")
    return True


def build_plan(runs_root: Path) -> dict:
    root = runs_root.resolve()
    if root.name != "context-str-runs" or not root.is_dir():
        raise ValueError("--runs-root must be the existing context-str-runs directory")
    runs = []
    for checkpoint_dir in sorted(root.rglob("checkpoints")):
        if not checkpoint_dir.is_dir():
            continue
        if checkpoint_dir.is_symlink():
            raise ValueError(f"Symlinked checkpoint directory: {checkpoint_dir}")
        run_dir = checkpoint_dir.parent
        if root not in run_dir.parents:
            raise ValueError(f"Run escapes runs root: {run_dir}")
        manifest_path = run_dir / "run_manifest.json"
        manifest = _read_manifest(manifest_path)
        files = sorted(checkpoint_dir.glob("epoch-*.pt"))
        for path in files:
            if not EPOCH_FILE.fullmatch(path.name):
                raise ValueError(f"Unexpected epoch checkpoint name: {path}")
            _assert_regular_local(path, checkpoint_dir)
        complete = _is_completed(manifest, run_dir)
        if not complete:
            runs.append({
                "run": str(run_dir.relative_to(root)),
                "status": manifest.get("status"),
                "action": "preserve_incomplete_run",
                "checkpoints": len(files),
            })
            continue
        selected = manifest.get("checkpoint", {})
        relative = selected.get("path")
        expected_sha = selected.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_sha, str) or not SHA256.fullmatch(expected_sha):
            raise ValueError(f"Completed run lacks selected checkpoint provenance: {manifest_path}")
        selected_path = run_dir / relative
        if selected_path.parent != checkpoint_dir or not EPOCH_FILE.fullmatch(selected_path.name):
            raise ValueError(f"Selected checkpoint path is not one safe epoch file: {relative}")
        _assert_regular_local(selected_path, checkpoint_dir)
        if sha256_file(selected_path) != expected_sha:
            raise ValueError(f"Selected checkpoint SHA256 mismatch: {selected_path}")
        records = manifest.get("epochs")
        if not isinstance(records, list) or not records:
            raise ValueError(f"Completed run lacks epoch records: {manifest_path}")
        record_by_path = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Malformed epoch record: {manifest_path}")
            record_path = record.get("checkpoint")
            record_sha = record.get("checkpoint_sha256")
            if not isinstance(record_path, str) or not isinstance(record_sha, str) or not SHA256.fullmatch(record_sha):
                raise ValueError(f"Epoch record lacks checkpoint provenance: {manifest_path}")
            if record_path in record_by_path:
                raise ValueError(f"Duplicate epoch checkpoint record: {record_path}")
            record_by_path[record_path] = record_sha
        if record_by_path.get(relative) != expected_sha:
            raise ValueError(f"Selected epoch record disagrees with run manifest: {manifest_path}")
        removals = []
        for path in files:
            if path == selected_path:
                continue
            record_path = f"checkpoints/{path.name}"
            recorded_sha = record_by_path.get(record_path)
            if recorded_sha is None:
                raise ValueError(f"Unrecorded epoch checkpoint; refusing to delete: {path}")
            sidecar = path.with_suffix(path.suffix + ".sha256")
            if sidecar.exists() or sidecar.is_symlink():
                _assert_regular_local(sidecar, checkpoint_dir)
                if sidecar.read_text(encoding="utf-8").strip() != f"{recorded_sha}  {path.name}":
                    raise ValueError(f"Checkpoint checksum sidecar disagrees with manifest: {sidecar}")
            removals.append({
                "path": str(path),
                "bytes": path.stat().st_size,
                "recorded_sha256": recorded_sha,
                "sidecar": str(sidecar) if sidecar.exists() else None,
            })
        runs.append({
            "run": str(run_dir.relative_to(root)),
            "status": "SUCCESS",
            "action": "keep_selected_only",
            "selected": str(selected_path),
            "selected_sha256": expected_sha,
            "run_manifest_sha256": sha256_file(manifest_path),
            "checkpoints": len(files),
            "removals": removals,
        })
    return {
        "schema_version": 1,
        "runs_root": str(root),
        "completed_runs": sum(run["action"] == "keep_selected_only" for run in runs),
        "incomplete_runs_preserved": sum(run["action"] == "preserve_incomplete_run" for run in runs),
        "checkpoints_to_remove": sum(len(run.get("removals", [])) for run in runs),
        "bytes_to_reclaim": sum(item["bytes"] for run in runs for item in run.get("removals", [])),
        "runs": runs,
    }


def _write_audit(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_plan(plan: dict) -> Path:
    root = Path(plan["runs_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    audit_path = root / f"checkpoint-pruning-{stamp}.json"
    if audit_path.exists():
        raise FileExistsError(f"Audit path already exists: {audit_path}")
    audit = {
        "schema_version": 1,
        "operation": "retain_only_manifest_selected_checkpoints_in_completed_runs",
        "created_utc": stamp,
        "plan": plan,
        "removed": [],
        "status": "IN_PROGRESS",
        "warning": "Deleted intermediate weights cannot be recovered without an external backup.",
    }
    _write_audit(audit_path, audit)
    for run in plan["runs"]:
        for item in run.get("removals", []):
            path = Path(item["path"])
            if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Checkpoint changed after preflight: {path}; audit={audit_path}")
            path.unlink()
            if item["sidecar"] is not None:
                sidecar = Path(item["sidecar"])
                if not sidecar.is_file() or sidecar.is_symlink():
                    raise ValueError(f"Sidecar changed after preflight: {sidecar}; audit={audit_path}")
                sidecar.unlink()
            audit["removed"].append(item["path"])
        if run.get("removals"):
            _write_audit(audit_path, audit)
            print(f"[prune] {run['run']}: removed {len(run['removals'])} non-selected checkpoints", flush=True)
    audit["status"] = "COMPLETE"
    _write_audit(audit_path, audit)
    return audit_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Delete only after full read-only preflight")
    args = parser.parse_args()
    plan = build_plan(args.runs_root)
    print(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "completed_runs": plan["completed_runs"],
        "incomplete_runs_preserved": plan["incomplete_runs_preserved"],
        "checkpoints_to_remove": plan["checkpoints_to_remove"],
        "bytes_to_reclaim": plan["bytes_to_reclaim"],
        "gib_to_reclaim": round(plan["bytes_to_reclaim"] / (1024 ** 3), 2),
        "selected": [run["selected"] for run in plan["runs"] if "selected" in run],
    }, indent=2), flush=True)
    if args.apply:
        audit_path = apply_plan(plan)
        print(f"AUDIT {audit_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
