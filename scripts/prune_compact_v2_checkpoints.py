"""Remove only inventoried old compact-v2 weights; preserve masks and run metadata.

Read-only by default. --apply requires the frozen inventory and a new audit
outside the source run. Stops on any new/changed checkpoint or active old run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def active_old_run(root):
    mine = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == mine:
            continue
        try:
            if p.stat().st_uid != os.getuid():
                continue
            argv = (p / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except FileNotFoundError:
            continue  # process terminated between directory enumeration and read
        train = any(v == "segmentation.train" for v in argv)
        launcher = any(Path(v).name == "run_totaltext_tighter_words_train_and_parseq_host.sh" for v in argv if v)
        target = any(str(root) in v or str(root.parent) in v or "segmentation_total_text_tighter_words_v2.yaml" in v for v in argv)
        if launcher or (train and target):
            raise RuntimeError(f"Old compact-v2 training is active: PID{p.name}; no deletion performed")


def preflight(plan_path, *, check_processes=True):
    plan = json.loads(plan_path.read_text())
    if plan["operation"] != "delete_old_compact_v2_checkpoints_only" or plan["keep_weights"] != []:
        raise ValueError("Explicit old-v2 weights-only deletion inventory required")
    root = Path(plan["run_directory"])
    if root.is_symlink() or root.name != "total-text-tighter-words-v2-500ep":
        raise ValueError("Unexpected source v2 run")
    directory = root / "checkpoints"
    if directory.is_symlink():
        raise ValueError("Checkpoint directory is a symlink")
    for name, digest in plan["preserve_metadata_sha256"].items():
        path = root / name
        if path.parent != root or sha(path) != digest:
            raise ValueError("Original training metadata changed")
    if check_processes:
        active_old_run(root)
    entries = plan["remove"]
    allowed = set()
    pending, absent = [], []
    for row in entries:
        name = row["filename"]
        if not re.fullmatch(r"epoch-\d{4}\.pt(?:\.sha256|\.tmp)?", name) or name in allowed:
            raise ValueError("Unsafe/duplicate deletion path")
        allowed.add(name)
        path = directory / name
        if path.is_symlink():
            raise ValueError("Symlinked deletion target")
        if not path.exists():
            absent.append(name)
            continue
        st = path.stat()
        if not path.is_file() or st.st_size != row["bytes"] or st.st_mtime_ns != row["mtime_ns"]:
            raise ValueError("Inventoried checkpoint changed: " + name)
        if row["kind"] == "sha256_sidecar" and sha(path) != row["sha256"]:
            raise ValueError("Inventoried checkpoint sidecar changed")
        pending.append(row)
    unknown = {p.name for p in directory.iterdir()} - allowed
    if unknown:
        raise ValueError("New/unreviewed checkpoint files: " + ", ".join(sorted(unknown)))
    return plan, pending, absent


def apply(plan_path, audit_path):
    plan, pending, absent = preflight(plan_path)
    root = Path(plan["run_directory"])
    if audit_path.exists() or audit_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a new audit outside the old run")
    audit = {"operation": plan["operation"], "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha(plan_path), "source_run": str(root), "removed": [],
        "already_absent": absent, "reclaimed_logical_bytes": 0, "status": "IN_PROGRESS",
        "preserved_metadata": plan["preserve_metadata_sha256"], "gt_deleted": False}
    write(audit_path, audit)
    try:
        for row in pending:
            for name, digest in plan["preserve_metadata_sha256"].items():
                if sha(root / name) != digest:
                    raise ValueError("Old training metadata changed during deletion")
            path = root / "checkpoints" / row["filename"]
            if path.is_symlink() or path.stat().st_mtime_ns != row["mtime_ns"] or path.stat().st_size != row["bytes"]:
                raise ValueError("Checkpoint changed after preflight")
            path.unlink()
            audit["removed"].append(row["filename"])
            audit["reclaimed_logical_bytes"] += row["bytes"]
            write(audit_path, audit)
        audit["status"] = "COMPLETE"
    except Exception as error:
        audit.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        write(audit_path, audit)
        raise
    write(audit_path, audit)
    print(json.dumps({k: audit[k] for k in ("status", "reclaimed_logical_bytes", "gt_deleted")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--audit", type=Path)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    if args.apply:
        if args.audit is None:
            p.error("--apply requires --audit")
        apply(args.plan, args.audit)
    else:
        plan, pending, absent = preflight(args.plan)
        print(json.dumps({"status": "PREFLIGHT_PASS", "pending_files": len(pending),
            "already_absent": len(absent), "reclaimable_logical_bytes": sum(r["bytes"] for r in pending),
            "gt_deleted": False, "weights_deleted": 0}))
