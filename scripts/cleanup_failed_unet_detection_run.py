"""Remove only failed epoch-zero copies; keep a verified manifest backup.

Dry-run is the default. Completed-run pruning is a separate, explicit option
consuming a previously reviewed plan; it is never inferred from a failure.
"""

import argparse
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.prune_segmentation_checkpoints import (
    _assert_regular_local, _read_manifest, _write_audit, apply_plan, sha256_file,
)
from scripts.run_unet_word_research_v2 import load_source_state


def backup_digest(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing/unsafe manifest backup: " + str(path))
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup_failed(run_root, backup, *, apply=False):
    run_root, backup = Path(run_root), Path(backup)
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("Existing nonsymlink failed run required")
    run_root = run_root.resolve()
    branch = run_root / "synthtext_pretrain"
    if branch.is_symlink() or not branch.is_dir():
        raise ValueError("Unsafe/missing synthetic branch")
    with ExitStack() as stack:
        for name in ("suite.lock", "runner.lock"):
            path = run_root / name
            _assert_regular_local(path, run_root)
            lock = stack.enter_context(path.open("rb"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit_path = run_root / "failed_run_cleanup.json"
        digest = backup_digest(backup)
        if audit_path.exists():
            audit = _read_manifest(audit_path)
            if (audit.get("status") != "COMPLETE" or audit.get("manifest_sha256") != digest
                    or any((run_root / p).exists() for p in audit["removed"])):
                raise ValueError("Cleanup audit/files differ; inspect before retrying")
            return {"status": "ALREADY_CLEANED", "removed_bytes": audit["removed_bytes"]}
        failure = _read_manifest(branch / "failure.json")
        if failure.get("status") != "FAILED" or failure.get("action") != "pretrain":
            raise ValueError("Only a failed synthetic pretraining run is eligible")
        if (branch / "run_manifest.json").exists():
            raise ValueError("Published/partial training receipt must be preserved")
        history = branch / "epochs.jsonl"
        if history.exists() and (history.is_symlink() or history.read_text().strip()):
            raise ValueError("Completed epoch records must be preserved")
        contract = _read_manifest(run_root / "run_contract.json")
        source = load_source_state(Path(contract["v1_root"]) / "control", contract["resolved_config"])
        inventory = []
        for name in ("last.pt", "best.pt"):
            path = branch / name
            _assert_regular_local(path, branch)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            state = saved["model_state_dict"]
            if (saved.get("epoch") != 0 or saved.get("global_step") != 0
                    or saved.get("train_metrics") or set(state) != set(source)
                    or any(state[k].dtype != source[k].dtype or state[k].shape != source[k].shape
                           or not torch.equal(state[k], source[k]) for k in source)):
                raise ValueError("Checkpoint contains trained/different weights: " + str(path))
            inventory.append({"relative_path": str(path.relative_to(run_root)),
                              "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = run_root / "synthtext_manifest.json"
        _assert_regular_local(manifest, run_root)
        if sha256_file(manifest) != digest:
            raise ValueError("Compressed backup does not reproduce the original manifest")
        inventory.append({"relative_path": manifest.name, "bytes": manifest.stat().st_size,
                          "sha256": digest})
        total = sum(p["bytes"] for p in inventory)
        result = {"status": "READY", "apply": apply, "removed_bytes": total,
                  "backup_bytes": backup.stat().st_size, "inventory": inventory}
        if not apply:
            return result
        audit = {**result, "status": "IN_PROGRESS", "manifest_backup": str(backup.resolve()),
                 "manifest_sha256": digest, "removed": []}
        _write_audit(audit_path, audit)
        for item in inventory:
            path = run_root / item["relative_path"]
            _assert_regular_local(path, path.parent)
            if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
                raise ValueError("File changed after cleanup preflight: " + str(path))
            path.unlink()
            audit["removed"].append(item["relative_path"])
            _write_audit(audit_path, audit)
        audit["status"] = "COMPLETE"
        _write_audit(audit_path, audit)
        return {**result, "status": "COMPLETE", "audit": str(audit_path)}


def reviewed_completed_plan(path, *, apply=False):
    plan = _read_manifest(Path(path))
    root = Path(plan["runs_root"]).resolve()
    if len(plan["runs"]) != 1 or plan["runs"][0]["action"] != "keep_selected_and_last":
        raise ValueError("Only the separately reviewed single-run plan is accepted")
    run = plan["runs"][0]
    directory = root / run["run"]
    if directory.is_symlink() or root not in directory.resolve().parents:
        raise ValueError("Completed run escapes the plan root")
    if sha256_file(directory / "run_manifest.json") != run["run_manifest_sha256"]:
        raise ValueError("Completed run manifest changed since review")
    receipt = _read_manifest(directory / "run_manifest.json")
    if receipt.get("status") != "SUCCESS" or receipt.get("training_complete") is not True:
        raise ValueError("Completed run no longer qualified")
    protected = [{"path": run["selected"], "sha256": run["selected_sha256"]},
                 *run["additional_protected"]]
    if len(protected) != 2:
        raise ValueError("Selected and last checkpoints must both be preserved")
    protected_paths = {Path(p["path"]) for p in protected}
    records = {str(directory / row["checkpoint"]): row["checkpoint_sha256"]
               for row in receipt["epochs"]}
    selected = directory / receipt["checkpoint"]["path"]
    last = directory / max(receipt["epochs"], key=lambda row: row["epoch"])["checkpoint"]
    if protected_paths != {selected, last}:
        raise ValueError("The actual selected and last checkpoints must be protected")
    for item in protected:
        p = Path(item["path"])
        _assert_regular_local(p, directory / "checkpoints")
        if item["sha256"] != records.get(str(p)) or sha256_file(p) != item["sha256"]:
            raise ValueError("Protected checkpoint differs from reviewed SHA")
    if (plan["checkpoints_to_remove"] != len(run["removals"])
            or plan["bytes_to_reclaim"] != sum(p["bytes"] for p in run["removals"])):
        raise ValueError("Reviewed removal counts differ")
    for item in run["removals"]:
        p = Path(item["path"])
        _assert_regular_local(p, directory / "checkpoints")
        if (p in protected_paths or p.stat().st_size != item["bytes"]
                or records.get(str(p)) != item["recorded_sha256"]
                or p.name not in {Path(row["checkpoint"]).name for row in receipt["epochs"]}):
            raise ValueError("Removal overlaps a protected/changed checkpoint")
        sidecar = Path(item["sidecar"]) if item["sidecar"] else None
        if sidecar:
            if sidecar != p.with_suffix(p.suffix + ".sha256"):
                raise ValueError("Removal sidecar is not the checkpoint sidecar")
            _assert_regular_local(sidecar, p.parent)
            if sidecar.read_text().strip() != f"{item['recorded_sha256']}  {p.name}":
                raise ValueError("Checkpoint sidecar changed since review")
    result = {"apply": apply, "checkpoints_to_remove": plan["checkpoints_to_remove"],
              "bytes_to_reclaim": plan["bytes_to_reclaim"], "protected": sorted(map(str, protected_paths))}
    if apply:
        result["audit"] = str(apply_plan(plan))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-run", type=Path, required=True)
    parser.add_argument("--manifest-backup", type=Path, required=True)
    parser.add_argument("--completed-plan", type=Path,
                        help="Separate explicit approval required; never auto-selected")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    # Complete every read-only preflight before starting either mutation.
    failed = cleanup_failed(args.failed_run, args.manifest_backup)
    complete = reviewed_completed_plan(args.completed_plan) if args.completed_plan else None
    print(json.dumps({"failed": failed, "completed": complete}, indent=2), flush=True)
    if args.apply:
        print(json.dumps(cleanup_failed(args.failed_run, args.manifest_backup, apply=True), indent=2))
        if args.completed_plan:
            print(json.dumps(reviewed_completed_plan(args.completed_plan, apply=True), indent=2))


if __name__ == "__main__":
    main()
