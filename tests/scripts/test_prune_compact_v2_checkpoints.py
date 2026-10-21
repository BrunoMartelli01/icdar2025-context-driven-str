import hashlib
import json

import pytest

from scripts.prune_compact_v2_checkpoints import apply, preflight


def setup_plan(tmp_path):
    root = tmp_path / "total-text-tighter-words-v2-500ep"
    directory = root / "checkpoints"
    directory.mkdir(parents=True)
    meta = root / "epochs.jsonl"
    meta.write_text("frozen metadata")
    p = directory / "epoch-0001.pt"
    p.write_bytes(b"unused weights")
    plan = {"operation": "delete_old_compact_v2_checkpoints_only", "keep_weights": [],
        "run_directory": str(root), "preserve_metadata_sha256": {"epochs.jsonl": hashlib.sha256(meta.read_bytes()).hexdigest()},
        "remove": [{"filename": p.name, "kind": "weights", "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}]}
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(plan))
    return pp, p, meta


def test_read_only_inventory_validates_without_unlinking(tmp_path):
    plan, p, meta = setup_plan(tmp_path)
    _, pending, absent = preflight(plan, check_processes=False)
    assert len(pending) == 1 and not absent and p.exists() and meta.exists()


def test_new_checkpoint_blocks_the_whole_deletion_plan(tmp_path):
    plan, p, _ = setup_plan(tmp_path)
    (p.parent / "epoch-0002.pt").write_bytes(b"new")
    with pytest.raises(ValueError, match="New/unreviewed"):
        preflight(plan, check_processes=False)
    assert p.exists()


def test_changed_training_metadata_blocks_deletion(tmp_path):
    plan, p, meta = setup_plan(tmp_path)
    meta.write_text("training continued")
    with pytest.raises(ValueError, match="metadata changed"):
        preflight(plan, check_processes=False)
    assert p.exists()


def test_symlink_target_blocks_deletion(tmp_path):
    plan, p, _ = setup_plan(tmp_path)
    target = tmp_path / "external.pt"
    target.write_bytes(p.read_bytes())
    p.unlink()
    p.symlink_to(target)
    with pytest.raises(ValueError, match="Symlinked"):
        preflight(plan, check_processes=False)
    assert target.exists()


def test_apply_removes_only_reviewed_weight_and_preserves_run_metadata(tmp_path, monkeypatch):
    plan, p, meta = setup_plan(tmp_path)
    monkeypatch.setattr("scripts.prune_compact_v2_checkpoints.active_old_run", lambda root: None)
    audit = tmp_path / "audit.json"
    apply(plan, audit)
    report = json.loads(audit.read_text())
    assert report["status"] == "COMPLETE" and report["gt_deleted"] is False
    assert report["removed"] == [p.name] and not p.exists()
    assert meta.read_text() == "frozen metadata"
    _, pending, absent = preflight(plan, check_processes=False)
    assert not pending and absent == [p.name]
