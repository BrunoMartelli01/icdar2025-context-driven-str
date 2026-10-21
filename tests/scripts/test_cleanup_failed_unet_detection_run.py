import gzip
import json
from pathlib import Path

import pytest
import torch

from scripts.cleanup_failed_unet_detection_run import cleanup_failed, reviewed_completed_plan
from scripts.prune_segmentation_checkpoints import sha256_file


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    root = tmp_path / "failed"
    branch = root / "synthtext_pretrain"
    branch.mkdir(parents=True)
    for name in ("suite.lock", "runner.lock"):
        (root / name).touch()
    (branch / "failure.json").write_text(json.dumps({"status": "FAILED", "action": "pretrain"}))
    (root / "run_contract.json").write_text(json.dumps({"v1_root": str(tmp_path / "source"), "resolved_config": {}}))
    source = {"weight": torch.tensor([1., 2.])}
    monkeypatch.setattr("scripts.cleanup_failed_unet_detection_run.load_source_state", lambda *_: source)
    payload = {"epoch": 0, "global_step": 0, "train_metrics": {}, "model_state_dict": source}
    for name in ("last.pt", "best.pt"):
        torch.save(payload, branch / name)
    manifest = root / "synthtext_manifest.json"
    manifest.write_text('{"original_annotations": "retained"}\n')
    backup = tmp_path / "manifest.json.gz"
    with gzip.open(backup, "wb") as stream:
        stream.write(manifest.read_bytes())
    return root, backup


def test_failed_cleanup_dry_run_apply_preserves_evidence_and_is_idempotent(failed_run):
    root, backup = failed_run
    original = (root / "synthtext_manifest.json").read_bytes()
    plan = cleanup_failed(root, backup)
    assert plan["status"] == "READY" and plan["removed_bytes"] > 0
    assert (root / "synthtext_pretrain/best.pt").exists()
    result = cleanup_failed(root, backup, apply=True)
    assert result["status"] == "COMPLETE"
    assert not (root / "synthtext_pretrain/best.pt").exists()
    assert (root / "synthtext_pretrain/failure.json").exists()
    assert (root / "run_contract.json").exists()
    with gzip.open(backup, "rb") as stream:
        assert stream.read() == original
    assert cleanup_failed(root, backup, apply=True)["status"] == "ALREADY_CLEANED"


@pytest.mark.parametrize("change", ["epoch", "step", "weights", "dtype", "history", "published", "backup", "symlink"])
def test_failed_cleanup_refuses_useful_changed_or_unsafe_artifacts(failed_run, change):
    root, backup = failed_run
    branch = root / "synthtext_pretrain"
    checkpoint = branch / "last.pt"
    saved = torch.load(checkpoint, weights_only=False)
    if change == "epoch": saved["epoch"] = 1
    if change == "step": saved["global_step"] = 1
    if change == "weights": saved["model_state_dict"]["weight"] += 1
    if change == "dtype": saved["model_state_dict"]["weight"] = saved["model_state_dict"]["weight"].double()
    torch.save(saved, checkpoint)
    if change == "history": (branch / "epochs.jsonl").write_text('{"epoch":1}\n')
    if change == "published": (branch / "run_manifest.json").write_text('{"status":"SUCCESS"}')
    if change == "backup":
        with gzip.open(backup, "wb") as stream: stream.write(b"changed")
    if change == "symlink":
        moved = branch / "other.pt"
        checkpoint.rename(moved)
        checkpoint.symlink_to(moved)
    with pytest.raises(ValueError): cleanup_failed(root, backup, apply=True)
    assert (branch / "best.pt").exists() and (root / "synthtext_manifest.json").exists()


def completed_plan(tmp_path):
    root = tmp_path / "context-str-runs"
    run = root / "complete"
    directory = run / "checkpoints"
    directory.mkdir(parents=True)
    paths = [directory / f"epoch-{i:04d}.pt" for i in (1, 2, 3)]
    for p in paths:
        p.write_bytes(p.name.encode())
        p.with_suffix(".pt.sha256").write_text(f"{sha256_file(p)}  {p.name}\n")
    receipt = run / "run_manifest.json"
    receipt.write_text(json.dumps({"status": "SUCCESS", "training_complete": True,
        "checkpoint": {"path": "checkpoints/epoch-0002.pt"},
        "epochs": [{"epoch": i, "checkpoint": "checkpoints/" + p.name,
                    "checkpoint_sha256": sha256_file(p)} for i, p in enumerate(paths, 1)]}))
    plan = {"runs_root": str(root), "checkpoints_to_remove": 1, "bytes_to_reclaim": paths[0].stat().st_size,
        "runs": [{"run": "complete", "action": "keep_selected_and_last", "selected": str(paths[1]),
          "selected_sha256": sha256_file(paths[1]), "run_manifest_sha256": sha256_file(receipt),
          "additional_protected": [{"path": str(paths[2]), "sha256": sha256_file(paths[2])}],
          "removals": [{"path": str(paths[0]), "bytes": paths[0].stat().st_size,
            "recorded_sha256": sha256_file(paths[0]), "sidecar": str(paths[0].with_suffix(".pt.sha256"))}]}]}
    file = tmp_path / "plan.json"
    file.write_text(json.dumps(plan))
    return file, paths, plan


def test_completed_plan_is_explicit_and_preserves_selected_and_last(tmp_path):
    file, paths, _ = completed_plan(tmp_path)
    reviewed_completed_plan(file)
    assert all(p.exists() for p in paths)
    reviewed_completed_plan(file, apply=True)
    assert not paths[0].exists() and paths[1].exists() and paths[2].exists()


@pytest.mark.parametrize("change", ["protect_wrong", "remove_selected", "sidecar", "count", "manifest"])
def test_completed_plan_tampering_fails_before_removing_anything(tmp_path, change):
    file, paths, plan = completed_plan(tmp_path)
    run = plan["runs"][0]
    if change == "protect_wrong":
        run["additional_protected"] = [{"path": str(paths[0]), "sha256": sha256_file(paths[0])}]
    if change == "remove_selected":
        run["removals"][0] = {**run["removals"][0], "path": str(paths[1]),
            "recorded_sha256": sha256_file(paths[1]), "sidecar": str(paths[1].with_suffix(".pt.sha256"))}
    if change == "sidecar": run["removals"][0]["sidecar"] = str(paths[2].with_suffix(".pt.sha256"))
    if change == "count": plan["checkpoints_to_remove"] += 1
    if change == "manifest": (paths[0].parent.parent / "run_manifest.json").write_text('{}')
    file.write_text(json.dumps(plan))
    with pytest.raises(ValueError): reviewed_completed_plan(file, apply=True)
    assert all(p.exists() for p in paths)
