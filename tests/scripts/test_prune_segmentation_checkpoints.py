"""Safety checks for completed-run checkpoint pruning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prune_segmentation_checkpoints import apply_plan, build_plan, sha256_file


def _run(root: Path, name: str, *, complete: bool) -> tuple[Path, Path]:
    run = root / name
    directory = run / "checkpoints"
    directory.mkdir(parents=True)
    first = directory / "epoch-0001.pt"
    second = directory / "epoch-0002.pt"
    first.write_bytes(b"first-weights")
    second.write_bytes(b"best-weights")
    records = []
    for epoch, path in enumerate((first, second), 1):
        digest = sha256_file(path)
        path.with_suffix(".pt.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        records.append({"epoch": epoch, "checkpoint": f"checkpoints/{path.name}", "checkpoint_sha256": digest})
    (run / "run_manifest.json").write_text(json.dumps({
        "status": "SUCCESS" if complete else "INCOMPLETE",
        "training_complete": complete,
        "checkpoint": {"path": "checkpoints/epoch-0002.pt", "sha256": sha256_file(second)},
        "epochs": records,
    }), encoding="utf-8")
    return first, second


def test_dry_run_and_apply_keep_only_selected(tmp_path: Path) -> None:
    root = tmp_path / "context-str-runs"
    first, selected = _run(root, "complete", complete=True)
    incomplete_first, incomplete_selected = _run(root, "incomplete", complete=False)
    plan = build_plan(root)
    assert plan["checkpoints_to_remove"] == 1
    assert plan["completed_runs"] == 1
    assert plan["incomplete_runs_preserved"] == 1
    assert first.exists()  # build_plan is read-only
    audit = apply_plan(plan)
    assert json.loads(audit.read_text(encoding="utf-8"))["status"] == "COMPLETE"
    assert not first.exists()
    assert not first.with_suffix(".pt.sha256").exists()
    assert selected.exists() and selected.with_suffix(".pt.sha256").exists()
    assert incomplete_first.exists() and incomplete_selected.exists()
    assert build_plan(root)["checkpoints_to_remove"] == 0


def test_selected_checkpoint_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "context-str-runs"
    first, selected = _run(root, "complete", complete=True)
    selected.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_plan(root)
    assert first.exists()


def test_unrecorded_epoch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "context-str-runs"
    first, selected = _run(root, "complete", complete=True)
    (first.parent / "epoch-0003.pt").write_bytes(b"unrecorded")
    with pytest.raises(ValueError, match="Unrecorded epoch checkpoint"):
        build_plan(root)
    assert first.exists() and selected.exists()


def test_symlink_checkpoint_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "context-str-runs"
    first, selected = _run(root, "complete", complete=True)
    first.unlink()
    first.symlink_to(selected)
    with pytest.raises(ValueError, match="Unsafe or missing checkpoint"):
        build_plan(root)


def test_refuse_broad_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context-str-runs"):
        build_plan(tmp_path)


def test_legacy_success_requires_full_configured_epoch_count(tmp_path: Path) -> None:
    root = tmp_path / "context-str-runs"
    first, selected = _run(root, "legacy", complete=True)
    run = selected.parent.parent
    config_path = run / "resolved_config.yaml"
    config_path.write_text("training:\n  epochs: 2\n", encoding="utf-8")
    manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("training_complete")
    manifest["success"] = True
    manifest["resolved_config"] = {"path": "resolved_config.yaml", "sha256": sha256_file(config_path)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_plan(root)["checkpoints_to_remove"] == 1
    config_path.write_text("training:\n  epochs: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="resolved config SHA256 mismatch"):
        build_plan(root)
    assert first.exists()
