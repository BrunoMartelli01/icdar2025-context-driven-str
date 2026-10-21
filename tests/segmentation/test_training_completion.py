from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
import yaml

from segmentation.data import SegmentationManifestError
from segmentation.train import EpochShuffleSampler, RealTrainingError, run_training


ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "approximate" / "segmentation_icdar13.yaml"
DATASET_ARTIFACT = "ART-ICDAR13-FST-OFFICIAL-SPLIT"


class TinyHalfResolutionSegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Conv2d(3, 1, kernel_size=1, stride=2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_split(
    root: Path,
    *,
    role: str,
    count: int = 8,
    reuse_paths_from: str | None = None,
) -> Path:
    split = f"official_train_heldout_{role}"
    image_dir = root / "images" / role
    mask_dir = root / "masks" / role
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(count):
        if reuse_paths_from is None:
            image_path = image_dir / f"{index:03d}.png"
            mask_path = mask_dir / f"{index:03d}.png"
            image = np.full((16, 16, 3), 20 + index * 10, dtype=np.uint8)
            mask = np.full((16, 16, 3), 255, dtype=np.uint8)
            mask[3:12, 2 + index % 3 : 11 + index % 3] = (0, 64, 128)
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask).save(mask_path)
        else:
            image_path = root / "images" / reuse_paths_from / f"{index:03d}.png"
            mask_path = root / "masks" / reuse_paths_from / f"{index:03d}.png"
        samples.append(
            {
                "id": f"{role}-{index:03d}",
                "image": str(image_path.relative_to(root)),
                "mask": str(mask_path.relative_to(root)),
                "image_sha256": _sha(image_path),
                "mask_sha256": _sha(mask_path),
            }
        )
    manifest = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "artifact_id": DATASET_ARTIFACT,
        "dataset": "ICDAR13-FST",
        "split": split,
        "split_role": role,
        "split_policy": {
            "artifact_id": "ART-ICDAR13-M9-APPROX-SPLIT-POLICY",
            "algorithm_id": "icdar13-task2.2-sha256-ranked-holdout-v1",
            "classification": "ASSUMPTION",
            "seed": 202503,
            "policy_fingerprint_sha256": (
                "3a7c3d4226e735b2e4b57fef5f8c84fa9522ad1b60852d86229a975ff07d6124"
            ),
        },
        "annotation_policy": {
            "artifact_id": "ART-ICDAR13-M9-APPROX-TRAINING-MASK-POLICY",
            "classification": "ASSUMPTION",
            "source_format": "ICDAR13 Task 2.2 color-coded per-character RGB BMP",
            "target_conversion": "icdar13-task2.2-exact-white-background-v1",
            "policy_fingerprint_sha256": (
                "ca318685a8fea2a0c162d80986acb88d60ec950fdec440652ff4d57bf4c0adf3"
            ),
            "foreground_conversion": "exact RGB white -> 0; every other RGB -> 1",
            "conversion_classification": "REPOSITORY_SPECIFIED",
            "dont_care_training_behavior": "foreground",
            "dont_care_classification": "ASSUMPTION",
            "txt_role": "audit/provenance only; does not alter BCE targets",
            "ignore_mask": False,
            "masked_bce": False,
            "ignore_index": None,
            "ambiguity_ids": ["DS-003"],
            "training_ready": True,
        },
        "samples": samples,
    }
    path = root / "manifests" / f"{role}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _setup_data(root: Path, *, overlap_paths: bool = False) -> None:
    _write_split(root, role="train")
    _write_split(root, role="validation", reuse_paths_from="train" if overlap_paths else None)


def _write_config(tmp_path: Path, *, run_name: str, epochs: int = 2, seed: int = 202503) -> Path:
    raw = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    raw["run"]["name"] = run_name
    raw["training"]["epochs"] = epochs
    raw["training"]["seed"] = seed
    raw["data"]["train"]["manifest"] = "manifests/train.yaml"
    raw["data"]["validation"]["manifest"] = "manifests/validation.yaml"
    path = tmp_path / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _run(
    config: Path,
    data_root: Path,
    output_root: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return run_training(
        config_path=config,
        data_root=data_root,
        output_root=output_root,
        manifest_root=data_root,
        device_name="cpu",
        _model_factory=TinyHalfResolutionSegmenter,
        **kwargs,
    )


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


def test_multi_epoch_validation_and_run_manifest_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root)
    config = _write_config(tmp_path, run_name="multi-epoch")
    report = _run(config, data_root, tmp_path / "runs")

    assert report["status"] == "SUCCESS"
    assert [record["epoch"] for record in report["epochs"]] == [1, 2]
    assert all(record["train"]["samples"] == 8 for record in report["epochs"])
    assert all(record["validation"]["samples"] == 8 for record in report["epochs"])
    assert report["validation"] == {
        "source_explicit": True,
        "leakage_policy": "disjoint_sample_ids_and_resolved_paths",
        "protocol_classification": "ASSUMPTION",
        "checkpoint_semantics": "metrics_only_last_checkpoint",
    }
    assert report["architecture"]["identity"] == "approximate-ag-unet-figure3-v1"
    assert len(report["checkpoint"]["sha256"]) == 64
    assert report["checkpoint"]["optimizer_state_included"] is True
    assert report["checkpoint"]["rng_state_included"] is True
    assert report["datasets"]["train"]["manifest_sha256"] == _sha(
        data_root / "manifests" / "train.yaml"
    )
    assert report["scientific_mode"] == "approximate"
    assert report["strict_blockers"]
    lines = (tmp_path / "runs" / "multi-epoch" / "epochs.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["schema_version"] == 1 for line in lines)


def test_step_limited_qualification_is_successful_but_not_complete(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root)
    config = _write_config(tmp_path, run_name="qualification")

    report = _run(
        config,
        data_root,
        tmp_path / "runs",
        qualification_max_steps=1,
    )

    assert report["status"] == "QUALIFICATION_SUCCESS"
    assert report["success"] is True
    assert report["training_complete"] is False
    assert report["result_class"] == "smoke-test result"
    assert report["qualification"] == {
        "enabled": True,
        "passed": True,
        "max_steps_per_phase": 1,
        "configured_epochs": 2,
        "completed_epoch": 1,
        "epochs_executed_this_invocation": 1,
        "training_steps_executed_this_invocation": 1,
        "full_training_complete": False,
    }
    assert len(report["epochs"]) == 1


def test_epoch_order_is_deterministic_and_epoch_addressable() -> None:
    first = EpochShuffleSampler(17, seed=41)
    second = EpochShuffleSampler(17, seed=41)
    assert list(first) == list(second)
    first.set_epoch(1)
    second.set_epoch(1)
    assert list(first) == list(second)
    second.set_epoch(2)
    assert list(first) != list(second)


def test_validation_paths_cannot_leak_from_training(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root, overlap_paths=True)
    config = _write_config(tmp_path, run_name="leakage")
    with pytest.raises(RealTrainingError, match="resolved paths overlap"):
        _run(config, data_root, tmp_path / "runs")
    failure = json.loads((tmp_path / "runs" / "leakage" / "failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert failure["success"] is False


def test_uninterrupted_and_resumed_continuation_are_exact(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root)
    full_config = _write_config(tmp_path, run_name="full", epochs=3)
    resumed_config = _write_config(tmp_path, run_name="resumed", epochs=3)
    full = _run(full_config, data_root, tmp_path / "runs")
    partial = _run(
        resumed_config,
        data_root,
        tmp_path / "runs",
        _max_epochs_this_invocation=1,
    )
    assert partial["status"] == "INCOMPLETE"
    resume_checkpoint = tmp_path / "runs" / "resumed" / partial["checkpoint"]["path"]
    resumed = _run(
        resumed_config,
        data_root,
        tmp_path / "runs",
        resume=resume_checkpoint,
    )

    full_payload = torch.load(
        tmp_path / "runs" / "full" / full["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    resumed_payload = torch.load(
        tmp_path / "runs" / "resumed" / resumed["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    _assert_nested_equal(full_payload["model_state_dict"], resumed_payload["model_state_dict"])
    _assert_nested_equal(
        full_payload["optimizer_state_dict"], resumed_payload["optimizer_state_dict"]
    )
    assert full_payload["epoch"] == resumed_payload["epoch"] == 3
    assert full_payload["global_step"] == resumed_payload["global_step"] == 3
    assert len(full["epochs"]) == len(resumed["epochs"])
    for full_epoch, resumed_epoch in zip(full["epochs"], resumed["epochs"]):
        assert full_epoch["train"] == resumed_epoch["train"]
        assert full_epoch["validation"] == resumed_epoch["validation"]


def test_resume_fails_closed_on_changed_training_contract(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root)
    config = _write_config(tmp_path, run_name="contract", epochs=3)
    partial = _run(
        config,
        data_root,
        tmp_path / "runs",
        _max_epochs_this_invocation=1,
    )
    checkpoint = tmp_path / "runs" / "contract" / partial["checkpoint"]["path"]
    changed = _write_config(tmp_path, run_name="contract", epochs=3, seed=202504)
    with pytest.raises(ValueError, match="run contract|training configuration"):
        _run(changed, data_root, tmp_path / "runs", resume=checkpoint)


def test_missing_real_manifest_fails_without_demo_fallback(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = _write_config(tmp_path, run_name="missing")
    with pytest.raises(SegmentationManifestError, match="Cannot read manifest"):
        _run(config, data_root, tmp_path / "runs")
    assert not (data_root / "images").exists()
    assert not (data_root / "masks").exists()
    failure = json.loads((tmp_path / "runs" / "missing" / "failure.json").read_text())
    assert failure["status"] == "FAILED"
    assert "manifest" in failure["error"].lower()


def test_unregistered_dataset_artifact_fails_before_training(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _setup_data(data_root)
    config = _write_config(tmp_path, run_name="unregistered")
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["data"]["train"]["artifact_id"] = "ART-NOT-REGISTERED"
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(RealTrainingError, match="not registered"):
        _run(config, data_root, tmp_path / "runs")
