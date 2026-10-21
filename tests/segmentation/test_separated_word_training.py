from copy import deepcopy
from pathlib import Path
import json

import pytest
import yaml
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation.train as training_module

from scripts.prepare_separated_word_training import (
    ARTIFACT, PROTOCOL, TARGET, VARIANTS, relative_input, training_manifest, write_frozen,
)
from segmentation.config import SegmentationConfigError, load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.train import RealTrainingError, run_training


REPO = Path(__file__).resolve().parents[2]
NEW_CONFIG = REPO / "configs/approximate/segmentation_total_text_separated_words_v1.yaml"
V4_CONFIG = REPO / "configs/approximate/segmentation_total_text_curved_word_boxes_v4.yaml"


def test_new_target_config_preserves_v4_training_and_keeps_distinct_identity():
    original = load_segmentation_config(V4_CONFIG)
    new = load_segmentation_config(NEW_CONFIG)
    assert original.architecture == new.architecture
    assert original.transforms == new.transforms
    assert original.checkpoint_selection == new.checkpoint_selection
    for key in original.training:
        if key != "protocol":
            assert original.training[key] == new.training[key]
    assert new.training.protocol == PROTOCOL
    assert new.data.train.artifact_id == ARTIFACT
    assert new.data.train.target_source == TARGET
    assert new.data.train.manifest_sha256 != original.data.train.manifest_sha256
    assert original.training.protocol == "practical_totaltext_curved_word_boxes_500_v1"


@pytest.mark.parametrize("mutation", ["v4_protocol", "v4_target", "short_run", "wrong_population"])
def test_cross_target_config_and_shortened_protocol_are_rejected(tmp_path, mutation):
    config = yaml.safe_load(NEW_CONFIG.read_text())
    if mutation == "v4_protocol":
        config["training"]["protocol"] = "practical_totaltext_curved_word_boxes_500_v1"
    elif mutation == "v4_target":
        config["data"]["train"]["target_source"] = "totaltext_filled_curved_box_v4"
    elif mutation == "short_run":
        config["training"]["epochs"] = 100
    else:
        config["data"]["train"]["expected_sample_count"] = 300
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(SegmentationConfigError):
        load_segmentation_config(path)


def test_frozen_writes_cannot_replace_other_artifacts(tmp_path):
    path = tmp_path / "manifest.yaml"
    write_frozen(path, {"value": 1})
    original = path.read_bytes()
    write_frozen(path, {"value": 1})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        write_frozen(path, {"value": 2})
    assert path.read_bytes() == original


def test_relative_inputs_reject_escape_through_symlinks(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (root / "escape.png").symlink_to(outside)
    assert relative_input(root / "mask.png", root) == "mask.png"
    with pytest.raises(ValueError, match="inside"):
        relative_input(root / "escape.png", root)


def synthetic_qualified_source(root):
    train = root / "total_text/train"
    train.mkdir(parents=True)
    config = train / "resolved_config.json"
    config.write_text('{}\n')
    row = {"image": str(root / "images/1.jpg"), "image_sha256": "a" * 64}
    for name in ("masks", "owners", "validity", "separators"):
        row[name] = f"total_text/train/{name}/1.png"
        row[name + "_sha256"] = "b" * 64
    data = {"dataset": "total_text", "split": "train", "images": 1255,
            "claim_label": "approximate reproduction", "complete_official_split": True,
            "missing_words": 0, "interword_contact_pairs": 0,
            "resolved_config_sha256": file_sha256(config),
            "samples": [{**deepcopy(row), "id": f"total_text:{i}"} for i in range(1255)]}
    manifest = train / "manifest.json"
    manifest.write_text(json.dumps(data))
    audit = {"datasets": [{"dataset": "total_text", "split": "train", "images": 1255,
                            "missing_words": 0, "merged_word_components": 0,
                            "manifest_sha256": file_sha256(manifest)}]}
    (root / "independent_audit.json").write_text(json.dumps(audit))
    return manifest


def test_bridge_preserves_source_hashes_ids_and_auxiliary_masks(tmp_path):
    synthetic_qualified_source(tmp_path)
    manifest = training_manifest(tmp_path, tmp_path)
    path = tmp_path / "bridge.yaml"
    write_frozen(path, manifest)
    loaded = load_segmentation_manifest(path)
    assert loaded.artifact_id == ARTIFACT and loaded.split_role == "train"
    assert len(loaded.samples) == 1255
    first = loaded.samples[0]
    assert first.sample_id == "total_text:0"
    assert first.image_sha256 == "a" * 64 and first.mask_sha256 == "b" * 64
    assert first.separation_annotation_path is not None
    assert first.word_owner_annotation_sha256 == "b" * 64


def test_bridge_rejects_gt_changed_after_independent_audit(tmp_path):
    path = synthetic_qualified_source(tmp_path)
    data = json.loads(path.read_text())
    data["samples"][0]["masks_sha256"] = "c" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="independently verified"):
        training_manifest(tmp_path, tmp_path)


def test_v1_bridge_cannot_mislabel_qualified_v2_targets(tmp_path):
    path=synthetic_qualified_source(tmp_path)
    data=json.loads(path.read_text());data['target_source']='compact_word_cores_v2'
    path.write_text(json.dumps(data))
    audit_path=tmp_path/'independent_audit.json'
    audit=json.loads(audit_path.read_text());audit['datasets'][0]['manifest_sha256']=file_sha256(path)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError,match='relabel'):
        training_manifest(tmp_path,tmp_path,variant='separated_v1')


@pytest.mark.parametrize("wrong_target", [False, True])
@pytest.mark.parametrize("variant", ["separated_v1", "compact_v2"])
def test_training_dispatch_and_target_guard_with_single_process_cpu_fixture(tmp_path, monkeypatch, wrong_target, variant):
    """Exercise dispatch/checkpoints; not a real U-Net or 8-worker qualification.

    The managed tool sandbox rejects AF_UNIX tensor-sharing sockets (EPERM).
    Use an explicit single-process fixture while retaining production worker8
    config validation. Production DataLoader and error propagation are unchanged.
    """
    def fixture_loader(*args, **kwargs):
        assert kwargs["num_workers"] == 8
        kwargs["num_workers"] = 0
        return DataLoader(*args, **kwargs)
    monkeypatch.setattr(training_module, "DataLoader", fixture_loader)
    source = synthetic_qualified_source(tmp_path)
    data = json.loads(source.read_text())
    if variant == "compact_v2":
        data["target_source"] = "compact_word_cores_v2"
    image = tmp_path / "images/1.jpg"
    image.parent.mkdir()
    Image.fromarray(np.full((16, 16, 3), 96, dtype=np.uint8)).save(image)
    for name in ("masks", "owners", "validity", "separators"):
        path = tmp_path / data["samples"][0][name]
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((16, 16), dtype=np.uint8)
        if name == "validity":
            pixels[:] = 255
        elif name == "owners":
            pixels[4:12, 4:12] = 1
        elif name == "masks":
            pixels[4:12, 4:12] = 255
        Image.fromarray(pixels).save(path)
        for row in data["samples"]:
            row[name + "_sha256"] = file_sha256(path)
    for row in data["samples"]:
        row["image_sha256"] = file_sha256(image)
    source.write_text(json.dumps(data))
    audit_path = tmp_path / "independent_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["datasets"][0]["manifest_sha256"] = file_sha256(source)
    audit_path.write_text(json.dumps(audit))
    manifest = training_manifest(tmp_path, tmp_path, variant=variant)
    if wrong_target:
        manifest["annotation_policy"]["target_source"] = "rejected_v4_target"
    manifest_path = tmp_path / "manifests/total_text_train.yaml"
    write_frozen(manifest_path, manifest)
    config = yaml.safe_load(NEW_CONFIG.read_text())
    identity = VARIANTS[variant]
    config["training"]["protocol"] = identity["protocol"]
    config["run"]["name"] = identity["run_name"]
    config["data"]["train"].update({"artifact_id":identity["artifact"],"target_source":identity["target"]})
    config["data"]["train"]["manifest_sha256"] = file_sha256(manifest_path)
    config_path = tmp_path / "config.yaml"
    write_frozen(config_path, config)
    model_factory = lambda: nn.Conv2d(3, 1, kernel_size=1, stride=2)
    kwargs = dict(config_path=config_path, data_root=tmp_path, output_root=tmp_path / "runs",
                  manifest_root=tmp_path, device_name="cpu", qualification_max_steps=1,
                  _model_factory=model_factory)
    if wrong_target:
        with pytest.raises(RealTrainingError, match="wider-gap target"):
            run_training(**kwargs)
    else:
        # Keep the CPU fixture bounded; scientific production settings are unchanged.
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            report = run_training(**kwargs)
        finally:
            torch.set_num_threads(old_threads)
        assert report["status"] == "QUALIFICATION_SUCCESS"
        assert report["training_complete"] is False
        assert report["result_class"] == "smoke-test result"
        assert report["datasets"]["train"]["sample_count"] == 1255
        assert len(report["epochs"]) == 1
        assert Path(report["checkpoint_output_location"]).is_file()
