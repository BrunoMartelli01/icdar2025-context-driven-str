from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.evaluate_scene_text_v1_parseq import _verify_training_run
from scripts.package_totaltext_curved_box_v4_overlays import _selected_rows, overlap_panel
from segmentation.config import SegmentationConfigError, load_segmentation_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/approximate/segmentation_total_text_curved_word_boxes_v4.yaml"


def test_v4_config_is_isolated_and_pinned() -> None:
    config = load_segmentation_config(CONFIG)
    assert config.training.protocol == "practical_totaltext_curved_word_boxes_500_v1"
    assert config.training.epochs == 500
    assert config.run.name == "total-text-filled-curved-word-boxes-v4-500ep"
    assert config.data.train.manifest_sha256 == (
        "7ad8525d66294e07b2d63be0dd68d08a2e83dbecc04a4d55fb65cd6110c51bea"
    )
    assert config.run.variant_class == "PRACTICAL_APPROXIMATION"


@pytest.mark.parametrize("field", ("epochs", "manifest_sha256", "run_name"))
def test_v4_config_rejects_cross_target_or_short_training(tmp_path: Path, field: str) -> None:
    modified = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if field == "epochs":
        modified["training"]["epochs"] = 100
    elif field == "manifest_sha256":
        modified["data"]["train"]["manifest_sha256"] = "0" * 64
    else:
        modified["run"]["name"] = "rejected-v3"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(modified), encoding="utf-8")
    with pytest.raises(SegmentationConfigError):
        load_segmentation_config(path)


def test_evaluator_requires_full_configured_500_epoch_run(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    record = {
        "status": "SUCCESS", "success": True, "epochs": [{} for _ in range(500)],
        "checkpoint_output_location": str(checkpoint),
        "checkpoint": {"sha256": sha256(checkpoint.read_bytes()).hexdigest()},
    }
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")
    assert _verify_training_run(manifest, checkpoint, expected_epochs=500) == record
    with pytest.raises(ValueError, match="500 epochs"):
        record["epochs"].pop()
        manifest.write_text(json.dumps(record), encoding="utf-8")
        _verify_training_run(manifest, checkpoint, expected_epochs=500)


def test_overlap_color_semantics() -> None:
    original = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    curved = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    image = np.asarray(overlap_panel(original, curved))
    assert tuple(image[0, 0]) == (0, 0, 0)
    assert tuple(image[0, 1]) == (255, 255, 255)
    assert tuple(image[1, 0]) == (255, 80, 80)
    assert tuple(image[1, 1]) == (255, 255, 255)
    curved[0, 0] = 1
    assert tuple(np.asarray(overlap_panel(original, curved))[0, 0]) == (0, 210, 230)


def test_visual_selection_rejects_fallbacks() -> None:
    rows = [{"id": "total_text:1"}, {"id": "total_text:2"}]
    clean = {
        "words": 3, "changed_words": 2, "geometric_fallbacks": 0,
        "coverage_or_topology_fallbacks": 0, "full_image_fallback": False,
        "max_uniform_buffer_pixels": 0, "v1_source_seed_pixels": 5,
        "retained_v1_source_seed_pixels": 5, "final_foreground_pixels": 120,
        "base_foreground_pixels": 100,
    }
    audit = {"total_text:1": clean, "total_text:2": {**clean, "full_image_fallback": True}}
    assert _selected_rows(rows, audit, 5) == [rows[0]]
