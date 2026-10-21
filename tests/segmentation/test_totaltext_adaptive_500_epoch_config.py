from pathlib import Path

import pytest
import yaml

from segmentation.config import SegmentationConfigError, load_segmentation_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/approximate/segmentation_total_text_adaptive_word_boxes_v3.yaml"


def test_isolated_adaptive_box_run_requires_500_epochs() -> None:
    config = load_segmentation_config(CONFIG)
    assert config.training.epochs == 500
    assert config.training.protocol == "practical_totaltext_adaptive_word_boxes_500_v1"
    assert config.run.name == "total-text-adaptive-filled-word-boxes-v3-500ep"
    assert config.training.checkpoint_selection == "training_foreground_iou"


def test_adaptive_box_protocol_rejects_100_epochs_or_wrong_run_name(tmp_path: Path) -> None:
    original = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for field, value in (("epochs", 100), ("run_name", "old-v3-run")):
        modified = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        if field == "epochs":
            modified["training"]["epochs"] = value
        else:
            modified["run"]["name"] = value
        path = tmp_path / f"{field}.yaml"
        path.write_text(yaml.safe_dump(modified), encoding="utf-8")
        with pytest.raises(SegmentationConfigError):
            load_segmentation_config(path)
    assert original["training"]["epochs"] == 500


def test_paper_hyperparameter_scene_text_config_still_requires_100_epochs() -> None:
    other = ROOT / "configs/approximate/segmentation_total_text_author_hyperparameters.yaml"
    config = load_segmentation_config(other)
    assert config.training.epochs == 100
