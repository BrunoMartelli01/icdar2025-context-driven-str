import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.review_native128_ocr import original_padding_proposals, inference_config
from scripts.evaluate_cached_totaltext_padding import proposals_for_padding


def test_original_padding_changes_only_crop_and_clips_at_real_image_edges():
    proposals = [{"rank": 1, "bbox": [2, 1, 15, 12], "bbox_native": [1, 1, 8, 8],
                  "contour_index": 7, "foreground_area_native": 23}]
    actual = original_padding_proposals(proposals, (14, 20), 6)["proposals"][0]
    assert actual["padded_bbox_original"] == [0, 0, 20, 14]
    assert actual["bbox"] == [2, 1, 15, 12]
    assert actual["bbox_native"] == [1, 1, 8, 8]
    assert actual["foreground_area_native"] == 23
    assert actual["contour_index"] == 7
    assert actual["padded_bbox_native"] is None
    assert "padded_bbox_original" not in proposals[0]


def test_six_original_pixels_are_not_six_native_pixels():
    mask = np.zeros((128, 128), np.uint8)
    mask[30:40, 30:50] = 1
    plan = {"padding_native_pixels": [0, 6], "min_unpadded_bbox_area_native": 150, "max_proposals_per_image": 10}
    variants = proposals_for_padding(mask, (384, 640), plan)
    original = original_padding_proposals(variants["0"]["proposals"], (384, 640), 6)
    assert original["proposals"][0]["padded_bbox_original"] == [144, 84, 256, 126]
    assert variants["6"]["proposals"][0]["padded_bbox_original"] == [120, 72, 280, 138]
    assert original["proposals"][0]["bbox"] == variants["6"]["proposals"][0]["bbox"]


def test_disabled_area_filter_keeps_tiny_components_but_preserves_top10():
    mask = np.zeros((128, 128), np.uint8)
    for i in range(12):
        mask[2+i*10:4+i*10, 5:7] = 1
    plan = {"padding_native_pixels": [0, 6], "min_unpadded_bbox_area_native": 0, "max_proposals_per_image": 10}
    variants = proposals_for_padding(mask, (384, 640), plan)
    assert len(variants["0"]["proposals"]) == 10
    assert variants["0"]["candidates_before_cap"] == 12
    assert original_padding_proposals([], (384, 640), 6)["proposals"] == []


def test_inference_snapshot_preserves_historical_worker_setting(tmp_path):
    cfg = OmegaConf.create({"reproduction_mode": "approximate", "architecture": {"implementation": "author_clarified_ag_unet"},
        "transforms": {"image_size": [256, 256], "mask_size": [128, 128], "normalization": "imagenet",
            "resize_geometry": "stretch", "image_interpolation": "bilinear"}, "training": {"num_workers": 0}})
    path = tmp_path / "legacy.yaml"
    OmegaConf.save(cfg, path)
    before = path.read_bytes()
    loaded = inference_config(path)
    assert loaded.training.num_workers == 0 and path.read_bytes() == before
    cfg.transforms.mask_size = [256, 256]
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="mask_size"):
        inference_config(path)
