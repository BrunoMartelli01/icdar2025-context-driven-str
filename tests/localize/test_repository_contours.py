"""Differential parity with the frozen author source, plus scientific edge cases."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
from PIL import Image

from localize.mask_to_blocks import BlockParam, localize_mask
from localize.repository_contours import (
    SOURCE_COMMIT, SOURCE_SHA256, RepositoryContourParams,
    audit_repository_contours, crop_repository_blocks, mask_to_blocks,
)


@pytest.fixture(scope="module")
def original():
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:localize/mask_to_blocks.py"], cwd=root
    )
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA256
    module = types.ModuleType("_verified_upstream_contours")
    sys.modules[module.__name__] = module
    try:
        exec(compile(source, "verified_frozen_author_source", "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[module.__name__]


@pytest.mark.parametrize("seed", range(20))
def test_frozen_upstream_parity(original, seed):
    rng = np.random.default_rng(seed)
    for shape in ((1, 1), (32, 47), (128, 128)):
        mask = np.zeros(shape, np.uint8)
        for _ in range(40):
            y, x = [int(rng.integers(n)) for n in shape]
            mask[y:y + int(rng.integers(1, 10)), x:x + int(rng.integers(1, 12))] = 255
        for area, pad in ((1, 0), (8, 3), (150, 6), (200, 200)):
            params = RepositoryContourParams(area, pad, 10)
            before = mask.copy()
            assert mask_to_blocks(mask, params) == original.mask_to_blocks(
                mask, original.BlockParam(area, pad, 10)
            )
            np.testing.assert_array_equal(mask, before)


def test_rectangle_area_is_not_foreground_area():
    mask = np.zeros((50, 70), np.uint8)
    mask[2:22, 2] = 1
    mask[21, 2:22] = 1  # 39 pixels, but bbox area 400
    mask[25:35, 40:60] = 1  # 200 pixels
    image = np.zeros((*mask.shape, 3), np.uint8)
    current = localize_mask(mask, image, BlockParam(150, 6, 8, "top_to_bottom_then_left_to_right"))
    upstream = audit_repository_contours(mask, RepositoryContourParams(150, 6, 10))
    assert len(current.retained) == 1
    assert len(upstream["retained"]) == 2
    assert upstream["retained"][0]["bbox_area"] == 400


def test_external_contours_omit_nested_islands():
    mask = np.zeros((30, 30), np.uint8)
    mask[2:28, 2:28] = 1
    mask[3:27, 3:27] = 0
    mask[12:16, 12:16] = 1
    report = audit_repository_contours(mask, RepositoryContourParams(1, 0, 10))
    assert report["raw_external_contours"] == 1
    assert report["retained"][0]["bbox"] == [2, 2, 28, 28]


def test_padding_sort_uses_clipped_padded_area():
    mask = np.zeros((40, 40), np.uint8)
    mask[0:10, 0:10] = 1
    mask[20:28, 20:28] = 1
    report = audit_repository_contours(mask, RepositoryContourParams(1, 6, 10))
    assert [r["bbox_area"] for r in report["retained"]] == [64, 100]
    assert [r["padded_bbox_area"] for r in report["retained"]] == [400, 256]


def test_separate_fragments_stay_separate_and_top10_ties_match(original):
    mask = np.zeros((20, 150), np.uint8)
    for x in range(0, 150, 10):
        mask[3:10, x:x + 3] = 1
    params = RepositoryContourParams(1, 6, 10)
    report = audit_repository_contours(mask, params)
    assert report["raw_external_contours"] == 15
    assert len(report["retained"]) == 10
    assert len(report["rejected"]) == 5
    assert mask_to_blocks(mask, params) == original.mask_to_blocks(mask, original.BlockParam(1, 6, 10))
    assert mask_to_blocks(mask, params) == mask_to_blocks(mask, params)


def test_crops_original_pixels_and_exclusive_border():
    mask = np.zeros((8, 9), np.uint8)
    mask[-1, -1] = 1
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    report, crops = crop_repository_blocks(mask, image, RepositoryContourParams(1, 2, 10))
    assert report["retained"][0]["padded_bbox"] == [6, 5, 9, 8]
    np.testing.assert_array_equal(crops[0], image[5:8, 6:9])
    assert not np.shares_memory(crops[0], image)


def test_empty_and_min_area_boundary():
    mask = np.zeros((30, 30), np.uint8)
    params = RepositoryContourParams(150, 6, 10)
    assert mask_to_blocks(mask, params) == []
    mask[2:12, 2:17] = 1
    assert len(mask_to_blocks(mask, params)) == 1
    assert mask_to_blocks(mask, RepositoryContourParams(151, 6, 10)) == []


@pytest.mark.parametrize("mask", [np.zeros((2, 3, 3)), np.zeros((0, 2)), np.full((2, 2), .7), np.full((2, 2), np.nan)])
def test_invalid_mask_fail_closed(mask):
    with pytest.raises(ValueError):
        mask_to_blocks(mask, RepositoryContourParams(1, 0, 10))


def test_mismatched_image_fail_closed():
    with pytest.raises(ValueError):
        crop_repository_blocks(np.zeros((2, 2)), np.zeros((3, 3, 3)), RepositoryContourParams(1, 0, 10))


@pytest.mark.parametrize("values", [(150, -1, 10), (150, 6, 0), (1.5, 6, 10), (True, 6, 10)])
def test_invalid_params(values):
    with pytest.raises(ValueError):
        RepositoryContourParams(*values)


def test_zero_min_area_disables_bbox_area_filter():
    mask = np.zeros((20, 20), np.uint8)
    mask[3, 3] = 255
    report = audit_repository_contours(mask, RepositoryContourParams(0, 0, 10))
    assert report["filter_area"] == "none"
    assert report["candidates_before_cap"] == 1
    assert report["retained"][0]["bbox_area"] == 1


def test_cpu_comparison_cli_and_overwrite_guard(tmp_path, monkeypatch, capsys):
    from diagnostics.compare_repository_localizer import main

    root = Path(__file__).resolve().parents[2]
    mask = np.zeros((50, 70), np.uint8)
    mask[2:22, 2] = 255
    mask[21, 2:22] = 255
    mask[25:35, 40:60] = 255
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.fromarray(np.zeros((50, 70, 3), np.uint8)).save(tmp_path / "rgb.png")
    monkeypatch.setattr(sys, "argv", ["comparison", "--mask", str(tmp_path / "mask.png"),
        "--image", str(tmp_path / "rgb.png"), "--config",
        str(root / "configs/approximate/localization_repository_contours_v1.json"),
        "--output", str(tmp_path / "result.json")])
    assert main() == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert len(result["repository"]["retained"]) == 2
    assert len(result["current_component_area"]["retained"]) == 1
    assert result["same_ordered_boxes"] is False
    assert "SHA256" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main()


def test_cli_rejects_strict_mode_before_loading_inputs(tmp_path, monkeypatch):
    from diagnostics.compare_repository_localizer import main

    config = tmp_path / "strict.json"
    config.write_text(json.dumps({"reproduction_mode": "strict"}))
    monkeypatch.setattr(sys, "argv", ["comparison", "--mask", "absent", "--image", "absent",
        "--config", str(config), "--output", str(tmp_path / "result.json")])
    with pytest.raises(ValueError, match="approximate"):
        main()
