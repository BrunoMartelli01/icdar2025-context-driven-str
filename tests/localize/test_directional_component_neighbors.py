from collections import Counter
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from diagnostics.directional_neighbors_train import estimate_orientation, annotation_word_colors
from localize.directional_component_neighbors import directional_neighbors
from localize.grouped_word_blocks import GroupedWordBlockParams


PARAMS = GroupedWordBlockParams(.35, .7, .5, 2.5, 6)
ORIENTATION = dict(minimum_glyphs=3, minimum_pca_eigenvalue_ratio=4., axis_tolerance_degrees=15.)


def mask_at(points, shape=(128, 128)):
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in points:
        mask[y:y + 10, x:x + 10] = 1
    return mask


@pytest.mark.parametrize("points,stage,axis", [
    ([(10, 10), (22, 10), (34, 10)], "DIRECTIONAL_H", "H"),
    ([(10, 10), (10, 22), (10, 34)], "DIRECTIONAL_HV", "V"),
    ([(10, 10), (22, 22), (34, 34)], "DIRECTIONAL_HVD", "D45"),
    ([(34, 10), (22, 22), (10, 34)], "DIRECTIONAL_HVD", "D135"),
])
def test_straight_words_each_direction(points, stage, axis):
    result = directional_neighbors(mask_at(points), PARAMS)
    groups = result["methods"][stage]["groups"]
    assert len(groups) == 1
    assert groups[0]["axis"] == axis
    assert len(groups[0]["raw_component_ids"]) == 3
    assert groups[0]["foreground_area"] == 300


def test_two_words_same_line_large_gap_stays_separate():
    result = directional_neighbors(mask_at([(10, 10), (22, 10), (80, 10), (92, 10)]), PARAMS)
    assert len(result["methods"]["DIRECTIONAL_HVD"]["groups"]) == 2


def test_horizontal_priority_does_not_consume_a_group_again_vertically():
    result = directional_neighbors(mask_at([(10, 10), (22, 10), (10, 22), (10, 34)]), PARAMS)
    groups = result["methods"]["DIRECTIONAL_HV"]["groups"]
    assert sorted(g["axis"] for g in groups) == ["H", "V"]
    assert sorted(len(g["raw_component_ids"]) for g in groups) == [2, 2]
    ids = [i for g in groups for i in g["raw_component_ids"]]
    assert len(ids) == len(set(ids)) == 4


def test_top10_area_order_border_clipping_determinism_no_mutation():
    mask = mask_at([(x * 30, y * 40) for y in range(3) for x in range(4)])
    before = mask.copy()
    first = directional_neighbors(mask, PARAMS)
    assert first == directional_neighbors(mask, PARAMS)
    assert np.array_equal(mask, before)
    for value in first["methods"].values():
        assert len(value["groups"]) == 12
        assert len(value["boxes"]) == 10
        assert value["boxes"][0] == [0, 0, 16, 16]
        assert all(0 <= b[0] < b[2] <= 128 and 0 <= b[1] < b[3] <= 128 for b in value["boxes"])
        assert Counter(i for g in value["groups"] for i in g["raw_component_ids"]) == Counter(range(1, 13))


def test_singleton_and_empty_foreground_are_valid():
    for points, count in [([], 0), ([(10, 10)], 1)]:
        r = directional_neighbors(mask_at(points), PARAMS)
        assert len(r["methods"]["DIRECTIONAL_HVD"]["groups"]) == count
    with pytest.raises(ValueError):
        directional_neighbors(np.full((10, 10), .5), PARAMS)


@pytest.mark.parametrize("points,category", [
    ([(0, 5), (10, 5), (20, 5)], "HORIZONTAL"),
    ([(5, 0), (5, 10), (5, 20)], "VERTICAL"),
    ([(0, 0), (10, 10), (20, 20)], "DIAGONAL"),
    ([(20, 0), (10, 10), (0, 20)], "DIAGONAL"),
    ([(0, 0)], "UNKNOWN"),
    ([(0, 0), (10, 0)], "UNKNOWN"),
    ([(0, 0), (10, 0), (0, 10), (10, 10)], "UNKNOWN"),
])
def test_gt_orientation_not_forced_from_bbox_aspect(points, category):
    assert estimate_orientation(points, **ORIENTATION)["orientation"] == category


def test_gt_parser_and_no_gt_in_grouping_interface():
    assert annotation_word_colors('1 2 3 "a"\n4 5 6 "b"\n\n# 7 8 9 ignored\n10 11 12 "c"') == [
        [(1, 2, 3), (4, 5, 6)], [(10, 11, 12)]]
    assert list(inspect.signature(directional_neighbors).parameters) == ["mask", "params"]
    config = json.loads(Path("configs/approximate/directional_neighbors_train_v1.json").read_text())
    assert config["scope"] == {"split": "TRAIN", "images": 229, "gt_words": 849}
    assert config["priority_axes_degrees"] == [0, 90, 45, 135]
    assert config["orientation_diagnostic"] == ORIENTATION
    assert not config["tuning"] and not config["test_loaded"] and not config["ocr_run"]
