import inspect

import numpy as np
import pytest

from localize.multiscale_component_persistence import (
    SCALES, group_multiscale_component_persistence, groups_in_original_coordinates,
)


def mask_with(boxes):
    mask = np.zeros((128, 128), np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask


def test_scale_space_merges_components_and_final_bbox_uses_raw_union():
    mask = mask_with([(10, 20, 14, 40), (18, 20, 22, 40)])
    result = group_multiscale_component_persistence(mask)
    pair = next(node for node in result.nodes if node["raw_component_count"] == 2)
    assert pair["birth_scale"] > 0
    assert pair["bbox_native"] == (10, 20, 22, 40)
    assert pair["foreground_area"] == 160
    assert any(row["radius_pixels"] > 0 for row in result.lines[0]["scale_partitions"])


def test_selected_spans_are_disjoint_complete_and_deterministic():
    mask = mask_with([(10 + 8 * index, 20, 13 + 8 * index, 40) for index in range(5)])
    first = group_multiscale_component_persistence(mask)
    second = group_multiscale_component_persistence(mask)
    flat = [value for row in first.selected for value in row["raw_component_ids"]]
    assert sorted(flat) == [1, 2, 3, 4, 5] and len(flat) == len(set(flat))
    assert first.as_dict() == second.as_dict()


def test_different_lines_never_merge_under_horizontal_dilation():
    result = group_multiscale_component_persistence(mask_with([
        (10, 10, 14, 25), (18, 10, 22, 25),
        (10, 50, 14, 65), (18, 50, 22, 65),
    ]))
    assert len(result.lines) == 2
    assert all(not ({1, 2} & set(row["raw_component_ids"]) and
                    {3, 4} & set(row["raw_component_ids"])) for row in result.nodes)


def test_singleton_empty_mapping_padding_and_top10():
    singleton = group_multiscale_component_persistence(mask_with([(0, 0, 2, 3)]))
    groups, retained = groups_in_original_coordinates(singleton, 256, 256)
    assert len(groups) == 1 and groups[0]["bbox"] == (0, 0, 4, 6)
    assert groups[0]["padded_bbox"] == (0, 0, 10, 12)
    assert retained == {(1,)}
    empty = group_multiscale_component_persistence(mask_with([]))
    assert empty.selected == ()
    many = group_multiscale_component_persistence(mask_with([
        (2 + 10 * (index % 11), 2 + 30 * (index // 11), 4 + 10 * (index % 11), 8 + 30 * (index // 11))
        for index in range(12)
    ]))
    _, retained = groups_in_original_coordinates(many, 128, 128)
    assert len(retained) == 10


def test_input_contract_and_no_gt_or_image_api():
    with pytest.raises(ValueError):
        group_multiscale_component_persistence(np.zeros((128, 128), np.float32))
    with pytest.raises(ValueError):
        group_multiscale_component_persistence(np.full((128, 128), 255, np.uint8))
    assert SCALES == (0.0, .05, .10, .15, .20, .25, .30, .35, .40, .50, .60)
    assert set(inspect.signature(group_multiscale_component_persistence).parameters) == {"mask"}
