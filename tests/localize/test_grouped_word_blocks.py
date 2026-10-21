from __future__ import annotations

import numpy as np
import pytest

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK
from localize.practical_remap_protocol import evaluate_test_masks
from scripts.evaluate_practical_remap_test import selected_train_parameters


def params(**overrides: float | int) -> GroupedWordBlockParams:
    values: dict[str, float | int | str] = {
        "horizontal_gap_ratio": 0.5,
        "vertical_overlap_min": 0.5,
        "max_center_y_ratio": 0.5,
        "max_height_ratio": 2.5,
        "pad": 0,
        "connectivity": 8,
        "tie_break": APPROXIMATE_TIE_BREAK,
    }
    values.update(overrides)
    return GroupedWordBlockParams(**values)  # type: ignore[arg-type]


def test_singleton_is_a_valid_group() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:12, 4:10] = 1
    result = group_word_blocks(mask, np.zeros((20, 30, 3), dtype=np.uint8), params())
    assert len(result.groups_before_cap) == len(result.retained) == 1
    assert result.retained[0].raw_component_ids == (1,)


def test_multi_character_word_is_grouped_by_geometry() -> None:
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[10:20, 3:9] = 1
    mask[10:20, 11:17] = 1
    mask[10:20, 19:25] = 1
    result = group_word_blocks(mask, np.zeros((30, 50, 3), dtype=np.uint8), params())
    assert len(result.retained) == 1
    assert result.retained[0].raw_component_ids == (1, 2, 3)
    assert result.retained[0].bbox == (3, 10, 25, 20)


def test_large_same_line_gap_keeps_two_words_separate() -> None:
    mask = np.zeros((30, 80), dtype=np.uint8)
    mask[10:20, 3:9] = 1
    mask[10:20, 11:17] = 1
    mask[10:20, 50:56] = 1
    mask[10:20, 58:64] = 1
    result = group_word_blocks(mask, np.zeros((30, 80, 3), dtype=np.uint8), params())
    assert [block.raw_component_ids for block in result.retained] == [(1, 2), (3, 4)]


def test_nearby_lines_do_not_group_without_vertical_overlap() -> None:
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[4:12, 5:11] = 1
    mask[14:22, 5:11] = 1
    result = group_word_blocks(mask, np.zeros((40, 30, 3), dtype=np.uint8), params())
    assert len(result.retained) == 2


def test_small_punctuation_component_is_retained_as_singleton() -> None:
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[10:20, 3:10] = 1
    mask[10:20, 12:19] = 1
    mask[4:6, 15:17] = 1
    result = group_word_blocks(mask, np.zeros((30, 50, 3), dtype=np.uint8), params())
    assert any(len(block.raw_component_ids) == 1 for block in result.retained)
    assert sorted(component_id for block in result.retained for component_id in block.raw_component_ids) == [1, 2, 3]


def test_order_top10_border_clipping_and_determinism() -> None:
    mask = np.zeros((12, 80), dtype=np.uint8)
    for index in range(11):
        mask[0:2, index * 7 : index * 7 + 2] = 1
    image = np.zeros((12, 80, 3), dtype=np.uint8)
    first = group_word_blocks(mask, image, params(pad=4, horizontal_gap_ratio=0.1))
    second = group_word_blocks(mask, image, params(pad=4, horizontal_gap_ratio=0.1))
    assert len(first.groups_before_cap) == 11
    assert len(first.retained) == 10
    assert first.rejected_group_ids
    assert first.retained[0].padded_bbox[1] == 0
    assert first.as_dict() == second.as_dict()


def test_frozen_test_evaluator_requires_the_complete_test_population() -> None:
    with pytest.raises(ValueError, match="exactly 233"):
        evaluate_test_masks([], [], params())


def test_test_evaluation_rejects_non_train_calibration_provenance() -> None:
    base = {
        "train_only_guard": {
            "split_role": "train", "sample_count": 229, "test_data_loaded": False,
        },
        "scientific_boundary": {"selection_used_test": False, "selection_used_ocr": False},
        "calibration": {"selected": {"parameters": vars(params())}},
    }
    assert selected_train_parameters(base).horizontal_gap_ratio == 0.5
    base["train_only_guard"]["test_data_loaded"] = True
    with pytest.raises(ValueError, match="contaminated"):
        selected_train_parameters(base)
