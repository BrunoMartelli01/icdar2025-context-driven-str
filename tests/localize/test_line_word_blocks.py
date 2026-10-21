from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from localize.line_word_blocks import (
    V2WordBlockParams,
    group_prepared_word_blocks_v2,
    group_word_blocks_v2,
    prepare_word_blocks_v2,
)
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK
from localize.practical_remap_v2_protocol import declared_v2_grid, evaluate_v1_v2_test_geometry
from scripts.calibrate_practical_remap_v2_train import _validate_train_manifest
from scripts.evaluate_practical_remap_v2_test import _validate_v1_ocr_cache


def params(**overrides: float | int | str) -> V2WordBlockParams:
    values: dict[str, float | int | str] = {
        "family": "hybrid",
        "absolute_gap_floor": 0.35,
        "local_window": 2,
        "local_gap_ratio": 2.0,
        "local_fallback_absolute_floor": 0.35,
        "line_vertical_overlap_min": 0.7,
        "line_max_center_y_ratio": 0.5,
        "line_max_height_ratio": 2.5,
        "fragment_x_overlap_min": 0.5,
        "fragment_max_vertical_gap_ratio": 0.5,
        "fragment_small_height_ratio": 0.45,
        "pad": 0,
        "connectivity": 8,
        "tie_break": APPROXIMATE_TIE_BREAK,
        "nmax": 10,
    }
    values.update(overrides)
    return V2WordBlockParams(**values)  # type: ignore[arg-type]


def rgb(mask: np.ndarray) -> np.ndarray:
    return np.zeros((*mask.shape, 3), dtype=np.uint8)


def test_fragment_consolidation_is_separate_from_word_grouping() -> None:
    mask = np.zeros((30, 25), dtype=np.uint8)
    mask[3:5, 7:10] = 1  # dot-like fragment
    mask[7:20, 7:11] = 1  # vertical stem
    result = group_word_blocks_v2(mask, rgb(mask), params())
    assert len(result.raw_components) == 2
    assert len(result.atoms) == 1
    assert result.atoms[0].raw_component_ids == (1, 2)
    assert len(result.retained) == 1


def test_local_ratio_preserves_irregular_intra_word_spacing() -> None:
    mask = np.zeros((25, 60), dtype=np.uint8)
    mask[8:18, 0:10] = 1
    mask[8:18, 13:23] = 1
    mask[8:18, 30:40] = 1
    result = group_word_blocks_v2(mask, rgb(mask), params(local_gap_ratio=2.5))
    assert len(result.line_candidates) == 1
    assert len(result.retained) == 1
    assert result.retained[0].raw_component_ids == (1, 2, 3)


def test_local_ratio_separates_two_words_on_the_same_line() -> None:
    mask = np.zeros((25, 80), dtype=np.uint8)
    for left, right in ((0, 10), (12, 22), (40, 50), (52, 62)):
        mask[8:18, left:right] = 1
    result = group_word_blocks_v2(mask, rgb(mask), params())
    assert len(result.line_candidates) == 1
    assert [block.raw_component_ids for block in result.retained] == [(1, 2), (3, 4)]


def test_one_character_word_is_valid() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:11] = 1
    result = group_word_blocks_v2(mask, rgb(mask), params())
    assert len(result.retained) == 1
    assert result.retained[0].raw_component_ids == (1,)


def test_small_punctuation_is_not_forced_into_a_distant_word() -> None:
    mask = np.zeros((30, 60), dtype=np.uint8)
    mask[12:22, 2:12] = 1
    mask[12:22, 14:24] = 1
    mask[3:5, 45:47] = 1
    result = group_word_blocks_v2(mask, rgb(mask), params())
    assert any(block.raw_component_ids == (1,) for block in result.retained)


def test_close_but_non_overlapping_lines_do_not_form_one_line() -> None:
    mask = np.zeros((40, 35), dtype=np.uint8)
    mask[4:12, 5:12] = 1
    mask[14:22, 5:12] = 1
    result = group_word_blocks_v2(mask, rgb(mask), params())
    assert len(result.line_candidates) == 2
    assert len(result.retained) == 2


def test_order_top10_and_border_clipping_are_deterministic() -> None:
    mask = np.zeros((20, 120), dtype=np.uint8)
    for index in range(11):
        mask[0:4, index * 10:index * 10 + 3] = 1
    first = group_word_blocks_v2(mask, rgb(mask), params(family="absolute", pad=3, absolute_gap_floor=0.1))
    second = group_word_blocks_v2(mask, rgb(mask), params(family="absolute", pad=3, absolute_gap_floor=0.1))
    assert len(first.groups_before_cap) == 11
    assert len(first.retained) == 10
    assert first.retained[0].padded_bbox[1] == 0
    assert first.as_dict() == second.as_dict()


def test_prepared_mask_rejects_different_line_or_fragment_geometry() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:11] = 1
    prepared = prepare_word_blocks_v2(mask, params())
    with pytest.raises(ValueError, match="incompatible"):
        group_prepared_word_blocks_v2(prepared, rgb(mask), params(fragment_small_height_ratio=0.4))


def test_grid_is_bounded_and_test_evaluation_requires_all_233_samples() -> None:
    assert len(declared_v2_grid()) == 35
    with pytest.raises(ValueError, match="exactly all 233"):
        evaluate_v1_v2_test_geometry([], [], None, None)  # type: ignore[arg-type]


def test_v1_ocr_cache_provenance_is_fail_closed() -> None:
    methods = ("ALG1", "PRACTICAL_REMAP", "PRED_COMPONENT_UNION", "GT_BOX")
    report = {
        "mode": "practical-remap-v1",
        "classification": "PRACTICAL_APPROXIMATION",
        "selection": {"train_calibration_sha256": "a" * 64},
        "frozen_test": {"geometry": {"per_sample": [{} for _ in range(233)]}},
        "first_ten_parseq": {
            "new_practical_parseq_only": True,
            "per_word": [
                {
                    "sample_id": str(index + 1),
                    "gt_index": 1,
                    "ocr_comparison": {name: {} for name in methods},
                }
                for index in range(66)
            ],
        },
        "scientific_boundary": {"table_5_claim": False},
    }
    assert len(_validate_v1_ocr_cache(report, expected_calibration_sha256="a" * 64)) == 66
    report["selection"]["train_calibration_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="provenance"):
        _validate_v1_ocr_cache(report, expected_calibration_sha256="a" * 64)


def test_v2_train_manifest_guard_reports_wrong_split_actionably() -> None:
    manifest = SimpleNamespace(
        split="official-training-source-approximate-train",
        split_role="train",
        samples=tuple(range(206)),
    )
    with pytest.raises(ValueError, match="icdar13_segmentation_author_protocol.yaml"):
        _validate_train_manifest(
            manifest, Path("manifests/icdar13_segmentation_train.yaml")
        )
