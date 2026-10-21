from __future__ import annotations

import numpy as np

from localize.gap_word_regions import line_gap_segmentation, linewise_rlsa


def _gap_fixture() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.uint8)
    for left in (10, 17, 24, 60):
        mask[40:50, left:left + 5] = 1
    return mask


def test_rlsa_fills_short_bounded_runs_without_bridging_large_gap() -> None:
    result = linewise_rlsa(_gap_fixture(), run_height_ratio=0.25)
    assert {group["raw_component_ids"] for group in result.groups} == {(1, 2, 3), (4,)}


def test_otsu_splits_the_large_line_local_gap_deterministically() -> None:
    result = line_gap_segmentation(_gap_fixture(), method="otsu")
    assert [group["raw_component_ids"] for group in result.groups.groups] == [(1, 2, 3), (4,)]
    assert result.line_statistics["lines_with_at_least_two_gaps"] == 1
    assert result.line_statistics["lines_with_real_two_group_separation"] == 1
    assert result.line_statistics["lines_using_fallback"] == 0
    assert result.line_statistics["boundary_count"] == 1
    assert result.line_statistics["merge_count"] == 2


def test_two_means_requires_declared_cluster_separation() -> None:
    significant = line_gap_segmentation(
        _gap_fixture(), method="two_means", minimum_cluster_separation=0.2
    )
    assert significant.line_statistics["lines_with_real_two_group_separation"] == 1
    rejected = line_gap_segmentation(
        _gap_fixture(), method="two_means", minimum_cluster_separation=10.0
    )
    assert rejected.line_statistics["lines_with_real_two_group_separation"] == 0
    assert rejected.line_statistics["lines_using_fallback"] == 1


def test_short_line_uses_frozen_conservative_fallback() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[30:40, 10:15] = 1
    mask[30:40, 17:22] = 1
    merged = line_gap_segmentation(mask, method="otsu", fallback_normalized_gap=0.35)
    assert len(merged.groups.groups) == 1
    assert merged.line_statistics["lines_using_fallback"] == 1

    mask[:, :] = 0
    mask[30:40, 10:15] = 1
    mask[30:40, 30:35] = 1
    split = line_gap_segmentation(mask, method="otsu", fallback_normalized_gap=0.35)
    assert len(split.groups.groups) == 2
    assert split.line_statistics["boundary_count"] == 1
