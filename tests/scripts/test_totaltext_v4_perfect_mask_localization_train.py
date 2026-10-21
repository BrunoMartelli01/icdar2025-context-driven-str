"""Accounting tests for the TRAIN perfect-target localization ceiling."""

import numpy as np

from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from scripts.diagnose_totaltext_v4_perfect_mask_localization_train import (
    _sample_geometry,
    _stage_boxes,
)


def test_stage_boxes_reconstruct_frozen_filter_padding_sort_and_cap():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:20, 10:30] = 1  # native bbox area 200: retained
    mask[40:44, 50:54] = 1  # native bbox area 16: rejected
    params = RepositoryContourParams(min_area=150, pad=6, nmax=1)
    audit = audit_repository_contours(mask, params)
    stages = _stage_boxes(audit, image_hw=(256, 256), min_area=150)
    assert len(stages["raw"]) == 2
    assert len(stages["area_pass"]) == 1
    assert stages["retained"] == [[8, 8, 72, 52]]


def test_stage_geometry_attributes_area_and_top10_losses_separately():
    words = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]]
    stages = {
        "raw": [words[0], words[1], words[2]],
        "area_pass": [words[0], words[2]],
        "retained": [words[0]],
    }
    result = _sample_geometry(words, stages)
    assert result["losses"] == {
        "no_raw_iou_0_5": 1,
        "area_filter": 1,
        "top10": 1,
        "retained_but_competition": 0,
        "retained_iou_0_5": 1,
    }
    assert result["one_to_one"]["retained"]["0.5"] == 1


def test_one_to_one_competition_not_confused_with_missing_target():
    words = [[0, 0, 10, 10], [0, 0, 10, 10]]
    stages = {name: [[0, 0, 10, 10]] for name in ("raw", "area_pass", "retained")}
    result = _sample_geometry(words, stages)
    assert result["losses"]["retained_iou_0_5"] == 2
    assert result["losses"]["retained_but_competition"] == 1


def test_tight_control_separates_area_filter_from_top10():
    words = [[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10], [60, 0, 70, 10]]
    stages = {
        "raw": [words[0]], "area_pass": [words[0]], "retained": [words[0]],
        "raw_tight_diagnostic_only": [words[0], words[1], words[2]],
        "area_pass_tight_diagnostic_only": [words[0], words[2]],
        "retained_tight_diagnostic_only": [words[0]],
    }
    result = _sample_geometry(words, stages)
    assert result["tight_losses"] == {
        "no_raw_tight_iou_0_5": 1,
        "area_filter_tight": 1,
        "top10_tight": 1,
        "retained_tight_iou_0_5": 1,
    }
