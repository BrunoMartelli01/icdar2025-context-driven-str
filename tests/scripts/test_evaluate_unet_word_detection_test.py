from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.evaluate_text_recall_refinements import read_config as source_config, CONFIG as SOURCE
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.evaluate_unet_word_detection_test import (
    CONFIG, aggregate, image_tensor, localize, read_config, score_detection,
)


def test_preprocessing_replays_qualified_letterbox_and_supports_native384():
    _, preprocessing = source_config(SOURCE, "approximate")
    rgb = np.arange(45 * 81 * 3, dtype=np.uint8).reshape(45, 81, 3)
    expected = _tensor(rgb, _transform(preprocessing, "letterbox", train=False))
    torch.testing.assert_close(image_tensor(rgb, 512, preprocessing), expected, rtol=0, atol=0)
    tensor = image_tensor(rgb, 768, preprocessing)
    assert tensor.shape == (1, 3, 768, 768) and tensor.dtype == torch.float32


def test_localization_splits_touching_words_and_drops_unseeded_component():
    _, source, _ = read_config(CONFIG, "approximate")
    maps = np.zeros((2, 64, 64), np.float32)
    maps[0, 10:30, 5:45] = .95
    maps[1, 15:25, 10:20] = maps[1, 15:25, 30:40] = .8
    maps[0, 40:50, 5:15] = .95  # Isolated foreground with no kernel.
    labels, proposals, audit = localize(maps, np.ones((64, 64), bool), (128, 128), source, .9)
    assert len(proposals) == 2
    assert not labels[40:50].any()
    assert len(audit["seeded_only"]["removed_instances"]) == 1
    assert not np.any(labels[maps[0] < .9])


def test_ignore_only_wholly_invalid_instances_and_match_one_to_one():
    labels = np.zeros((12, 12), np.int32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2
    labels[1:4, 8:11] = 3
    valid = np.ones_like(labels, bool)
    valid[6:9, 6:9] = False
    valid[1:4, 1:3] = False  # Partial ignore keeps the entire first box.
    proposals = [{"instance_id": 1, "bbox": [0, 0, 10, 10]},
                 {"instance_id": 2, "bbox": [20, 0, 30, 10]},
                 {"instance_id": 3, "bbox": [0, 0, 10, 10]}]
    gt = [{"bbox": [0, 0, 10, 10]}, {"bbox": [20, 0, 30, 10]}]
    r = score_detection(labels, proposals, valid, gt, .5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 0)
    assert r["ignored_prediction_ids"] == [2]


def test_matching_includes_exact_iou_half_and_empty_predictions():
    labels = np.ones((2, 2), np.int32)
    valid = np.ones_like(labels, bool)
    gt = [{"bbox": [0, 0, 10, 10]}]
    proposal = [{"instance_id": 1, "bbox": [0, 0, 5, 10]}]
    assert score_detection(labels, proposal, valid, gt, .5)["tp"] == 1
    assert score_detection(labels, proposal, valid, gt, .500001)["tp"] == 0
    r = score_detection(labels * 0, [], valid, gt, .5)
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 1)


def test_aggregation_denominators_failures_duplicates_and_empty_policy():
    rows = [{"sample_id": "a", "status": "success", "tp": 2, "tn": 0,
             "fp": 1, "fn": 1, "gt_words": 3, "care_proposals": 3},
            {"sample_id": "b", "status": "success", "tp": 0, "tn": 0,
             "fp": 2, "fn": 2, "gt_words": 2, "care_proposals": 2}]
    result = aggregate(rows, 2, 5)
    assert result["word_f1"] == .4
    assert result["accuracy_tn0"] == .25
    with pytest.raises(ValueError, match="denominator"):
        aggregate(rows, 2, 6)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        aggregate(rows[:1], 2)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        aggregate([rows[0], rows[0]], 2)
    failed = deepcopy(rows)
    failed[1]["status"] = "failed"
    with pytest.raises(ValueError, match="Failed sample"):
        aggregate(failed, 2)
    empty = {"sample_id": "a", "status": "success", **{k: 0 for k in (
        "tp", "tn", "fp", "fn", "gt_words", "care_proposals")}}
    assert aggregate([empty], 1)["accuracy_tn0"] is None


def test_configuration_cannot_enter_strict_or_import_namespace_defaults(tmp_path):
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(CONFIG, "strict")
    external = tmp_path / "configuration.json"
    external.write_bytes(CONFIG.read_bytes())
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(external, "approximate")
