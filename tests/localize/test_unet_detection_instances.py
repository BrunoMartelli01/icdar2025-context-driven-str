from copy import deepcopy

import numpy as np
import pytest

from localize.unet_detection_methods import detection_instances, rectified_word_crop
from localize.word_multitask import parseq_word_proposals, word_instances
from scripts.run_unet_detection_methods import DEFAULT_CONFIG, read_config


@pytest.fixture
def cfg(): return read_config(DEFAULT_CONFIG, "approximate")[0]


def test_progressive_expansion_preserves_two_words_and_unseeded_fallback(cfg):
    p = np.zeros((4, 48, 80), np.float32)
    p[0, 5:20, 5:65] = .9  # Adjacent words share full-text support.
    p[0, 30:35, 70:75] = .9  # A distant unseeded word must survive.
    p[1, 10:14, 10:16] = .9; p[1, 10:14, 48:54] = .9
    p[2, 8:17, 8:22] = .9; p[2, 8:17, 44:60] = .9
    p[3, 6:19, 6:64] = .9
    labels, audit = detection_instances(p, np.ones(p.shape[1:], bool), cfg)
    assert np.array_equal(labels > 0, p[0] >= .65)
    assert labels[12, 12] != labels[12, 50] and labels[12, 12] > 0
    assert audit["seed_components"] == 2 and audit["unseeded_fallback_components"] == 1
    assert len(np.unique(labels[labels > 0])) == 3


def test_two_head_control_exactly_reuses_existing_watershed(cfg):
    p = np.random.default_rng(12).random((2, 32, 32)).astype(np.float32)
    canvas = np.ones((32, 32), bool)
    a = detection_instances(p, canvas, cfg)
    b = word_instances(p, canvas, cfg["postprocessing"])
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_equal_initial_kernel_scales_preserve_historical_instance_labels(cfg):
    p = np.random.default_rng(21).random((2, 32, 32)).astype(np.float32)
    canvas = np.ones((32, 32), bool)
    expanded = np.concatenate((p, np.repeat(p[1:2], 2, axis=0)), axis=0)
    assert np.array_equal(detection_instances(expanded, canvas, cfg)[0], word_instances(p, canvas, cfg["postprocessing"])[0])


def test_outside_canvas_and_invalid_probabilities(cfg):
    p = np.ones((4, 32, 32), np.float32); canvas = np.zeros((32, 32), bool)
    assert not detection_instances(p, canvas, cfg)[0].any()
    p[2, 0, 0] = np.nan
    with pytest.raises(ValueError): detection_instances(p, canvas, cfg)


def curve():
    labels = np.zeros((96, 96), np.int32)
    for x in range(15, 78):
        y = int(35 + 9 * np.sin((x - 15) / 63 * np.pi))
        labels[y:y + 9, x] = 1
    rgb = np.stack(np.meshgrid(np.arange(192, dtype=np.uint8), np.arange(192, dtype=np.uint8)) + [np.full((192, 192), 90, np.uint8)], axis=-1)
    return labels, rgb


@pytest.mark.parametrize("view", ["axis", "oriented", "bezier"])
def test_rectification_retains_original_matching_box_and_uses_assigned_support(cfg, view):
    labels, rgb = curve(); p = parseq_word_proposals(labels, (192, 192), padding=0, min_area=0, max_instances=None)[0]
    original = deepcopy(p)
    crop, meta = rectified_word_crop(rgb, labels, p, view, cfg)
    assert p == original and crop.ndim == 3 and min(crop.shape[:2]) > 0
    if view != "axis":
        assert meta["reason"] == "applied" and meta["spec"]["encloses_assigned_foreground"]
    labels[3:10, 3:10] = 2  # Unassigned text cannot change this crop.
    assert np.array_equal(crop, rectified_word_crop(rgb, labels, p, view, cfg)[0])


def test_invalid_fit_has_exact_axis_fallback_with_reason(cfg):
    labels, rgb = curve(); p = parseq_word_proposals(labels, (192, 192), padding=0, min_area=0, max_instances=None)[0]
    cfg["rectification"]["minimum_pixels"] = 100000
    axis = rectified_word_crop(rgb, labels, p, "axis", cfg)[0]
    crop, meta = rectified_word_crop(rgb, labels, p, "bezier", cfg)
    assert np.array_equal(crop, axis) and meta["reason"] == "insufficient_foreground" and meta["fallback"] == "axis"
