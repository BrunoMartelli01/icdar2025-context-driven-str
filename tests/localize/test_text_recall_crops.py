import inspect

import numpy as np
import pytest

from localize.text_recall_crops import distance_partition, prepare_recipes
from scripts.evaluate_text_recall_crops import CONFIG, read_config
from segmentation.word_center_offsets import make_word_center_offsets


def test_distance_watershed_separates_two_lobes_and_retains_bridge_pixels():
    mask = np.zeros((48, 64), bool)
    mask[9:35, 5:27] = mask[9:35, 37:59] = True
    mask[21:23, 27:37] = True
    cfg = read_config(CONFIG, "approximate")
    labels, audit = distance_partition(mask, cfg["distance"])
    assert audit["base_components"] == 1 and audit["markers"] == 2
    assert labels[20, 15] != labels[20, 45]
    assert np.array_equal(labels > 0, mask)


def test_distance_watershed_single_rectangle_and_tiny_fallback():
    cfg = read_config(CONFIG, "approximate")
    mask = np.zeros((40, 64), bool); mask[12:25, 4:58] = True
    labels, audit = distance_partition(mask, cfg["distance"])
    assert audit["markers"] == 1
    assert np.array_equal(labels > 0, mask)
    mask[:] = False; mask[0, 0] = True
    labels, audit = distance_partition(mask, cfg["distance"])
    assert audit["markers"] == 1 and labels[0, 0] == 1


def test_distance_empty_and_deterministic_plateaus():
    cfg = read_config(CONFIG, "approximate")
    mask = np.zeros((32, 32), bool)
    assert not distance_partition(mask, cfg["distance"])[0].any()
    mask[3:29, 3:29] = True
    first = distance_partition(mask, cfg["distance"])[0]
    assert np.array_equal(first, distance_partition(mask, cfg["distance"])[0])


def test_learned_offset_cues_separate_flat_fused_support_shape_cannot_separate():
    cfg = read_config(CONFIG, "approximate")
    owners = np.zeros((64, 64), np.int32)
    owners[10:45, 5:25] = 1; owners[10:45, 25:45] = 2
    canvas = np.ones_like(owners, bool)
    offsets, _ = make_word_center_offsets(owners, canvas)
    recipes = prepare_recipes((owners > 0).astype(float), offsets, canvas, (128, 64), cfg, threshold=.5)
    assert len(recipes["findcontours"]["proposals"]) == 1
    assert len(recipes["offset_watershed"]["proposals"]) == 2
    for key in ("connected_components", "distance_watershed", "offset_watershed"):
        assert np.array_equal(recipes[key]["labels"] > 0, owners > 0)


def test_inference_boundary_has_no_gt_inputs_and_crops_use_true_pixel_area():
    assert list(inspect.signature(prepare_recipes).parameters) == [
        "probability", "offsets", "canvas", "image_wh", "config", "threshold"]
    cfg = read_config(CONFIG, "approximate")
    p = np.zeros((32, 32)); p[5:20, 5:20] = .9; p[8:17, 8:17] = 0
    recipes = prepare_recipes(p, np.zeros((2, 32, 32)), np.ones((32, 32), bool), (64, 32), cfg, threshold=.5)
    proposal = recipes["findcontours"]["proposals"][0]
    assert proposal["foreground_area"] == 15*15-9*9
    assert proposal["bbox_native"] == [5, 5, 20, 20]
    assert proposal["bbox"] == [10, 0, 40, 24]


@pytest.mark.parametrize("kind", ["probability", "offset", "canvas", "threshold"])
def test_invalid_inputs_fail_before_localization(kind):
    cfg = read_config(CONFIG, "approximate")
    p = np.zeros((16, 16)); offsets = np.zeros((2, 16, 16)); canvas = np.ones((16, 16), bool); threshold = .5
    if kind == "probability": p[0, 0] = np.nan
    if kind == "offset": offsets[0, 0, 0] = 2
    if kind == "canvas": canvas = canvas.astype(np.uint8)
    if kind == "threshold": threshold = 1
    with pytest.raises(ValueError): prepare_recipes(p, offsets, canvas, (16, 16), cfg, threshold=threshold)
