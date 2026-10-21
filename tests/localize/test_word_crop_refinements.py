import inspect

import numpy as np
import pytest

from localize.word_crop_refinements import bounded_growth, filter_fallbacks, prepare_recipes, resize_probability
from scripts.evaluate_text_recall_refinements import CONFIG, read_config


def test_probability_resize_preserves_constant_and_rejects_nonprobabilities():
    p = np.full((4, 4), .3, np.float32)
    np.testing.assert_allclose(resize_probability(p, 6), .3, atol=1e-7)
    with pytest.raises(ValueError):
        resize_probability(p * np.nan, 6)


def test_growth_preserves_word_ids_separates_touching_words_and_respects_radius():
    p = np.full((24, 24), .6, np.float32)
    seeds = np.zeros_like(p, np.int32)
    seeds[8:12, 5:8] = 1; seeds[8:12, 13:16] = 2
    canvas = np.ones_like(p, bool)
    grown, meta = bounded_growth(p, seeds, canvas, low_threshold=.5, radius=3.)
    assert set(np.unique(grown)) == {0, 1, 2}
    np.testing.assert_array_equal(grown[seeds > 0], seeds[seeds > 0])
    assert meta["added_pixels"] > 0
    assert not grown[0].any()
    assert grown[9, 10] and grown[9, 11] and grown[9, 10] != grown[9, 11]


def test_growth_cannot_cross_background_or_canvas_and_does_not_invent_empty_words():
    p = np.zeros((16, 16), np.float32); p[5:10, 2:6] = .9; p[5:10, 7:10] = .6
    labels = np.zeros_like(p, np.int32); labels[5:10, 2:6] = 7
    canvas = np.ones_like(p, bool); canvas[0:4] = False
    grown, _ = bounded_growth(p, labels, canvas, low_threshold=.5, radius=4.)
    assert not grown[:, 7:].any()
    empty, _ = bounded_growth(p, labels * 0, canvas, low_threshold=.5, radius=4.)
    assert not empty.any()
    with pytest.raises(ValueError):
        bounded_growth(p, labels, canvas, low_threshold=.95, radius=4.)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fallback_filter_keeps_small_seeded_words_and_exact_threshold_boundary(dtype):
    labels = np.zeros((8, 8), np.int32)
    labels[1, 1] = 1; labels[3:5, 3:5] = 2; labels[6, 6] = 3
    kernel = np.zeros_like(labels, dtype=dtype); kernel[1, 1] = .5; kernel[3, 3] = .35; kernel[6, 6] = .349
    soft, meta = filter_fallbacks(labels, kernel, seed_threshold=.5, fallback_threshold=.35)
    assert soft[1, 1] == 1 and soft[3, 3] == 2 and soft[6, 6] == 0
    assert len(meta["removed_instances"]) == 1
    seeded, _ = filter_fallbacks(labels, kernel, seed_threshold=.5, fallback_threshold=.5)
    assert set(np.unique(seeded)) == {0, 1}


def test_recipes_split_touching_words_without_gt_and_keep_baseline_foreground():
    cfg, _ = read_config(CONFIG, "approximate")
    p = np.zeros((2, 64, 64), np.float32)
    p[0, 10:40, 5:45] = .95
    p[1, 15:35, 10:20] = p[1, 15:35, 30:40] = .9
    recipes = prepare_recipes(p, np.ones((64, 64), bool), (128, 64), cfg, text_threshold=.9)
    assert len(recipes["baseline"]["proposals"]) == 2
    assert len(recipes["refined"]["proposals"]) == 2
    np.testing.assert_array_equal(recipes["baseline"]["labels"] > 0, p[0] > 0)
    assert list(inspect.signature(prepare_recipes).parameters) == [
        "probabilities", "canvas", "image_wh", "config", "text_threshold"]
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(CONFIG, "strict")
