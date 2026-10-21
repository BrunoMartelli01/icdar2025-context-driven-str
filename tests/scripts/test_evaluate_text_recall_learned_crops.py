import inspect

import numpy as np
import pytest

from scripts.evaluate_text_recall_learned_crops import CONFIG, prepare_recipes, read_config
from segmentation.word_center_offsets import make_word_center_offsets


def test_learned_transfer_config_isolated_and_no_gt_inputs():
    cfg = read_config(CONFIG, "approximate")
    assert cfg["methods"] == ["kernel_watershed", "offset_votes_v2"]
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(CONFIG, "strict")
    assert list(inspect.signature(prepare_recipes).parameters) == [
        "probability", "kernel", "offsets", "canvas", "image_wh", "config", "threshold"]


def test_existing_kernel_and_coherent_votes_separate_same_fused_mask():
    cfg = read_config(CONFIG, "approximate")
    owners = np.zeros((64, 64), np.int32)
    owners[10:45, 5:25] = 1; owners[10:45, 25:45] = 2
    canvas = np.ones_like(owners, bool)
    offsets, _ = make_word_center_offsets(owners, canvas)
    kernel = np.zeros_like(owners, float); kernel[17:37, 9:20] = kernel[17:37, 30:40] = .9
    recipes = prepare_recipes((owners > 0).astype(float), kernel, offsets, canvas, (64, 64), cfg, threshold=.5)
    for recipe in recipes.values():
        assert len(recipe["proposals"]) == 2
        assert np.array_equal(recipe["labels"] > 0, owners > 0)


def test_missing_kernel_does_not_discard_unseeded_foreground():
    cfg = read_config(CONFIG, "approximate")
    p = np.zeros((32, 32)); p[3:12, 3:12] = .9; p[20:27, 20:27] = .9
    r = prepare_recipes(p, np.zeros_like(p), np.ones((2, 32, 32)), np.ones_like(p, bool), (32, 32), cfg, threshold=.5)
    assert len(r["kernel_watershed"]["proposals"]) == 2
    assert np.array_equal(r["kernel_watershed"]["labels"] > 0, p > 0)
