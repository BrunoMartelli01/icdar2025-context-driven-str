from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import label

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.totaltext_curved_box_v4 import (
    make_curved_word_boxes, pixel_guided_curved_polygon, smooth_curved_polygon,
)
from segmentation.totaltext_word_regions import make_word_regions


def _word(points: tuple[tuple[int, int], ...], text: str) -> SceneTextInstance:
    return SceneTextInstance(points, text, False, "word", "c")


def test_interpolated_curved_box_has_new_boundary_positions() -> None:
    points = ((5, 25), (35, 8), (65, 25), (65, 38), (35, 22), (5, 38))
    smooth = smooth_curved_polygon(points, (80, 50))
    assert len(smooth) > len(points)
    assert points[0] in smooth
    assert points[1] in smooth
    assert smooth != points
    seed = np.zeros((50, 80), dtype=bool)
    seed[20:24, 16:20] = True
    seed[16:20, 34:38] = True
    seed[23:27, 52:56] = True
    guided = pixel_guided_curved_polygon(points, (80, 50), seed)
    assert len(guided) > len(points)


def test_filled_curved_boxes_preserve_text_and_separate_words() -> None:
    words = (
        _word(((5, 25), (35, 8), (65, 25), (65, 38), (35, 22), (5, 38)), "ARC"),
        _word(((78, 15), (97, 8), (116, 15), (116, 34), (97, 27), (78, 34)), "TWO"),
    )
    base_mask, base_valid, base_sep, base_owner, _ = make_word_regions((125, 50), words)
    owner = np.asarray(base_owner)
    seed = np.zeros(owner.shape, dtype=np.uint8)
    for word_id in (1, 2):
        ys, xs = np.where(owner == word_id)
        seed[ys[::11], xs[::11]] = 255
    result, valid, separator, labels, audit = make_curved_word_boxes(
        (125, 50), words, base_owner, Image.fromarray(seed, mode="L"),
        base_sep, base_valid,
    )
    new = np.asarray(labels)
    assert audit.words == 2
    assert audit.changed_words >= 1
    assert audit.v1_source_seed_pixels == audit.retained_v1_source_seed_pixels
    assert set(np.unique(new)) == {0, 1, 2}
    assert np.array_equal(np.asarray(result) > 0, new > 0)
    assert np.all(new[seed > 0] == owner[seed > 0])
    for word_id in (1, 2):
        assert label(new == word_id, np.ones((3, 3), dtype=np.uint8))[1] == 1
    assert np.array_equal(np.asarray(valid), np.asarray(base_valid))
    assert not np.any((new > 0) & (np.asarray(separator) > 0))
    assert np.asarray(base_mask).shape == np.asarray(result).shape
