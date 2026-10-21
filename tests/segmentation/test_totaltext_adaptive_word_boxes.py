from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from segmentation.totaltext_adaptive_word_boxes import fit_adaptive_word_boxes


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array, mode="I;16" if array.dtype == np.uint16 else "L")


def test_horizontal_word_is_filled_box_not_dilated_letters() -> None:
    owner = np.zeros((36, 100), dtype=np.uint16)
    owner[5:30, 5:95] = 1
    pixels = np.zeros_like(owner, dtype=np.uint8)
    for x in (10, 32, 54, 76):
        pixels[12:20, x : x + 6] = 255
    target, labels, audit = fit_adaptive_word_boxes(
        _image(owner), _image(pixels), Image.new("L", (100, 36), 255)
    )
    result = np.asarray(target) > 0
    assert audit.changed_words == 1
    assert audit.mean_bins_per_changed_word >= 4
    assert audit.source_seed_pixels == audit.retained_source_seed_pixels
    assert np.all(result[pixels > 0])
    assert np.all(result <= (owner > 0))
    assert result[17, 24]  # the empty inter-character span is filled
    assert result.sum() >= 0.65 * (owner > 0).sum()
    assert np.array_equal(np.asarray(labels) > 0, result)


def test_two_words_remain_disjoint_and_ignore_is_unchanged() -> None:
    owner = np.zeros((60, 110), dtype=np.uint16)
    owner[5:25, 5:45] = 1
    owner[35:55, 60:105] = 2
    pixels = np.zeros_like(owner, dtype=np.uint8)
    pixels[12:19, 10:15] = 255
    pixels[12:19, 30:35] = 255
    pixels[42:49, 65:70] = 255
    pixels[42:49, 90:95] = 255
    valid = np.full(owner.shape, 255, dtype=np.uint8)
    valid[0, :] = 0
    target, labels, audit = fit_adaptive_word_boxes(
        _image(owner), _image(pixels), _image(valid)
    )
    mapped = np.asarray(labels)
    assert audit.words == 2
    assert set(np.unique(mapped)) == {0, 1, 2}
    assert np.all((mapped > 0) <= (owner > 0))
    assert np.all(mapped[valid == 0] == 0)
    assert np.array_equal(np.asarray(target) > 0, mapped > 0)


def test_vertical_word_uses_transposed_long_axis_and_keeps_seed() -> None:
    owner = np.zeros((100, 38), dtype=np.uint16)
    owner[5:95, 6:32] = 1
    pixels = np.zeros_like(owner, dtype=np.uint8)
    for y in (10, 32, 54, 76):
        pixels[y : y + 6, 14:22] = 255
    target, _, audit = fit_adaptive_word_boxes(
        _image(owner), _image(pixels), Image.new("L", (38, 100), 255)
    )
    result = np.asarray(target) > 0
    assert audit.changed_words == 1
    assert np.all(result[pixels > 0])
    assert result[24, 18]


def test_missing_seed_falls_back_and_bad_geometry_fails_closed() -> None:
    owner = np.zeros((20, 60), dtype=np.uint16)
    owner[5:15, 5:55] = 1
    zero = np.zeros(owner.shape, dtype=np.uint8)
    _, labels, audit = fit_adaptive_word_boxes(
        _image(owner), _image(zero), Image.new("L", (60, 20), 255)
    )
    assert audit.no_seed_fallbacks == 1
    assert np.array_equal(np.asarray(labels), owner)
    with pytest.raises(ValueError, match="geometry must match"):
        fit_adaptive_word_boxes(
            _image(owner), Image.new("L", (30, 20)), Image.new("L", (60, 20), 255)
        )
