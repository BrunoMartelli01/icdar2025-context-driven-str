"""Coordinate, evaluation and mode boundaries of the direct YOLO crop experiment."""
import json

import numpy as np
import pytest

from scripts.evaluate_totaltext_yolo_direct_parseq import (DEFAULT_CONFIG, direct_crop,
                                                         metrics, validate_config)
from scripts.run_crop_group_word_selection import evaluate_predictions


def test_direct_crop_preserves_exact_original_rgb_pixels_and_no_padding():
    image = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
    crop = direct_crop(image, {"bbox": [2, 1, 5, 4]})
    np.testing.assert_array_equal(crop, image[1:4, 2:5])
    assert crop.shape == (3, 3, 3) and crop.flags.c_contiguous
    for box in ([2, 1, 2, 4], [-1, 0, 3, 4], [0, 0, 8, 4], [0., 0, 3, 4]):
        with pytest.raises(ValueError):
            direct_crop(image, {"bbox": box})


def test_duplicate_empty_and_symbol_only_readings_stay_in_precision_denominator():
    predictions = [{"bbox": [0, 0, 10, 10], "raw_text": text}
                   for text in ("Hello!", "HELLO", "", "$$$")]
    gt = [{"bbox": [0, 0, 10, 10], "text": "hello"}]
    score = evaluate_predictions(predictions, gt, 0.5)
    assert score["predictions"] == 4 and score["correct"] == 1
    assert metrics(score["correct"], score["predictions"], len(gt)) == {
        "correct": 1, "outputs": 4, "gt_words": 1,
        "precision": 0.25, "recall": 1., "hmean": 0.4}


def test_wrong_word_and_wrong_location_cannot_match():
    gt = [{"bbox": [0, 0, 10, 10], "text": "hello"}]
    predictions = [{"bbox": [0, 0, 10, 10], "raw_text": "bye"},
                   {"bbox": [20, 0, 30, 10], "raw_text": "hello"}]
    assert evaluate_predictions(predictions, gt, 0.5)["correct"] == 0


def test_scientific_config_rejects_strict_and_added_padding():
    config = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(config)
    for patch in ({"reproduction_mode": "strict"}, {"padding_pixels": 1}):
        with pytest.raises(ValueError):
            validate_config({**config, **patch})
