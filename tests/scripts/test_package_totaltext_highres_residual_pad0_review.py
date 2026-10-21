from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from scripts.package_totaltext_highres_residual_pad0_review import (
    classify_crop, matched_crop_to_word, render_image,
)


def test_crop_status_distinguishes_correct_ocr_and_bad_box() -> None:
    words = [{"bbox": [0, 0, 20, 20], "text": "Book"},
             {"bbox": [30, 0, 50, 20], "text": "Store"}]
    predictions = [{"bbox": [0, 0, 20, 20], "text": "book"},
                   {"bbox": [0, 0, 45, 20], "text": "Book"},
                   {"bbox": [30, 0, 50, 20], "text": "Wrong"}]
    matches = matched_crop_to_word(predictions, words)
    assert matches == {0: 0}
    assert classify_crop(predictions[0], words, matched_word_index=0)["status"] == "CORRECT"
    assert classify_crop(predictions[1], words, matched_word_index=None)["status"] == "TEXT_OK_BOX_BAD"
    assert classify_crop(predictions[2], words, matched_word_index=None)["status"] == "OCR_WRONG"


def test_single_composite_contains_all_crop_cards() -> None:
    rgb = np.full((80, 120, 3), 255, dtype=np.uint8)
    official = np.zeros((80, 120), dtype=np.uint8)
    target = np.zeros((256, 256), dtype=np.uint8)
    prediction_mask = np.zeros_like(target)
    proposals = [{"rank": rank + 1, "bbox": [rank * 10, 0, rank * 10 + 8, 20]}
                 for rank in range(5)]
    predictions = [{"bbox": item["bbox"], "text": str(index)}
                   for index, item in enumerate(proposals)]
    assessments = [{"status": "OCR_WRONG", "best_gt_iou": 0.1, "best_gt_text": "GT"}
                   for _ in proposals]
    png = render_image("total_text:1", rgb, official, target, prediction_mask,
                       proposals, predictions, [], {}, assessments)
    with Image.open(BytesIO(png)) as image:
        assert image.format == "PNG"
        assert image.width == 1400
        assert image.height == 445 + 2 * 186
