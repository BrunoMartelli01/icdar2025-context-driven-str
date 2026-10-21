from __future__ import annotations

import numpy as np
import pytest
import torch

from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.evaluate_totaltext_frequency_gated_skip_parseq_val import (
    _ocr_text, native_prediction,
)


def test_native_probability_and_letterbox_canvas() -> None:
    logits = torch.zeros((1, 1, 256, 256))
    logits[0, 0, 4, 4] = 2.0
    logits[0, 0, 4, 5] = 2.0
    canvas = np.ones((256, 256), dtype=bool)
    canvas[4, 5] = False
    mask = native_prediction(logits, canvas)
    assert mask.dtype == np.uint8
    assert mask[4, 4] == 1
    assert mask[4, 5] == 0
    assert mask[0, 0] == 0


def test_native_prediction_rejects_resampled_shape() -> None:
    with pytest.raises(ValueError, match="native 256x256"):
        native_prediction(torch.zeros((1, 1, 128, 128)), np.ones((256, 256)))


def test_reused_parseq_requires_exact_artifact_hashes() -> None:
    record = {"crop_rgb_sha256": "crop", "raw_text": "WORD",
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256}
    assert _ocr_text(record, "crop") == "WORD"
    with pytest.raises(ValueError, match="cache mismatch"):
        _ocr_text(record, "another")
