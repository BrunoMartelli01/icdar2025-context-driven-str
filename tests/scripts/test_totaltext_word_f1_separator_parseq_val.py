from __future__ import annotations

import numpy as np
import pytest
import torch

from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.evaluate_totaltext_word_f1_separator_parseq_val import (
    _check_ocr, native_word_mask,
)


def test_native_word_mask_control_and_separator() -> None:
    canvas = np.ones((256, 256), dtype=bool)
    canvas[0, 0] = False
    foreground = torch.full((256, 256), -10.0)
    foreground[0, 0] = 10.0
    foreground[4, 5] = 10.0
    foreground[4, 6] = 10.0
    separator = torch.full((256, 256), -10.0)
    separator[4, 6] = 10.0

    control = native_word_mask(foreground.unsqueeze(0), canvas, separator_head=False)
    two_head = native_word_mask(torch.stack((foreground, separator)), canvas,
                                separator_head=True)
    assert control.dtype == np.uint8
    assert int(control.sum()) == 2
    assert int(two_head.sum()) == 1
    assert two_head[4, 5] == 1
    assert two_head[4, 6] == 0


def test_native_word_mask_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="native 256x256"):
        native_word_mask(torch.zeros((1, 128, 128)), np.ones((256, 256)),
                         separator_head=False)


def test_reused_ocr_requires_frozen_model_provenance() -> None:
    value = {"crop_rgb_sha256": "abc", "raw_text": "TEST",
             "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
             "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256}
    assert _check_ocr(value, "abc") == "TEST"
    with pytest.raises(ValueError, match="cache provenance"):
        _check_ocr(value, "different")
