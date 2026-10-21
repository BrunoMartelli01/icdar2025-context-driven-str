from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_totaltext_train_padding_sweep as sweep


def test_padding_changes_only_boxes_not_raw_mask_or_contour_count():
    native = np.zeros((128, 128), dtype=np.uint8)
    native[10:13, 20:24] = 1
    result = sweep.proposals_for_mask(native, (256, 512), [0, 6])
    assert result[0]["raw_contours"] == result[6]["raw_contours"] == 1
    assert result[0]["candidates_before_cap"] == result[6]["candidates_before_cap"] == 1
    assert result[0]["proposals"][0]["bbox_original"] == [80, 20, 96, 26]
    assert result[6]["proposals"][0]["bbox_original"] == [56, 8, 120, 38]


def test_no_area_filter_still_enforces_top10():
    native = np.zeros((128, 128), dtype=np.uint8)
    for index in range(12):
        native[10, 4 + index * 10] = 1
    result = sweep.proposals_for_mask(native, (128, 128), [0])
    assert result[0]["raw_contours"] == 12
    assert result[0]["candidates_before_cap"] == 12
    assert len(result[0]["proposals"]) == 10


def test_test_manifest_fails_before_loading_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sweep, "load_segmentation_manifest",
        lambda *_args, **_kwargs: SimpleNamespace(
            dataset="total_text", split_role="test", samples=[object()] * 1255,
        ),
    )
    with pytest.raises(ValueError, match="TRAIN-1255"):
        sweep._context(SimpleNamespace(train_manifest=tmp_path / "test.yaml"))
