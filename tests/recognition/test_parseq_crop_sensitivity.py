from __future__ import annotations

from pathlib import Path

import numpy as np

from recognition.icdar13_annotations import ICDAR13RecognitionSample, ICDAR13Word
from recognition.parseq_crop_sensitivity import (
    LocalizationVariant,
    _variant_sample,
    build_variant_grid,
)


class FakeRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, crop: np.ndarray) -> dict[str, object]:
        self.calls += 1
        # The component crop contains the value used as a deterministic word fixture.
        return {"raw_text": "WORD" if int(crop.max()) == 20 else "noise"}


def _sample() -> ICDAR13RecognitionSample:
    return ICDAR13RecognitionSample(
        sample_id="1",
        image_path=Path("image.png"),
        image_sha256="a" * 64,
        annotation_path=Path("gt.txt"),
        annotation_sha256="b" * 64,
        image_size_wh=(12, 10),
        words=(ICDAR13Word((2, 2, 5, 4), "word"),),
    )


def test_variant_grid_is_deterministic_and_requires_baseline() -> None:
    variants = build_variant_grid((50, 150), (0, 6), (4, 8))
    assert variants[0].variant_id == "amin-50__pad-0__conn-4"
    assert variants[-1].variant_id == "amin-150__pad-6__conn-8"
    assert variants[-1].is_baseline is True


def test_variant_sample_matches_geometry_before_measuring_text() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[2:5, 2:6] = 20
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 2:6] = 1
    recognizer = FakeRecognizer()
    cache: dict = {}

    result = _variant_sample(
        sample=_sample(),
        image_rgb=image,
        mask=mask,
        variant=LocalizationVariant(1, 0, 8),
        recognize_crop=recognizer,
        prediction_cache=cache,
    )

    assert recognizer.calls == 1
    assert result["crops"][0]["padded_bbox_xyxy_exclusive"] == [2, 2, 6, 5]
    gt = result["ground_truth_spatial_summaries"][0]
    assert gt["bbox_xyxy_exclusive_diagnostic"] == [2.0, 2.0, 6.0, 5.0]
    assert gt["best_spatial_crop"]["bbox_iou"] == 1.0
    assert gt["best_spatial_crop"]["casefold_exact"] is True

    # Re-evaluating the same source-coordinate box reuses the PARSeq prediction.
    _variant_sample(
        sample=_sample(),
        image_rgb=image,
        mask=mask,
        variant=LocalizationVariant(1, 0, 8),
        recognize_crop=recognizer,
        prediction_cache=cache,
    )
    assert recognizer.calls == 1


def test_padding_changes_geometry_but_not_paper_fixed_component_area() -> None:
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[2:5, 2:6] = 20
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 2:6] = 1
    result = _variant_sample(
        sample=_sample(),
        image_rgb=image,
        mask=mask,
        variant=LocalizationVariant(10, 2, 4),
        recognize_crop=FakeRecognizer(),
        prediction_cache={},
    )

    crop = result["crops"][0]
    assert crop["component_area"] == 12
    assert crop["bbox_xyxy_exclusive"] == [2, 2, 6, 5]
    assert crop["padded_bbox_xyxy_exclusive"] == [0, 0, 8, 7]
    assert result["ground_truth_spatial_summaries"][0]["best_spatial_crop"][
        "gt_coverage"
    ] == 1.0
