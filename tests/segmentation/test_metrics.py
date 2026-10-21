from __future__ import annotations

import pytest
import torch

from segmentation.metrics import (
    binarize_probabilities,
    foreground_iou_f1_global_micro,
    foreground_iou_f1_per_image_macro,
)


def test_hand_calculated_micro_iou_and_f1() -> None:
    prediction = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.bool)
    target = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.bool)
    result = foreground_iou_f1_global_micro(
        prediction, target, empty_foreground_policy="perfect_if_both_empty"
    )
    assert result.foreground_iou == pytest.approx(1 / 3)
    assert result.foreground_f1 == pytest.approx(1 / 2)
    assert result.intersection_pixels == 1
    assert result.union_pixels == 3


def test_macro_and_micro_are_both_named_and_differ() -> None:
    prediction = torch.zeros((2, 1, 2, 3), dtype=torch.bool)
    target = torch.zeros_like(prediction)
    prediction[0, 0, 0, 0] = True
    target[0, 0, 0, 0] = True
    prediction[1, 0, :, :] = True
    target[1, 0, 0, 0] = True
    macro = foreground_iou_f1_per_image_macro(
        prediction, target, empty_foreground_policy="perfect_if_both_empty"
    )
    micro = foreground_iou_f1_global_micro(
        prediction, target, empty_foreground_policy="perfect_if_both_empty"
    )
    assert macro.aggregation_mode == "per_image_macro"
    assert micro.aggregation_mode == "global_micro"
    assert macro.foreground_iou == pytest.approx((1 + 1 / 6) / 2)
    assert micro.foreground_iou == pytest.approx(2 / 7)


@pytest.mark.parametrize(
    ("policy", "expected"),
    (("perfect_if_both_empty", 1.0), ("zero_if_both_empty", 0.0)),
)
def test_empty_policies_are_explicit(policy: str, expected: float) -> None:
    empty = torch.zeros((2, 1, 3, 3), dtype=torch.bool)
    result = foreground_iou_f1_per_image_macro(
        empty, empty, empty_foreground_policy=policy  # type: ignore[arg-type]
    )
    assert result.foreground_iou == expected
    assert result.foreground_f1 == expected
    assert result.empty_sample_count == 2
    with pytest.raises(ValueError, match="No non-empty"):
        foreground_iou_f1_global_micro(
            empty, empty, empty_foreground_policy="exclude_both_empty"
        )


def test_valid_mask_excludes_pixels_and_threshold_is_inclusive() -> None:
    probabilities = torch.tensor([[[[0.49, 0.50], [0.9, 0.1]]]])
    prediction = binarize_probabilities(probabilities, threshold=0.5)
    assert prediction.tolist() == [[[[False, True], [True, False]]]]
    target = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.bool)
    valid = torch.tensor([[[[1, 1], [0, 1]]]], dtype=torch.bool)
    result = foreground_iou_f1_global_micro(
        prediction,
        target,
        valid_mask=valid,
        empty_foreground_policy="perfect_if_both_empty",
    )
    assert result.foreground_iou == 1.0
    assert result.foreground_f1 == 1.0


def test_nonbinary_metric_input_fails() -> None:
    with pytest.raises(ValueError, match="must be binary"):
        foreground_iou_f1_global_micro(
            torch.tensor([[[[0.2]]]]),
            torch.tensor([[[[1.0]]]]),
            empty_foreground_policy="perfect_if_both_empty",
        )
