from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.frozen_segmentation import (
    FrozenAuthorProtocolSegmenter,
    FrozenSegmentationConfig,
    backproject_binary_mask,
)


ROOT = Path(__file__).parents[2]


def test_binary_backprojection_is_nearest_and_preserves_binary_values() -> None:
    native = np.zeros((128, 128), dtype=np.uint8)
    native[:64, :64] = 1

    projected = backproject_binary_mask(native, (7, 11))

    assert projected.shape == (7, 11)
    assert projected.dtype == np.uint8
    assert set(np.unique(projected).tolist()) == {0, 1}
    assert np.all(projected[:4, :6] == 1)
    assert np.all(projected[4:, :] == 0)
    assert np.all(projected[:, 6:] == 0)


@pytest.mark.parametrize(
    "mask,message",
    [
        (np.zeros((127, 128), dtype=np.uint8), "128x128"),
        (np.zeros((128, 128), dtype=np.float32), "binary uint8"),
        (np.full((128, 128), 2, dtype=np.uint8), "binary uint8"),
    ],
)
def test_binary_backprojection_rejects_non_frozen_inputs(mask: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        backproject_binary_mask(mask, (10, 10))


def test_downstream_adapter_rejects_validity_filtered_evaluation_config() -> None:
    with pytest.raises(ValueError, match=r"evaluation\.(dont_care|valid_mask)"):
        FrozenAuthorProtocolSegmenter(
            FrozenSegmentationConfig(
                evaluation_config=(
                    ROOT
                    / "configs/approximate/segmentation_icdar13_author_protocol_test_task1_ignore.yaml"
                ),
                checkpoint=Path("not-reached.pt"),
                device="cpu",
            )
        )
