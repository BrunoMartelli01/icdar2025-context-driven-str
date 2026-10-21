from __future__ import annotations

import numpy as np

from recognition.segmentation_word_topology import (
    _components,
    _word_topology,
    recover_native_mask_from_projected,
)


def test_recover_native_mask_handles_nonintegral_projection() -> None:
    native = np.zeros((128, 128), dtype=np.uint8)
    native[10:20, 30:45] = 1
    # Simulate nearest projection using the same interval convention as the
    # frozen adapter (the output is deliberately non-integral in both axes).
    ys = np.floor(np.arange(257) * 128 / 257).astype(int)
    xs = np.floor(np.arange(193) * 128 / 193).astype(int)
    projected = native[np.ix_(ys, xs)]
    recovered = recover_native_mask_from_projected(projected)
    assert np.array_equal(recovered, native)


def test_word_topology_reports_components_and_recall() -> None:
    target = np.zeros((128, 128), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[20:30, 20:50] = 1
    prediction[20:25, 20:30] = 1
    prediction[20:25, 35:45] = 1
    topology, rows = _word_topology(
        word_index=1,
        transcription="TEST",
        gt_box_original=(20, 20, 50, 30),
        native_box=(20, 20, 50, 30),
        target=target,
        prediction=prediction,
        target_components=_components(target),
        prediction_components=_components(prediction),
        image_size_wh=(128, 128),
    )
    assert topology["target_component_count_intersecting_word"] == 1
    assert topology["prediction_component_count_intersecting_word"] == 2
    assert topology["word_recall_from_prediction_pixels"] == 1 / 3
    assert len(rows) == 2
    assert all(row["box_purity"] == 1.0 for row in rows)
