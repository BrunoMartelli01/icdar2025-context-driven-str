from __future__ import annotations

import numpy as np

from recognition.segmentation_target_component_topology import (
    _components,
    level_word_metrics,
    map_original_box_to_mask,
    match_target_prediction_components,
)


def test_original_box_mapping_is_coverage_preserving_and_clamped() -> None:
    assert map_original_box_to_mask(
        (9, 9, 10, 10), image_size_wh=(11, 11), mask_shape=(4, 4)
    ) == (3, 3, 4, 4)
    assert map_original_box_to_mask(
        (0, 0, 11, 11), image_size_wh=(11, 11), mask_shape=(4, 4)
    ) == (0, 0, 4, 4)


def test_component_matching_reports_split_and_pixel_purity() -> None:
    target = np.zeros((8, 8), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[2:6, 2:6] = 1
    prediction[2:6, 2:3] = 1
    prediction[2:6, 5:6] = 1
    target_components = _components(target)
    prediction_components = _components(prediction)
    _, target_ids, region = level_word_metrics(
        mask=target, components=target_components,
        native_or_source_box=(2, 2, 6, 6), transcription="TEST",
    )
    _, prediction_ids, _ = level_word_metrics(
        mask=prediction, components=prediction_components,
        native_or_source_box=(2, 2, 6, 6), transcription="TEST",
    )
    result = match_target_prediction_components(
        target_components=target_components,
        prediction_components=prediction_components,
        target_ids=target_ids,
        prediction_ids=prediction_ids,
        region=region,
    )
    assert [row["fraction_pixels_inside_target_foreground"] for row in result["predicted_components"]] == [1.0, 1.0]
    assert len(result["split_target_components"]) == 1
    assert len(result["merge_predicted_components"]) == 0
