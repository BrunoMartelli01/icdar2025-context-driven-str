import cv2
import numpy as np

from localize.probability_threshold_word_connectivity import (
    component_ids_intersecting_original_box,
    localize_native_components,
    nearest_axis_cells,
)
from pipeline.frozen_segmentation import backproject_binary_mask


def test_nearest_cell_mapping_reconstructs_frozen_backprojection():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[2:5, 7:11] = 1
    x_starts, x_ends = nearest_axis_cells(128, 333)
    y_starts, y_ends = nearest_axis_cells(128, 257)
    reconstructed = np.zeros((257, 333), dtype=np.uint8)
    for y, x in np.argwhere(mask):
        reconstructed[y_starts[y]:y_ends[y], x_starts[x]:x_ends[x]] = 1
    assert np.array_equal(reconstructed, backproject_binary_mask(mask, (257, 333)))


def test_native_components_use_mapped_area_padding_sort_and_top10():
    mask = np.zeros((128, 128), dtype=np.uint8)
    for index in range(12):
        size = 1 if index else 2
        mask[2 + index * 8:2 + index * 8 + size, 2:2 + size] = 1
    result = localize_native_components(
        mask, output_width=256, output_height=256,
        min_area=4, pad=6, connectivity=8, nmax=10,
    )
    assert len(result.raw_components) == 12
    assert len(result.candidates_before_cap) == 12
    assert len(result.retained) == 10
    assert result.retained[0].mapped_area == 16
    assert result.retained[0].bbox_original == (4, 4, 8, 8)
    assert result.retained[0].padded_bbox_original == (0, 0, 14, 14)
    assert [row.mapped_area for row in result.retained] == sorted(
        (row.mapped_area for row in result.retained), reverse=True
    )


def test_component_box_intersection_uses_foreground_not_component_bbox():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[5, 5] = 1
    mask[7, 7] = 1
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    cells = nearest_axis_cells(128, 256)
    assert component_ids_intersecting_original_box(
        labels, (10, 10, 12, 12), x_cells=cells, y_cells=cells
    ) == (1,)
    assert component_ids_intersecting_original_box(
        labels, (14, 14, 16, 16), x_cells=cells, y_cells=cells
    ) == (2,)
    assert component_ids_intersecting_original_box(
        labels, (12, 12, 14, 14), x_cells=cells, y_cells=cells
    ) == ()


def test_nearest_downsampling_drops_native_pixels_without_false_area():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[127, 127] = 1
    result = localize_native_components(
        mask, output_width=64, output_height=64,
        min_area=1, pad=6, connectivity=8, nmax=10,
    )
    assert len(result.raw_components) == 1
    assert result.raw_components[0].mapped_area == 0
    assert result.raw_components[0].bbox_original == (0, 0, 0, 0)
    assert result.retained == ()


def test_downsampled_bbox_ignores_extrema_that_do_not_survive_both_axes():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[1, 2] = 1  # y=1 is dropped by 128->64 nearest sampling
    mask[2, 2] = 1
    result = localize_native_components(
        mask, output_width=64, output_height=64,
        min_area=1, pad=0, connectivity=8, nmax=10,
    )
    assert len(result.retained) == 1
    assert result.retained[0].mapped_area == 1
    assert result.retained[0].bbox_original == (1, 1, 2, 2)
