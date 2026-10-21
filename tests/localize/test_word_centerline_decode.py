"""GT-blind instance identity, disk competition, and oracle representation."""

import numpy as np
import pytest

from localize.word_centerline import METHODS, centerline_instances
from localize.word_multitask import word_instances
from segmentation.word_centerline import centerline_targets


POST = {"method": "watershed", "text_threshold": .65, "kernel_threshold": .5,
        "center_threshold": .5, "center_min_distance": 3, "watershed_weights": [.5, .25, .25],
        "line_threshold": .5, "radius_floor": .5, "max_radius": 128., "min_line_area": 0}


def maps():
    words = np.zeros((2, 20, 32), np.float32)
    words[0, 3:17, 2:30] = .9
    words[1, 7:12, 3:5] = words[1, 7:12, 26:28] = .9
    line = np.zeros((20, 32), np.float32)
    line[8, 5:11] = line[8, 21:27] = 1
    return {"words": words, "line": line, "radius": np.full_like(line, 3.)}, np.ones(line.shape, bool)


def test_baseline_is_exact_existing_decoder():
    evidence, canvas = maps()
    result = centerline_instances(evidence, canvas, POST, "baseline")
    expected = word_instances(evidence["words"], canvas, POST)
    assert np.array_equal(result[0], expected[0]) and result[1] == expected[1]


def test_line_markers_split_fused_foreground_and_preserve_unseeded_fallback():
    evidence, canvas = maps()
    labels, audit, foreground = centerline_instances(evidence, canvas, POST, "line_markers")
    assert audit["instances"] == 2 and labels[8, 6] != labels[8, 25]
    assert np.array_equal(labels > 0, evidence["words"][0] >= .65)
    evidence["words"][0, 0, 31] = 1
    labels, audit, _ = centerline_instances(evidence, canvas, POST, "line_markers")
    assert audit["unseeded_fallback_components"] == 1 and labels[0, 31] > 0
    evidence["line"][:] = 0
    labels, audit, _ = centerline_instances(evidence, canvas, POST, "line_markers")
    assert audit["seed_components"] == 0 and audit["unseeded_fallback_components"] == 2


def test_radius_recovers_beyond_frozen_mask_and_canvas_excludes_padding():
    evidence, canvas = maps()
    evidence["words"][:] = 0
    canvas[:, :3] = False
    labels, audit, foreground = centerline_instances(evidence, canvas, POST, "line_radius")
    assert foreground.any() and audit["instances"] == 2
    assert not labels[:, :3].any()
    assert labels[8, 5] != labels[8, 24]
    markers, _, _ = centerline_instances(evidence, canvas, POST, "line_markers")
    assert not markers.any()


def test_disk_competition_preserves_ids_and_row_major_ties_deterministically():
    evidence, canvas = maps()
    evidence["line"][:] = 0
    evidence["line"][10, 10] = evidence["line"][10, 20] = .5
    evidence["radius"][:] = 8
    labels, audit, _ = centerline_instances(evidence, canvas, POST, "line_radius")
    assert labels[10, 10] == labels[10, 15] == 1
    assert labels[10, 16] == labels[10, 20] == 2
    assert audit["competing_disk_pixels"] > 0
    assert np.array_equal(labels, centerline_instances(evidence, canvas, POST, "line_radius")[0])
    clipped, audit, _ = centerline_instances(evidence, canvas, {**POST, "max_radius": 4.}, "line_radius")
    assert audit["radius_clipped_pixels"] == 2
    assert not clipped[10, 15]


def test_perfect_targets_oracle_is_auditable_and_no_truth_needed_by_decoder():
    owners = np.zeros((32, 64), np.int32)
    owners[8:20, 4:28] = 1
    owners[7:19, 36:60] = 2
    target = centerline_targets(owners, np.ones_like(owners), {"ribbon_fraction": .2, "min_half_width": 0.})
    evidence = {"words": np.stack([owners > 0, owners > 0]).astype(np.float32),
                "line": target["line"], "radius": target["radius"]}
    labels, audit, foreground = centerline_instances(evidence, np.ones_like(owners, bool), POST, "line_radius")
    assert audit["instances"] == 2
    iou = np.count_nonzero(foreground & (owners > 0)) / np.count_nonzero(foreground | (owners > 0))
    assert iou > .65
    assert labels[13, 12] != labels[13, 48]


@pytest.mark.parametrize("method", METHODS)
def test_empty_predictions_and_invalid_evidence(method):
    evidence, canvas = maps()
    evidence["words"][:] = evidence["line"][:] = 0
    assert not centerline_instances(evidence, canvas, POST, method)[0].any()
    evidence["radius"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        centerline_instances(evidence, canvas, POST, method)


def test_filter_is_explicit_and_reported():
    evidence, canvas = maps()
    evidence["line"][:] = 0
    evidence["line"][3, 3] = 1
    labels, audit, _ = centerline_instances(evidence, canvas, {**POST, "min_line_area": 2}, "line_radius")
    assert not labels.any() and audit["filtered_line_components"] == 1
