"""Barrier boundaries, fused kernels, deterministic BFS and original coordinates."""

import numpy as np
import pytest

from localize.adaspot_gap import gap_instances
from localize.word_multitask import parseq_word_proposals, word_instances
from scripts.run_adaspot_gap import DEFAULT_CONFIG, read_config


@pytest.fixture
def post():
    return read_config(DEFAULT_CONFIG, "approximate")[0]["postprocessing"]


def fused_maps():
    words = np.zeros((2, 12, 16), np.float32)
    words[:, 3:9, 2:12] = .9
    gap = np.zeros((12, 16), np.float32); gap[:, 6] = .9
    return words, gap, np.ones((12, 16), bool)


def test_barrier_splits_fused_kernel_and_blocks_growth(post):
    words, gap, canvas = fused_maps()
    baseline = gap_instances(words, gap, canvas, post, use_gap=False)[0]
    labels, audit, foreground = gap_instances(words, gap, canvas, post)
    assert baseline.max() == 1 and labels.max() == 2
    assert not labels[:, 6].any() and not foreground[:, 6].any()
    assert audit["gap_blocked_kernel_pixels"] == audit["gap_blocked_pixels"] == 6
    assert np.array_equal(labels, gap_instances(words, gap, canvas, post)[0])


def test_off_decoder_is_exact_previous_kernel_bfs(post):
    words, gap, canvas = fused_maps()
    actual = gap_instances(words, gap, canvas, post, use_gap=False)[0]
    assert np.array_equal(actual, word_instances(words, canvas, post)[0])


def test_gap_exact_threshold_is_retained_and_unseeded_is_explicit(post):
    words, gap, canvas = fused_maps()
    gap[:] = post["gap_threshold"]
    assert gap_instances(words, gap, canvas, post)[0].max() == 1
    words[1] = 0
    labels, audit, _ = gap_instances(words, gap, canvas, post)
    assert labels.max() == 1 and audit["unseeded_fallback_components"] == 1


def test_outside_canvas_and_missing_gap_are_not_silently_accepted(post):
    words, gap, canvas = fused_maps(); canvas[:, 2:4] = False
    assert not gap_instances(words, gap, canvas, post)[0][:, 2:4].any()
    with pytest.raises(ValueError, match="requires predicted"):
        gap_instances(words, None, canvas, post)
    gap[0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid paired"):
        gap_instances(words, gap, canvas, post)


def test_adjacent_ids_preserve_true_area_and_original_coordinate_mapping():
    labels = np.zeros((16, 16), np.int32)
    labels[4:8, 3:5] = 1; labels[4:8, 5:8] = 2
    rows = parseq_word_proposals(labels, (80, 40), padding=0, min_area=0, max_instances=None)
    assert len(rows) == 2 and {p["foreground_area"] for p in rows} == {8, 12}
    assert all(0 <= p["bbox"][0] < p["bbox"][2] <= 80 and
               0 <= p["bbox"][1] < p["bbox"][3] <= 40 for p in rows)
