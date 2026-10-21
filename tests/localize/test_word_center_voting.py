import numpy as np
import pytest

from localize.word_center_offsets import _splat, word_center_look_signal, split_text_with_center_offsets
from localize.word_multitask import parseq_word_proposals
from segmentation.word_center_offsets import make_word_center_offsets


@pytest.fixture
def voting():
    return {"text_threshold": .65, "smoothing_sigma": 1., "min_peak_density": 1., "min_center_distance": 3,
            "assignment_radius": 3.}


def scene():
    owners = np.zeros((32, 32), np.int32)
    owners[5:27, 2:16] = 1
    owners[5:27, 16:30] = 2
    offsets, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    return owners, np.where(owners, .9, 0).astype(np.float32), offsets, np.ones_like(owners, bool)


def test_vectors_vote_for_two_centers_in_one_fused_mask_and_preserve_shape(voting):
    owners, text, offsets, canvas = scene()
    before = text.copy()
    labels, signal = split_text_with_center_offsets(text, offsets, canvas, voting)
    assert signal["audit"]["base_components"] == 1 and signal["audit"]["center_markers"] == 2
    assert signal["audit"]["instances"] == 2
    assert np.array_equal(labels > 0, text >= voting["text_threshold"])
    assert np.array_equal(text, before)
    for word in (1, 2): assert len(np.unique(labels[owners == word])) == 1
    assert labels[10, 5] != labels[10, 25]
    repeated = split_text_with_center_offsets(text, offsets, canvas, voting)
    assert np.array_equal(labels, repeated[0]) and signal["centers"] == repeated[1]["centers"]
    boxes = parseq_word_proposals(labels, (64, 64), padding=0, min_area=0, max_instances=10)
    assert len(boxes) == 2 and boxes[0]["foreground_area"] == 22*14


def test_no_cue_and_all_rejected_votes_keep_classic_components(voting):
    _, text, offsets, canvas = scene()
    no_peaks = {**voting, "min_peak_density": 1e6}
    labels, signal = split_text_with_center_offsets(text, offsets, canvas, no_peaks)
    assert signal["audit"]["instances"] == 1 and signal["audit"]["unseeded_fallback_components"] == 1
    outside = np.ones_like(offsets)
    labels, signal = split_text_with_center_offsets(text, outside, canvas, voting)
    assert signal["audit"]["rejected_endpoint_votes"] == int((text > 0).sum())
    assert signal["audit"]["instances"] == 1 and not signal["vote_density"].any()
    assert np.array_equal(labels > 0, text > 0)


def test_background_cannot_vote_and_canvas_excludes_padding(voting):
    _, text, offsets, canvas = scene()
    offsets[:, text == 0] = .9
    signal = word_center_look_signal(text, offsets, canvas, voting)
    assert signal["audit"]["center_markers"] == 2
    canvas[:, :16] = False
    signal = word_center_look_signal(text, offsets, canvas, voting)
    assert signal["audit"]["center_markers"] == 1
    assert not signal["foreground"][:, :16].any() and not signal["vote_density"][:, :16].any()


def test_component_local_voting_does_not_seed_another_disconnected_region(voting):
    owners = np.zeros((32, 32), np.int32)
    owners[3:12, 2:12] = 1; owners[19:28, 20:30] = 2
    offsets, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    text = (owners > 0).astype(np.float32)
    labels, signal = split_text_with_center_offsets(text, offsets, np.ones_like(owners, bool), voting)
    assert signal["audit"]["base_components"] == 2 and signal["audit"]["instances"] == 2
    assert {row["base_component"] for row in signal["centers"]} == {1, 2}


def test_concave_word_marker_projects_into_shape(voting):
    owners = np.zeros((32, 32), np.int32)
    owners[3:27, 3:9] = owners[3:27, 23:29] = owners[23:29, 3:29] = 1
    offsets, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    text = (owners > 0).astype(np.float32)
    labels, signal = split_text_with_center_offsets(text, offsets, np.ones_like(owners, bool), voting)
    assert signal["audit"]["center_markers"] == 1
    x, y = signal["centers"][0]["vote_peak_xy_native"]
    mx, my = signal["centers"][0]["marker_xy_native"]
    assert owners[y, x] == 0 and owners[my, mx] == 1
    assert np.array_equal(labels > 0, owners > 0)


def test_subpixel_bilinear_vote_mass_and_empty_foreground(voting):
    density = _splat(np.array([3.25]), np.array([2.5]), np.array([.8]), (8, 10))
    assert density.sum() == pytest.approx(.8)
    assert density[2, 3] == pytest.approx(.3) and density[3, 4] == pytest.approx(.1)
    labels, signal = split_text_with_center_offsets(np.zeros((8, 10)), np.zeros((2, 8, 10)), np.ones((8, 10), bool), voting)
    assert not labels.any() and signal["audit"]["instances"] == 0


@pytest.mark.parametrize("problem", ["nan", "probability", "offset", "canvas", "shape", "threshold", "radius"])
def test_invalid_signal_fails_explicitly(problem, voting):
    _, text, offsets, canvas = scene()
    if problem == "nan": offsets[0, 0, 0] = np.nan
    if problem == "probability": text[0, 0] = 1.1
    if problem == "offset": offsets[0, 0, 0] = 1.1
    if problem == "canvas": canvas = canvas.astype(np.uint8)
    if problem == "shape": offsets = offsets[:1]
    if problem == "threshold": voting["text_threshold"] = 1
    if problem == "radius": voting["min_center_distance"] = 0
    with pytest.raises(ValueError): word_center_look_signal(text, offsets, canvas, voting)
