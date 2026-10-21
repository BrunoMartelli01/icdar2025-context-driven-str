"""Meaningful invariants for soft link growth and GT-free selective replacement."""

from copy import deepcopy
import json

import numpy as np
import pytest

from localize.unet_crop_links import patches_from_labels
from localize.unet_seeded_selective import (
    refine_from_evidence, replacement_guard, seeded_link_instances, seed_problem,
)
from localize.unet_word_research_v2 import NEIGHBORS
from scripts.run_unet_seeded_selective import DEFAULT_CONFIG, read_config


@pytest.fixture
def cfg():
    return deepcopy(read_config(DEFAULT_CONFIG, "approximate"))


def maps():
    p = np.zeros((2, 12, 12), np.float32)
    p[0, 2:10, 1:11] = .95
    p[1, 4:8, 2:4] = .95; p[1, 4:8, 8:10] = .95
    return p, np.ones((12, 12), bool)


@pytest.mark.parametrize("link_value", [0., .01, .5, 1.])
def test_even_completely_broken_links_keep_nuclei_and_do_not_create_fragments(cfg, link_value):
    c, src = cfg; p, canvas = maps()
    links = np.full((8, 12, 12), link_value)
    labels, audit = seeded_link_instances(p, links, canvas, src["postprocessing"], c["seeded_links"])
    assert np.array_equal(labels > 0, p[0] >= src["postprocessing"]["text_threshold"])
    assert len(np.unique(labels[labels > 0])) == 2
    assert np.all(labels[4:8, 2:4] == 1) and np.all(labels[4:8, 8:10] == 2)
    assert audit["unseeded_fallback_components"] == 0
    assert np.array_equal(labels, seeded_link_instances(p, links, canvas, src["postprocessing"], c["seeded_links"])[0])


def test_soft_reciprocal_links_actually_move_the_boundary(cfg):
    c, src = cfg
    p = np.zeros((2, 3, 9)); p[0] = .95; p[1, :, [0, 8]] = 1
    links = np.ones((8, 3, 9))
    canvas = np.ones((3, 9), bool)
    opts = {"link_weight": 1., "landscape_weight": 0.}
    original, _ = seeded_link_instances(p, links, canvas, src["postprocessing"], opts)
    # A single weak direction is sufficient to penalize a reciprocal edge.
    for d, (dy, dx) in enumerate(NEIGHBORS):
        if dx == 1: links[d, :, 2] = .01
    moved, _ = seeded_link_instances(p, links, canvas, src["postprocessing"], opts)
    assert not np.array_equal(moved, original)
    assert np.all(moved[:, :3] == 1) and np.all(moved[:, 3:] == 2)


def test_unseeded_disconnected_text_survives_once_and_canvas_is_respected(cfg):
    c, src = cfg; p, canvas = maps()
    p[0, 0, :2] = .95; canvas[0, 1] = False
    labels, audit = seeded_link_instances(p, np.zeros((8, 12, 12)), canvas, src["postprocessing"], c["seeded_links"])
    assert labels[0, 0] == 3 and labels[0, 1] == 0
    assert audit["instances"] == 3 and audit["unseeded_fallback_components"] == 1


@pytest.mark.parametrize("bad", ["nan", "range", "shape", "canvas", "weights", "threshold"])
def test_invalid_evidence_is_an_explicit_failure(cfg, bad):
    c, src = cfg; p, canvas = maps(); links = np.ones((8, 12, 12))
    if bad == "nan": links[0, 0, 0] = np.nan
    elif bad == "range": links[0, 0, 0] = 1.1
    elif bad == "shape": links = links[:7]
    elif bad == "canvas": canvas = canvas.astype(np.uint8)
    elif bad == "weights": c["seeded_links"] = {"link_weight": 0., "landscape_weight": 0.}
    elif bad == "threshold": src["postprocessing"]["kernel_threshold"] = 0.
    with pytest.raises(ValueError): seeded_link_instances(p, links, canvas, src["postprocessing"], c["seeded_links"])


def seed_fixture():
    labels = np.zeros((12, 12), np.int32); labels[3:9, 2:8] = 1
    p = np.zeros((2, 12, 12), np.float32); p[0, labels == 1] = .95; p[1, 4:8, 3:7] = .95
    record = {"crops": [{"seed_id": 1, "seed_bbox": [2, 3, 8, 9], "context_bbox": [0, 0, 12, 12], "whole_image_fallback": False}]}
    return {"probabilities": p, "labels": labels}, record


def test_gate_ignores_neighbor_and_background_confidence(cfg):
    c, src = cfg; first, _ = seed_fixture()
    before = seed_problem(first["probabilities"], first["labels"], 1, src["postprocessing"], c["gate"])
    first["probabilities"][0, first["labels"] == 0] = 1.
    assert seed_problem(first["probabilities"], first["labels"], 1, src["postprocessing"], c["gate"]) == before
    assert not before[0]
    first["probabilities"][0, first["labels"] == 1] = .8
    flagged, audit = seed_problem(first["probabilities"], first["labels"], 1, src["postprocessing"], c["gate"])
    assert flagged and "low_word_confidence" in audit["reasons"]


def test_confident_seed_is_exactly_preserved_without_loading_second_pass(cfg):
    c, src = cfg; first, record = seed_fixture()
    def forbidden(*_): raise AssertionError("Unselected crop should not load second-pass evidence")
    patches, audit = refine_from_evidence(first, record, (12, 12), forbidden, src, c, decoder="seeded", selective=True)
    original = patches_from_labels(first["labels"], (12, 12))[0][0]
    assert patches[0]["bbox"] == original["bbox"] and np.array_equal(patches[0]["mask"], original["mask"])
    assert audit["evaluated_crops"] == 0 and audit["retained_seeds"] == 1


def test_failed_refinement_reverts_to_exact_original_seed(cfg):
    c, src = cfg; first, record = seed_fixture(); first["probabilities"][0, first["labels"] == 1] = .8
    child = np.zeros((12, 12), np.int32); child[3:5, 2:4] = 1
    patches, audit = refine_from_evidence(first, record, (12, 12), lambda *_: {"decoded_labels": child},
                                          src, c, decoder="word", selective=True)
    original = patches_from_labels(first["labels"], (12, 12))[0][0]
    assert patches[0]["bbox"] == original["bbox"] and np.array_equal(patches[0]["mask"], original["mask"])
    assert audit["rejected_seeds"] == 1 and audit["decisions"][0]["guard"]["reason"] == "anchor_loss"


def test_valid_split_is_accepted_and_preserves_total_anchor(cfg):
    c, src = cfg; first, record = seed_fixture(); first["probabilities"][0, first["labels"] == 1] = .8
    child = first["labels"].copy(); child[3:9, 5:8] = 2
    patches, audit = refine_from_evidence(first, record, (12, 12), lambda *_: {"decoded_labels": child},
                                          src, c, decoder="seeded", selective=True)
    assert len(patches) == 2 and sum(p["mask"].sum() for p in patches) == 36 and audit["accepted_seeds"] == 1


def test_guard_detects_neighbor_inside_rgb_box_even_without_mask_overlap(cfg):
    c, _ = cfg
    original = {"bbox": [0, 0, 2, 2], "mask": np.ones((2, 2), bool)}
    other = {"bbox": [4, 0, 6, 2], "mask": np.ones((2, 2), bool)}
    mask = np.zeros((2, 6), bool); mask[:, :2] = True; mask[1, 5] = True
    accepted, audit = replacement_guard([{"bbox": [0, 0, 6, 2], "mask": mask}], original, [other], c["guard"])
    assert not accepted and audit["reason"] == "neighbor_in_crop"


def test_guard_thresholds_are_inclusive_and_fragment_count_is_bounded(cfg):
    c, _ = cfg; opts = c["guard"]
    original = {"bbox": [0, 0, 20, 1], "mask": np.ones((1, 20), bool)}
    child = {"bbox": [0, 0, 17, 1], "mask": np.ones((1, 17), bool)}
    assert replacement_guard([child], original, [], opts)[0]
    child["mask"][0, -1] = False
    assert replacement_guard([child], original, [], opts)[1]["reason"] == "anchor_loss"
    assert replacement_guard([original] * 4, original, [], opts)[1]["reason"] == "too_many_children"


def test_approximate_namespace_and_mode_are_required(tmp_path):
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(DEFAULT_CONFIG, "strict")
    copied = tmp_path / "config.json"; copied.write_text(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(copied, "approximate")
