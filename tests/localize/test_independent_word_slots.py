from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageOps
import pytest

from localize.independent_word_slots import (
    EmptyDetectorBoundaryError, assign_to_slots, boundary_bbox, detector_boundary_in_original,
)
from scripts.evaluate_totaltext_detector_slots import (
    ANALYSIS_CODE, CODE, _controls, _detector_reuse, _image_frame, _local_paths, _slots,
    _validate_detector_plan, _validate_detector_shard,
)
from scripts.crop_clip_runtime import sha


def prediction(cid, box, score=1.):
    return {"candidate_id": cid, "group_id": cid, "bbox": box,
            "score": score, "raw_text": cid, "word": cid}


def slot(index, box):
    return {"source_index": index, "bbox": box, "score": .9}


def test_local_paths_preserve_deepsolo_venv_interpreter_symlink(tmp_path: Path):
    base = tmp_path / "system-python"
    base.write_bytes(b"fixture")
    interpreter = tmp_path / "venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(base)
    profile = {key: str(tmp_path / key) for key in (
        "frozen_refinement_run", "test_manifest", "deepsolo_runtime_manifest",
        "deepsolo_source_root", "output_root")}
    profile["deepsolo_python"] = str(interpreter)

    paths = _local_paths(profile)

    assert paths["deepsolo_python"] == interpreter.absolute()
    assert paths["deepsolo_python"].resolve() == base.resolve()


def test_boundary_envelope_clamps_and_rejects_invalid_polygon():
    assert boundary_bbox([[-2, 1], [8.2, 1], [8.2, 9.1], [-2, 9.1]], (10, 10)) == [0, 1, 9, 10]
    with pytest.raises(ValueError):
        boundary_bbox([[0, 0], [1, 1]], (10, 10))
    with pytest.raises(ValueError):
        boundary_bbox([[0, 0], [1, 0], [2, 0]], (10, 10))


def test_multiword_crop_can_join_both_slots_but_is_emitted_once():
    slots = [slot(0, [0, 0, 10, 10]), slot(1, [10, 0, 20, 10])]
    candidates = [prediction("parent", [0, 0, 20, 10]),
                  prediction("right", [10, 0, 20, 10], .8)]
    selected, assignments, memberships = assign_to_slots(candidates, slots, min_iou=.4)
    assert [member["slot_source_index"] for member in memberships] == [0, 1]
    assert "parent" in [item["candidate_id"] for item in memberships[0]["eligible_candidates"]]
    assert "parent" in [item["candidate_id"] for item in memberships[1]["eligible_candidates"]]
    assert {item["candidate_id"] for item in selected} == {"parent", "right"}
    assert len({item["candidate_id"] for item in assignments}) == 2


def test_assignment_abstains_without_overlap_and_validates_duplicate_ids():
    selected, assignments, memberships = assign_to_slots(
        [prediction("a", [0, 0, 10, 10])], [slot(0, [20, 0, 30, 10])], min_iou=.1)
    assert selected == assignments == []
    assert memberships[0]["eligible_candidates"] == []
    with pytest.raises(ValueError):
        assign_to_slots([prediction("a", [0, 0, 10, 10])] * 2, [], min_iou=.1)
    with pytest.raises(ValueError):
        assign_to_slots([], [slot(0, [0, 0, 10, 10])] * 2, min_iou=.1)


def test_detector_slot_contract_and_predeclared_controls():
    config = {"schema_version": 1, "reproduction_mode": "approximate",
              "table_5_claim": False,
              "detector_candidate": "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE",
              "detector_device": "cuda:0", "source_selection": "guarded_consolidated",
              "evaluation_matching_iou": .5,
              "detector_inference_score_threshold": .05,
              "detector_score_controls": [.05, .4],
              "crop_slot_iou_controls": [.1, .5]}
    assert _controls(config) == [(.05, .1), (.05, .5), (.4, .1), (.4, .5)]
    response = {"success": True, "detections": [{"source_index": 3, "score": .8,
                "geometry": {"boundary_points": [[0, 0], [10, 0], [10, 5], [0, 5]]}}]}
    assert _slots(response, (20, 10)) == [{"source_index": 3, "score": .8,
                                          "bbox": [0, 0, 10, 5]}]
    with pytest.raises(ValueError):
        _slots({"success": True, "detections": response["detections"] * 2}, (20, 10))
    with pytest.raises(ValueError):
        _controls({**config, "detector_score_controls": [.01]})


@pytest.mark.parametrize("orientation", range(9))
def test_inverse_detector_orientation_matches_every_pixel_of_pil_transform(orientation):
    # Unique pixel values let us check the full inverse against an independent decoder.
    original = np.arange(15, dtype=np.uint8).reshape(3, 5)
    image = Image.fromarray(original)
    image.getexif()[274] = orientation
    decoded = np.asarray(ImageOps.exif_transpose(image))
    ys, xs = np.indices(decoded.shape)
    points = np.column_stack((xs.ravel(), ys.ravel()))
    restored = np.asarray(detector_boundary_in_original(points, (5, 3), orientation))
    expected_y, expected_x = np.unravel_index(decoded.ravel(), original.shape)
    np.testing.assert_array_equal(restored, np.column_stack((expected_x, expected_y)))


def test_exif_six_maps_out_of_bounds_detector_polygon_to_original_without_loss(tmp_path):
    path = tmp_path / "orientation-six.jpg"
    image = Image.new("RGB", (392, 294))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    frame = _image_frame(path)
    assert frame["original_wh"] == [392, 294]
    assert frame["detector_decoded_wh"] == [294, 392]
    detection = {"source_index": 7, "score": .25,
                 "geometry": {"boundary_points": [[7, 324], [23, 324], [23, 348], [7, 348]]}}
    response = {"success": True, "detections": [detection]}
    with pytest.raises(EmptyDetectorBoundaryError):
        _slots(response, (392, 294))
    rejected = []
    assert _slots(response, (392, 294), exif_orientation=frame["exif_orientation"],
                  rejected=rejected) == [{"source_index": 7, "score": .25,
                                          "bbox": [324, 270, 348, 286]}]
    assert rejected == []
    assert detection["geometry"]["boundary_points"][0] == [7, 324]


def test_empty_detector_envelope_requires_auditable_rejection_and_keeps_other_slots():
    outside = {"source_index": 1, "score": .8,
               "geometry": {"boundary_points": [[12, 2], [15, 2], [15, 6], [12, 6]]}}
    valid = {"source_index": 2, "score": .9,
             "geometry": {"boundary_points": [[0, 0], [5, 0], [5, 5], [0, 5]]}}
    response = {"success": True, "detections": [outside, valid]}
    with pytest.raises(EmptyDetectorBoundaryError):
        _slots(response, (10, 10))
    rejected = []
    assert _slots(response, (10, 10), rejected=rejected) == [slot(2, [0, 0, 5, 5])]
    assert rejected == [{"source_index": 1, "score": .8,
                         "reason": "no_positive_in_image_bbox",
                         "unclipped_bbox": [12., 2., 15., 6.], "clipped_bbox": [10, 2, 10, 6]}]
    with pytest.raises(ValueError, match="duplicate"):
        _slots({"success": True, "detections": [outside, outside]}, (10, 10), rejected=[])


@pytest.mark.parametrize("points", [[], [[0, 0], [1, 1]],
                                   [[0, 0], [1, float("nan")], [1, 1]],
                                   [[0, 0], [1, float("inf")], [1, 1]]])
def test_malformed_detector_geometry_is_fatal_even_with_rejection_logging(points):
    response = {"success": True, "detections": [
        {"source_index": 0, "score": .8, "geometry": {"boundary_points": points}}]}
    rejected = []
    with pytest.raises(ValueError, match="invalid detector boundary"):
        _slots(response, (10, 10), rejected=rejected)
    assert rejected == []


def test_unsupported_exif_orientation_is_fatal(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        detector_boundary_in_original([[0, 0], [1, 0], [1, 1]], (10, 10), 9)
    path = tmp_path / "bad-exif.jpg"
    image = Image.new("RGB", (10, 10))
    exif = image.getexif()
    exif[274] = 9
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match="unsupported"):
        _image_frame(path)


def detector_plan_fixture():
    return {"schema_version": 1, "reproduction_mode": "approximate", "table_5_claim": False,
            "config": {"detector_candidate": "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"},
            "config_sha256": "a" * 64, "frozen_run": "/frozen", "frozen_run_plan_sha256": "b" * 64,
            "source": "/source", "source_plan_sha256": "c" * 64, "manifest": "/manifest",
            "manifest_sha256": "d" * 64, "runtime_manifest": "/runtime",
            "runtime_manifest_sha256": "e" * 64, "deepsolo_python_entrypoint": "/venv/bin/python",
            "checkpoint_sha256": "f" * 64, "device": "cuda:0", "detector_inference_threshold": .05,
            "execution": {"batch_size": 1}, "controls": [{"detector_min_score": .05}],
            "ground_truth_use": "after all detector and assignment shards are frozen",
            "sample_ids": ["sample"], "code_sha256": {name: "0" * 64 for name in CODE}}


def test_frozen_detector_reuse_allows_analysis_changes_and_blocks_worker_changes():
    previous = detector_plan_fixture()
    current = {**previous, "code_sha256": {**previous["code_sha256"],
                                           **{name: "1" * 64 for name in ANALYSIS_CODE}}}
    _validate_detector_plan(previous, current)
    current["code_sha256"]["deepsolo/worker.py"] = "2" * 64
    with pytest.raises(RuntimeError, match="dependency code differs"):
        _validate_detector_plan(previous, current)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "manifest_sha256", "device",
                                   "detector_inference_threshold", "execution", "controls",
                                   "config", "source_plan_sha256"])
def test_frozen_detector_reuse_blocks_changed_scientific_inputs(field):
    previous = detector_plan_fixture()
    with pytest.raises(RuntimeError, match=field):
        _validate_detector_plan(previous, {**previous, field: "changed"})


def test_frozen_detector_reuse_rejects_missing_samples_and_duplicate_ids():
    previous = detector_plan_fixture()
    with pytest.raises(RuntimeError, match="sample IDs"):
        _validate_detector_plan(previous, {**previous, "sample_ids": ["missing"]})
    with pytest.raises(RuntimeError, match="sample IDs"):
        _validate_detector_plan({**previous, "sample_ids": ["sample", "sample"]}, previous)


def detector_shard_fixture(image_path, plan):
    return {"sample_id": "sample", "plan_sha256": "a" * 64,
            "image_sha256": sha(image_path), "image_wh": [10, 10], "ground_truth_use": "none",
            "response": {"schema_version": 1, "request_id": "fixture", "success": True,
                         "detections": [], "recognized_strings": [], "selected_output": "",
                         "reduction_policy": "highest_score", "error": None,
                         "latency": {}, "memory": {}, "placement": {"device": "cuda:0"},
                         "artifact": {"checkpoint_sha256": plan["checkpoint_sha256"],
                                      "candidate": {"artifact_id": plan["config"]["detector_candidate"]},
                                      "inference_options": {"device": "cuda:0",
                                                            "score_threshold": .05,
                                                            "lexicon_type": None}}}}


@pytest.mark.parametrize("field,value", [("plan_sha256", "b" * 64),
                                        ("image_sha256", "b" * 64), ("image_wh", [10, 11]),
                                        ("sample_id", "other"), ("ground_truth_use", "labels")])
def test_frozen_detector_shard_rejects_corrupt_provenance(tmp_path, field, value):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10)).save(image)
    plan = detector_plan_fixture()
    shard = detector_shard_fixture(image, plan)
    _validate_detector_shard(shard, "sample", "a" * 64, image, (10, 10), plan)
    with pytest.raises(RuntimeError, match="cached detector shard differs"):
        _validate_detector_shard({**shard, field: value}, "sample", "a" * 64, image, (10, 10), plan)


def test_detector_reuse_records_hashes_without_mutating_frozen_inputs(tmp_path):
    image = tmp_path / "image.jpg"
    Image.new("RGB", (10, 10)).save(image)
    source = tmp_path / "source"
    (source / "detector").mkdir(parents=True)
    previous = detector_plan_fixture()
    plan_path = source / "resolved_configuration.json"
    plan_path.write_text(json.dumps(previous))
    shard = detector_shard_fixture(image, previous)
    shard["plan_sha256"] = sha(plan_path)
    shard_path = source / "detector/sample.json"
    shard_path.write_text(json.dumps(shard))
    before = {path: path.read_bytes() for path in (plan_path, shard_path)}
    current = {**previous, "code_sha256": {**previous["code_sha256"],
                                           **{name: "1" * 64 for name in ANALYSIS_CODE}}}
    reuse = _detector_reuse(source, current, {"sample": (image, (10, 10))})
    assert reuse["source_plan_sha256"] == sha(plan_path)
    assert reuse["shard_sha256"] == {"sample": sha(shard_path)}
    assert reuse["source_code_sha256"] == previous["code_sha256"]
    assert {path: path.read_bytes() for path in before} == before
    image.write_bytes(b"changed image bytes")
    with pytest.raises(RuntimeError, match="cached detector shard differs"):
        _detector_reuse(source, current, {"sample": (image, (10, 10))})
