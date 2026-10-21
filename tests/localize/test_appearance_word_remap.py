from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from localize.appearance_remap_config import load_appearance_config
from localize.appearance_word_remap import (
    AppearanceRemapParams, appearance_guided_local_remap, project_native_labels,
)
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_appearance_remap_train import (
    _contained, _write_exclusive, crop_geometry, read_native_mask, verify_train_contract,
)


CONFIG = Path("configs/approximate/localization_appearance_guided_local_remap_v1.yaml")


def settings(**overrides):
    config = load_appearance_config(CONFIG)
    return config.v1, replace(config.remap, **{"min_area_original": 0, **overrides})


def scene(boxes, colors=None):
    mask = np.zeros((128, 128), np.uint8)
    rgb = np.full((128, 128, 3), 255, np.uint8)
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        mask[y1:y2, x1:x2] = 1
        rgb[y1:y2, x1:x2] = colors[index] if colors is not None else (0, 0, 0)
    return mask, rgb


def run(mask, rgb, **overrides):
    v1, params = settings(**overrides)
    return appearance_guided_local_remap(mask, rgb, v1_params=v1, params=params)


def two_bases(colors=None):
    # Six-pixel internal gaps, eight-pixel gap between two v1 groups.
    return scene([(10, 20, 13, 40), (19, 20, 22, 40),
                  (30, 20, 33, 40), (39, 20, 42, 40)], colors)


def test_same_appearance_rescues_only_the_two_adjacent_bases():
    mask, rgb = two_bases()
    before = mask.copy(), rgb.copy()
    result = run(mask, rgb)
    assert len(result.base.retained) == 2
    assert len(result.retained) == 1
    block = result.retained[0]
    assert block.raw_component_ids == (1, 2, 3, 4)
    assert block.bbox == (10, 20, 42, 40)
    assert block.foreground_area == int(mask.sum()) == 240
    assert sum(row["accepted"] for row in result.merge_decisions) == 1
    assert np.array_equal(before[0], mask) and np.array_equal(before[1], rgb)
    assert json.loads(json.dumps(result.as_dict(), allow_nan=False))["table_5_accuracy"] is None


def test_rejects_different_foreground_colors_without_dropping_bases():
    mask, rgb = two_bases([(180, 0, 0)]*2 + [(0, 0, 180)]*2)
    result = run(mask, rgb)
    assert len(result.retained) == 2
    assert "color_incompatible" in result.pair_decisions[0]["reasons"]


def test_opposite_foreground_background_polarity_is_a_veto():
    mask, rgb = two_bases()
    rgb[:, 26:] = 0
    rgb[(mask > 0) & (np.indices(mask.shape)[1] >= 26)] = 255
    result = run(mask, rgb, max_color_delta_e76=150)
    assert len(result.retained) == 2
    assert "contrast_polarity_incompatible" in result.pair_decisions[0]["reasons"]


def test_thickness_incompatibility_blocks_new_merge():
    mask, rgb = scene([(10, 20, 12, 40), (20, 20, 28, 40)])
    result = run(mask, rgb)
    assert len(result.retained) == 2
    assert "stroke_incompatible" in result.pair_decisions[0]["reasons"]


def test_missing_contrast_is_fail_closed_for_merge_not_for_base():
    mask, rgb = two_bases()
    rgb[:] = 100
    result = run(mask, rgb)
    assert len(result.retained) == 2
    assert not any(row["eligible_for_new_merge"] for row in result.raw_descriptors)
    assert "insufficient_appearance_evidence" in result.pair_decisions[0]["reasons"]


def test_whole_group_color_check_prevents_transitive_color_drift():
    mask, rgb = scene([(10, 20, 13, 40), (21, 20, 24, 40), (32, 20, 35, 40)],
                      [(0, 0, 0), (40, 40, 40), (80, 80, 80)])
    result = run(mask, rgb)
    assert all(row["accepted"] for row in result.pair_decisions)
    triple = next(row for row in result.merge_decisions if len(row["base_group_ids"]) == 3)
    assert not triple["accepted"] and "color_incompatible" in triple["reasons"]
    assert len(result.retained) == 2


def test_new_groups_are_never_reused_to_chain_more_than_three_bases():
    mask, rgb = scene([(10+11*k, 20, 13+11*k, 40) for k in range(5)])
    result = run(mask, rgb)
    assert len(result.base.groups_before_cap) == 5
    assert len(result.retained) == 2
    assert max(len(group["base_group_ids"]) for group in result.groups_before_filter) == 3
    assert sorted(k for group in result.groups_before_filter for k in group["raw_component_ids"]) == [1,2,3,4,5]


def test_failed_immediate_neighbour_cannot_be_skipped():
    mask, rgb = scene([(10, 20, 13, 40), (21, 20, 24, 40), (32, 20, 35, 40)],
                      [(0, 0, 0), (180, 0, 0), (0, 0, 0)])
    result = run(mask, rgb, max_normalized_gap=2.0)
    assert len(result.retained) == 3
    pairs = {(row["left_base_group_id"], row["right_base_group_id"]) for row in result.pair_decisions}
    assert pairs == {(1, 2), (2, 3)}


def test_distinct_close_lines_are_not_joined():
    mask, rgb = scene([(10, 20, 13, 40), (21, 43, 24, 63)])
    result = run(mask, rgb)
    assert len(result.retained) == 2 and not result.pair_decisions


def test_singleton_punctuation_and_empty_mask():
    mask, rgb = scene([(12, 20, 13, 21)])
    result = run(mask, rgb)
    assert len(result.retained) == 1
    assert result.retained[0].foreground_area == 1
    assert not result.raw_descriptors[0]["eligible_for_new_merge"]
    mask[:] = 0
    empty = run(mask, rgb)
    assert not empty.retained and not empty.raw_descriptors


def test_holes_and_nested_islands_keep_true_foreground_and_separate_ids():
    mask, rgb = scene([(10, 10, 30, 30)])
    mask[11:29, 11:29] = 0
    mask[19:21, 19:21] = 1
    rgb[:] = 255
    rgb[mask > 0] = 0
    result = run(mask, rgb, enabled=False)
    assert len(result.raw_descriptors) == 2
    assert sorted(row["foreground_area_original"] for row in result.raw_descriptors) == [4, 76]
    assert sum(b.foreground_area for b in result.retained) == 80


def test_disabled_merging_is_exactly_the_matched_v1_control():
    mask, rgb = two_bases()
    for area in (0, 150):
        result = run(mask, rgb, enabled=False, min_area_original=area)
        assert result.groups_before_cap == result.control_groups_before_cap
        for actual, control in zip(result.retained, result.control_retained):
            assert actual.padded_bbox == control.padded_bbox
            assert actual.foreground_area == control.foreground_area
            assert np.array_equal(actual.crop, control.crop)
        if area == 0:
            assert [b.raw_component_ids for b in result.retained] == [b.raw_component_ids for b in result.base.retained]
        else:
            assert not result.retained and len(result.base.retained) == 2


def test_non_integer_projection_matches_frozen_nearest_and_uses_original_area():
    mask, _ = two_bases()
    projected = project_native_labels(mask, (257, 381))
    assert np.array_equal(projected, backproject_binary_mask(mask, (257, 381)))
    rgb = np.full((257, 381, 3), 255, np.uint8)
    rgb[projected > 0] = 0
    result = run(mask, rgb, enabled=False)
    assert sum(row["foreground_area_original"] for row in result.raw_descriptors) == int(projected.sum())
    assert all(row["bbox_original"][2] > row["bbox_original"][0] for row in result.raw_descriptors)
    # Geometry is evaluated after mapping, not gap128/height128.
    pair = result.pair_decisions[0]
    # Stretching width and height differently can also change the v1 partition.
    # Follow the recorded pair IDs, not an assumed native-frame partition.
    by_id = {group["group_id"]: group for group in result.base.groups_before_cap}
    a = by_id[pair["left_base_group_id"]]
    b = by_id[pair["right_base_group_id"]]
    aa, bb = a["bbox"], b["bbox"]
    expected = (bb[0]-aa[2])/((aa[3]-aa[1]+bb[3]-bb[1])/2)
    assert pair["normalized_gap_original"] == expected


def test_top10_border_clipping_order_and_determinism():
    mask, rgb = scene([(11*k, 0, 11*k+2, 3) for k in range(11)])
    first, second = run(mask, rgb), run(mask, rgb)
    assert len(first.retained) == 10
    assert first.retained[0].padded_bbox[:2] == (0, 0)
    assert first.rejected[-1]["reason"] == "top10"
    assert first.as_dict() == second.as_dict()


def test_short_source_image_allowed_when_component_identity_is_preserved():
    mask, _ = two_bases()
    projected = project_native_labels(mask, (102, 422))
    assert np.array_equal(projected, backproject_binary_mask(mask, (102, 422)))
    rgb = np.full((102, 422, 3), 255, np.uint8)
    rgb[projected > 0] = 0
    result = run(mask, rgb, enabled=False)
    assert sorted(row["native_component_id"] for row in result.raw_descriptors) == [1, 2, 3, 4]
    assert sum(row["foreground_area_original"] for row in result.raw_descriptors) == int(projected.sum())


@pytest.mark.parametrize("boxes", [
    [(10, 1, 13, 2)],  # The only foreground row disappears in nearest downsampling.
    [(10, 0, 13, 1), (10, 2, 13, 3)],  # A discarded background row merges two CCs.
])
def test_short_source_still_rejects_actual_component_loss_or_merge(boxes):
    mask, _ = scene(boxes)
    projected = project_native_labels(mask, (64, 128))
    rgb = np.full((64, 128, 3), 255, np.uint8)
    rgb[projected > 0] = 0
    with pytest.raises(RuntimeError, match="lost or merged"):
        run(mask, rgb)


@pytest.mark.parametrize("overrides", [
    {"max_color_delta_e76": float("nan")}, {"max_stroke_ratio": .5},
    {"min_vertical_overlap": 1.1}, {"min_area_original": -1},
    {"min_background_pixels": True}, {"enabled": 1},
])
def test_parameters_fail_closed(overrides):
    with pytest.raises(ValueError):
        settings(**overrides)


def test_binary_contract_rejects_soft_masks_and_foreground_255():
    for mask in (np.zeros((128,128),np.float32), np.full((128,128),255,np.uint8)):
        with pytest.raises(ValueError, match="binary"):
            run(mask, np.zeros((128,128,3),np.uint8))
    with pytest.raises(ValueError, match="positive"):
        project_native_labels(np.zeros((128,128),np.int32),(0,128))


def test_config_is_explicit_approximate_and_has_no_namespace_fallback(tmp_path):
    value = yaml.safe_load(CONFIG.read_text())
    path = tmp_path/"configs"/"strict"/"appearance.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="approximate"):
        load_appearance_config(path)
    path = tmp_path/"configs"/"approximate"/"appearance.yaml"
    path.parent.mkdir(parents=True)
    value["reproduction_mode"] = "strict"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="reproduction_mode"):
        load_appearance_config(path)


def contract_fixture():
    cfg = load_appearance_config(CONFIG)
    ids = [str(i) for i in range(100,329)]
    manifest = SimpleNamespace(split_role="train", split=cfg.resolved["qualification"]["split"],
                               samples=[SimpleNamespace(sample_id=k) for k in ids])
    checkpoint = {"sha256": cfg.resolved["qualification"]["checkpoint_sha256"]}
    cache = {"split":manifest.split, "sample_count":229, "test_data_loaded":False,
             "checkpoint":checkpoint, "samples":[{"sample_id":k} for k in ids],
             "inference":{"native_shape":[128,128], "threshold":.5, "activation":"sigmoid",
                          "comparison":"greater_than_or_equal", "backprojection_applied":False}}
    calibration = {"train_only_guard":{"test_data_loaded":False, "split_role":"train", "sample_count":229},
                   "scientific_boundary":{"selection_used_ocr":False, "selection_used_test":False},
                   "frozen_segmentation":{"checkpoint":checkpoint},
                   "calibration":{"selected":{"parameters":asdict(cfg.v1)}},
                   "train_source":{"samples":[{"sample_id":k} for k in ids]}}
    return cache, manifest, calibration, cfg


def test_train_guard_rejects_test_contamination_and_duplicate_ids():
    cache, manifest, calibration, cfg = contract_fixture()
    assert len(verify_train_contract(cache,manifest,calibration,cfg)[0]) == 229
    manifest.split_role = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        verify_train_contract(cache,manifest,calibration,cfg)
    manifest.split_role = "train"
    calibration["scientific_boundary"]["selection_used_test"] = True
    with pytest.raises(ValueError, match="contaminated"):
        verify_train_contract(cache,manifest,calibration,cfg)
    calibration["scientific_boundary"]["selection_used_test"] = False
    cache["samples"][0] = cache["samples"][1]
    with pytest.raises(ValueError, match="identities"):
        verify_train_contract(cache,manifest,calibration,cfg)


def test_cached_mask_file_and_decoded_hash_are_both_checked(tmp_path):
    native = np.zeros((128,128),np.uint8)
    native[3:7,4:9] = 1
    path = tmp_path/"sample.png"
    Image.fromarray(native*255).save(path)
    entry = {"sample_id":"100", "native_mask_path":"sample.png", "foreground_pixels":int(native.sum()),
             "native_mask_file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
             "native_mask_array_sha256":hashlib.sha256(native.tobytes()).hexdigest()}
    assert np.array_equal(read_native_mask(tmp_path,entry),native)
    entry["native_mask_array_sha256"] = "0"*64
    with pytest.raises(ValueError, match="decoded"):
        read_native_mask(tmp_path,entry)
    entry["native_mask_file_sha256"] = "0"*64
    with pytest.raises(ValueError, match="PNG SHA"):
        read_native_mask(tmp_path,entry)


def test_artifact_path_escape_and_report_overwrite_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        _contained(tmp_path,"../outside")
    path = tmp_path/"output.json"
    _write_exclusive(path,"original")
    with pytest.raises(FileExistsError):
        _write_exclusive(path,"replacement")
    assert path.read_text() == "original"


def test_crop_wordness_is_separate_from_per_word_best_iou():
    blocks = tuple(SimpleNamespace(rank=i+1, raw_component_ids=(i+1,), bbox=box,
                                   padded_bbox=box, foreground_area=20)
                   for i,box in enumerate([(0,0,10,10),(0,0,25,10),(50,50,60,60)]))
    crops, counts, words = crop_geometry(blocks,((0,0,10,10),(15,0,25,10)),.25)
    assert counts == {"WORD_GOOD":1,"WORD_PARTIAL":0,"MULTIWORD_MERGE":1,"UNMATCHED":1}
    assert crops[2]["best_gt_index"] is None
    assert words[0]["best_iou"] == 1
    assert set(inspect.signature(appearance_guided_local_remap).parameters) == {"native_mask","image_rgb","v1_params","params"}
