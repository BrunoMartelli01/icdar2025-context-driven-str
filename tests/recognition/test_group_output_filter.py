import json
from pathlib import Path

import numpy as np
import pytest

from recognition.group_output_filter import abstain_uncorroborated, suppress_duplicate_regions
from recognition.group_word_selection import ProbabilityMass
from scripts.run_crop_group_postfilters import load_config, validate_config


def row(cid, text, box, *, rgb=None, baseline=False):
    return {"candidate_id": cid, "bbox": box, "crop_rgb_sha256": rgb or cid,
            "methods": ["baseline" if baseline else "threshold_tree"],
            "ocr": {"raw_text": text}}


def group(gid, core, shared=()):
    return {"group_id": gid, "candidate_ids": list(core) + list(shared),
            "core_candidate_ids": list(core), "shared_candidate_ids": list(shared)}


def selected(gid, cid, text, box, score):
    return {"group_id": gid, "candidate_id": cid, "raw_text": text,
            "word": text.casefold(), "bbox": box, "score": score}


def test_duplicate_suppression_removes_same_region_even_if_ocr_readings_differ():
    a, b = [0, 0, 10, 10], [1, 0, 11, 10]
    rows = [row("a", "word", a), row("b", "w0rd", b)]
    groups = [group("a", ["a"]), group("b", ["b"])]
    predictions = [selected("a", "a", "word", a, .9),
                   selected("b", "b", "w0rd", b, .8)]
    mass = ProbabilityMass(np.ones((16, 16), np.float32), (16, 16))
    kept, removed = suppress_duplicate_regions(predictions, groups, rows, mass,
                                                box_iou_min=.5, soft_iou_min=.5)
    assert [item["group_id"] for item in kept] == ["a"]
    assert removed == [{"group_id": "b", "candidate_id": "b",
                        "reason": "duplicate_mask_region", "winner_group_id": "a"}]
    assert suppress_duplicate_regions(list(reversed(predictions)), groups, rows, mass,
                                      box_iou_min=.5, soft_iou_min=.5)[0][0]["group_id"] == "a"


def test_duplicate_suppression_preserves_distinct_mask_islands_and_parent_child_readings():
    a, b = [0, 0, 10, 10], [2, 0, 12, 10]
    rows = [row("a", "east", a), row("b", "west", b), row("parent", "east west", [0, 0, 12, 10])]
    groups = [group("a", ["a"], ["parent"]), group("b", ["b"], ["parent"])]
    predictions = [selected("a", "a", "east", a, .9), selected("b", "b", "west", b, .8)]
    mass = ProbabilityMass(np.ones((16, 16), np.float32), (16, 16))
    kept, removed = suppress_duplicate_regions(predictions, groups, rows, mass,
                                                box_iou_min=.5, soft_iou_min=.5)
    assert kept == predictions and removed == []  # Shared multiword parent.
    groups = [group("a", ["a"]), group("b", ["b"])]
    probability = np.zeros((16, 16), np.float32)
    probability[:10, :2] = 1
    probability[:10, 10:12] = 1
    mass = ProbabilityMass(probability, (16, 16))
    kept, removed = suppress_duplicate_regions(predictions, groups, rows, mass,
                                                box_iou_min=.5, soft_iou_min=.5)
    assert kept == predictions and removed == []  # Box overlap alone is insufficient.


def test_abstention_uses_baseline_or_distinct_core_crops_not_shared_parent():
    box = [0, 0, 10, 10]
    rows = [row("base", "A", box, baseline=True),
            row("repeat1", "Word", box, rgb="one"), row("repeat2", "word!", box, rgb="two"),
            row("same1", "noise", box, rgb="same"), row("same2", "noise", box, rgb="same"),
            row("solo", "other", box), row("parent", "other", box)]
    groups = [group("base", ["base"]), group("repeat", ["repeat1", "repeat2"]),
              group("same", ["same1", "same2"]), group("solo", ["solo"], ["parent"])]
    predictions = [selected("base", "base", "A", box, .3),
                   selected("repeat", "repeat1", "Word", box, .8),
                   selected("same", "same1", "noise", box, .8),
                   selected("solo", "solo", "other", box, .8)]
    # Normalization in this experiment removes punctuation, as PARSeq evaluation does.
    predictions[1]["word"] = "word"
    kept, rejected = abstain_uncorroborated(predictions, groups, rows)
    assert [item["group_id"] for item in kept] == ["base", "repeat"]
    assert [item["group_id"] for item in rejected] == ["same", "solo"]
    assert rejected[-1]["corroborating_crop_count"] == 1


def test_postfilters_ignore_annotations_and_reject_malformed_identity():
    box = [0, 0, 10, 10]
    rows = [row("a", "word", box, baseline=True)]
    groups = [group("g", ["a"])]
    predictions = [selected("g", "a", "word", box, .5)]
    original = abstain_uncorroborated(predictions, groups, rows)
    rows[0]["ground_truth"] = "incorrect"; groups[0]["gt_match"] = False
    assert abstain_uncorroborated(predictions, groups, rows) == original
    with pytest.raises(ValueError, match="invalid selected prediction identity"):
        abstain_uncorroborated([selected("g", "missing", "word", box, .5)], groups, rows)


def test_postfilter_config_is_isolated_and_rejects_dataset_tuning():
    path = Path(__file__).resolve().parents[2] / "configs/approximate/crop_group_postfilters_v1.json"
    config = load_config(path)
    assert config["reproduction_mode"] == "approximate" and "dataset" not in config
    assert config["evidence_abstention"]["confidence_cutoff"] is None
    changed = json.loads(json.dumps(config)); changed["reproduction_mode"] = "strict"
    with pytest.raises(ValueError):
        validate_config(changed)
    changed = json.loads(json.dumps(config)); changed["dataset"] = "total_text"
    with pytest.raises(ValueError):
        validate_config(changed)
