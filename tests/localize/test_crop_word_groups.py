import json
from pathlib import Path

import pytest

from localize.crop_word_groups import group_crops
from recognition.crop_word_group_metric import grouped_oracle
from scripts.analyze_totaltext_crop_word_groups import validate_config


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/totaltext_crop_word_groups_train_v1.json"


def config():
    return json.loads(CONFIG.read_text())


def crop(cid, box, tile=None):
    origin = {"method": "threshold_tree", "threshold": .65}
    if tile:
        origin = {"method": "tiled_segmentation", "threshold": .65, "tile_id": tile}
    return {"candidate_id": cid, "bbox": box, "methods": [origin["method"]],
            "origins": [origin], "quad": None, "neutral_fraction": 0}


def group(rows, strategy="complete_link", cutoff=.55):
    c = config()
    return group_crops(rows, strategy=strategy, iou_min=cutoff,
                       center_limit=c["center_limit"], ambiguity=c["ambiguity"])


def test_threshold_and_window_same_word_group_without_text():
    rows = [crop("whole", [10, 20, 50, 40]), crop("tile", [11, 19, 51, 41], "g2-1")]
    result = group(rows)
    assert len(result["groups"]) == 1
    assert set(result["groups"][0]["candidate_ids"]) == {"whole", "tile"}
    assert result["uses_ocr"] is False and result["uses_ground_truth"] is False


def test_transitive_chain_does_not_fuse_complete_link_groups():
    rows = [crop("a", [0, 0, 10, 10]), crop("b", [3, 0, 13, 10]), crop("c", [6, 0, 16, 10])]
    assert len(group(rows, "connected", .5)["groups"]) == 1
    complete = group(rows, cutoff=.5)
    assert len(complete["groups"]) == 2
    assert all(g["minimum_member_iou"] >= .5 for g in complete["groups"])


def test_parent_crop_is_shared_between_two_words_and_preserves_id():
    rows = [crop("left", [0, 0, 10, 10]), crop("right", [10, 0, 20, 10]),
            crop("parent", [0, 0, 20, 10])]
    result = group(rows)
    child_groups = [g for g in result["groups"] if "left" in g["core_candidate_ids"]
                    or "right" in g["core_candidate_ids"]]
    assert len(child_groups) == 2
    assert all("parent" in g["candidate_ids"] for g in child_groups)
    assert len(result["ambiguous_parents"]) == 1
    assert set(result["ambiguous_parents"][0]["competing_group_ids"]) == {g["group_id"] for g in child_groups}
    assert {cid for g in result["groups"] for cid in g["candidate_ids"]} == {"left", "right", "parent"}


def test_parent_is_shared_across_three_words():
    rows = [crop(str(i), [10 * i, 0, 10 * (i + 1), 10]) for i in range(3)]
    result = group(rows + [crop("parent", [0, 0, 30, 10])])
    child_groups = [g for g in result["groups"] if "parent" not in g["core_candidate_ids"]]
    assert len(child_groups) == 3
    assert all("parent" in g["candidate_ids"] for g in child_groups)


def test_grouping_is_input_order_invariant_and_ignores_extra_ocr_fields():
    rows = [crop("a", [0, 0, 10, 10]), crop("b", [1, 0, 11, 10]), crop("c", [30, 0, 40, 10])]
    first = group(rows)
    for row in rows:
        row["raw_text"] = "unrelated secret GT-like string"
        row["confidence"] = 999
    assert first == group(list(reversed(rows)))


def test_shared_crop_cannot_count_twice_for_repeated_spelling():
    groups = [{"candidate_ids": ["same"]}, {"candidate_ids": ["same"]}]
    assert len(grouped_oracle(groups, {0: ["same"], 1: ["same"]}, 2)) == 1


def test_distinct_crops_can_recognize_repeated_spelling_and_reassign_paths():
    groups = [{"candidate_ids": ["a", "b"]}, {"candidate_ids": ["a"]}]
    assert grouped_oracle(groups, {0: ["a"], 1: ["b"]}, 2) == {0, 1}
    assert len(grouped_oracle([{"candidate_ids": ["a", "b"]}], {0: ["a"], 1: ["b"]}, 2)) == 1


@pytest.mark.parametrize("mutation", ["strict", "variants", "capacity"])
def test_config_rejects_scientific_protocol_changes(mutation):
    c = config()
    if mutation == "strict":
        c["reproduction_mode"] = "strict"
    elif mutation == "variants":
        c["final_variations"] = True
    else:
        c["evaluation"]["crop_capacity"] = 2
    with pytest.raises(ValueError):
        validate_config(c)


def test_invalid_crops_fail_and_empty_population_is_supported():
    assert group([])["groups"] == []
    assert grouped_oracle([], {}, 1) == set()
    with pytest.raises(ValueError, match="duplicate"):
        group([crop("a", [0, 0, 1, 1]), crop("a", [0, 0, 1, 1])])
    with pytest.raises(ValueError, match="nonempty"):
        group([crop("a", [1, 0, 1, 2])])
    row = crop("a", [0, 0, 10, 10])
    row["neutral_fraction"] = .1
    with pytest.raises(ValueError, match="variations"):
        group([row])
