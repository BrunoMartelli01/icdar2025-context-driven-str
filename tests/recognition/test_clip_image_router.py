import copy
import math

import pytest

from recognition.clip_crop_admission import FEATURES
from recognition.clip_image_router import accepted, choose_threshold, image_scores, partition, summarize


def example(confidence):
    return {"raw_text":"word", "confidence":confidence,
            "native":{"confidence_mean":confidence,"confidence_min":confidence},
            "clip":dict.fromkeys((*FEATURES,"scene_ensemble"), confidence)}


def test_worst_word_controls_entire_image():
    reference = dict.fromkeys(("native_rank", *FEATURES, "scene_ensemble"), [.1,.5,.9])
    rows = [example(.99), example(.1), example(.99)]
    before = copy.deepcopy(rows)
    scores = image_scores(rows,reference)
    assert scores["native_balanced"] == .1
    assert scores["native_glyph"] == pytest.approx(1/6)
    assert rows == before
    assert not accepted(scores["native_balanced"],.5)


@pytest.mark.parametrize("rows",[[],[dict(raw_text="")],[dict(raw_text="   ")]])
def test_empty_output_forces_whole_image_fallback(rows):
    assert all(v is None for v in image_scores(rows,{}).values())


def test_split_is_deterministic_order_independent_and_disjoint():
    ids = [f"id:{i}" for i in range(125)]
    a, b = partition(ids,80,12),partition(ids[::-1],80,12)
    assert set(a["calibration"]) == set(b["calibration"])
    assert len(a["calibration"]) == 80 and len(a["validation"]) == 45
    assert not set(a["calibration"]) & set(a["validation"])
    assert set(a["calibration"]+a["validation"]) == set(ids)


@pytest.mark.parametrize("ids,count",[(["a","a"],1),(["a","b"],0),(["a","b"],2)])
def test_invalid_split_fails(ids,count):
    with pytest.raises(ValueError):partition(ids,count,12)


def test_threshold_boundary_is_inclusive_and_rejects_abstention():
    assert accepted(.5,.5)
    assert not accepted(.49,.5)
    assert not accepted(None,0.) and not accepted(1.,None)
    with pytest.raises(ValueError):accepted(math.nan,.5)


def population():
    return [dict(sample_id="good",tp=2,words=2,gt_words=3,outputs_all=3),
            dict(sample_id="bad",tp=3,words=4,gt_words=5,outputs_all=4),
            dict(sample_id="good2",tp=3,words=3,gt_words=3,outputs_all=3)]


def test_accuracy_on_detected_differs_from_tn0_with_missing_words():
    rows=population();scores={"good":{"f":.9},"bad":{"f":.3},"good2":{"f":.2}}
    result=summarize(rows,scores,"f",.8)
    assert result["accuracy_on_detected_words"] == 1.
    assert result["accuracy_tn0_including_fn"] == pytest.approx(2/3)
    assert result["tn"] == 0 and result["missing_words_in_accepted_images"] == 1
    assert result["already_correct_images_sent_to_deepsolo"] == 1
    assert result["already_correct_words_sent_to_deepsolo"] == 6
    assert result["accepted_outputs_including_ignored"] == 3
    assert result["fully_correct_and_complete_images"] == 0


def test_threshold_search_keeps_ties_whole_and_cannot_use_validation():
    rows=population();scores={"good":{"f":.9},"bad":{"f":.3},"good2":{"f":.2}}
    choice=choose_threshold(rows,scores,"f",.99)
    assert choice["threshold"] == .9
    assert choice["calibration"]["accepted_images"] == 1
    scores["bad"]["f"]=.9
    assert choose_threshold(rows,scores,"f",.99)["threshold"] is None
    choice=choose_threshold(rows[:1],scores,"f",.99)
    rows[1]["tp"]=0
    assert choose_threshold(rows[:1],scores,"f",.99) == choice


def test_abstain_all_is_undefined_accuracy_not_one_hundred_percent():
    rows=population();scores={r["sample_id"]:{"f":0.} for r in rows}
    result=choose_threshold(rows,scores,"f",.99)
    assert result["threshold"] is None
    assert result["calibration"]["accuracy_on_detected_words"] is None
    assert result["calibration"]["fallback_images"] == 3
