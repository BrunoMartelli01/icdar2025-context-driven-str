"""Boundary tests for transfer routing; no GPU, downloads or new raster files."""
from functools import partial
import hashlib
import json

import pytest

from scripts.evaluate_totaltext_highres_routing import checked, crop_rows, summarize_methods
from scripts.run_clip_router_e2e import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256

summarize_methods = partial(summarize_methods, iou_threshold=.5, paper_threshold=.8)


def ocr(text="WORD", digest="crop"):
    return {"crop_rgb_sha256": digest, "raw_text": text,
        "token_log_probabilities": [0.] * len(text), "token_probabilities": [1.] * len(text),
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256}


def proposal(rank=1, digest="crop"):
    return {"rank": rank, "bbox": [0, 0, 10, 10], "crop_rgb_sha256": digest}


def sample(sid="one", gt=None):
    return {"observed": {"sample_id": sid}, "ground_truth": gt if gt is not None else [
        {"text": "WORD", "bbox": [0, 0, 10, 10]}]}


def ds(text="WORD", bbox=None):
    return {"text": text, "bbox": bbox or [0, 0, 10, 10], "ignored": False}


def test_original_empty_and_duplicate_readings_remain_outputs():
    proposals = [proposal(), proposal(2), proposal(3, "blank")]
    rows = crop_rows(proposals, {"crop": ocr(), "blank": ocr("", "blank")})
    assert [r["raw_text"] for r in rows] == ["WORD", "WORD", ""]
    assert rows[-1]["confidence"] == 0
    report, _ = summarize_methods([sample()], {"one": {"rows": rows, "action": "accept_parseq"}},
                                  {"one": [ds()]}, {})
    assert report["parseq_highres"]["spatial"]["tp"] == 1
    assert report["parseq_highres"]["spatial"]["fp"] == 2


@pytest.mark.parametrize("change", [
    {"crop_rgb_sha256": "other"}, {"parseq_checkpoint_sha256": "other"},
    {"parseq_source_archive_sha256": "other"}, {"token_probabilities": [1.]}])
def test_changed_crop_weights_or_invalid_character_scores_fail(change):
    value = {**ocr(), **change}
    with pytest.raises(ValueError):
        crop_rows([proposal()], {"crop": value})


def test_duplicate_or_missing_proposal_identity_fails():
    with pytest.raises(ValueError):
        crop_rows([proposal(), proposal()], {"crop": ocr()})


def test_every_rejected_image_uses_all_fallback_outputs_and_preserves_empty_fn():
    rows = crop_rows([proposal()], {"crop": ocr("FAIL")})
    samples = [sample(), sample("empty", [{"text": "missing", "bbox": [0, 0, 10, 10]}])]
    routes = {"one": {"rows": rows, "action": "deepsolo"},
              "empty": {"rows": [], "action": "deepsolo"}}
    report, per_image = summarize_methods(samples, routes, {"one": [ds(), ds("extra")], "empty": []}, {})
    assert len(per_image) == 2
    assert (report["clip_then_deepsolo"]["spatial"]["tp"],
            report["clip_then_deepsolo"]["spatial"]["fp"],
            report["clip_then_deepsolo"]["spatial"]["fn"]) == (1, 1, 1)
    assert report["clip_then_deepsolo"]["spatial"]["deepsolo_images"] == 2


def test_word_occurrence_without_geometry_has_separate_counts():
    rows = crop_rows([proposal()], {"crop": ocr()})
    rows[0]["bbox"] = [20, 20, 30, 30]
    report, _ = summarize_methods([sample()], {"one": {"rows": rows, "action": "accept_parseq"}},
                                  {"one": []}, {})
    assert report["parseq_highres"]["lexical"]["tp"] == 1
    assert report["parseq_highres"]["spatial"]["tp"] == 0


def test_accepted_unlocalized_t1_has_no_fabricated_spatial_score():
    routes = {"one": {"rows": [], "action": "deepsolo"}}
    paper = {"blip2": {"one": {"scores": {"paper_selected": .8},
                                "selected_source": "T1", "T1_raw_text": "WORD"}}}
    report, _ = summarize_methods([sample()], routes, {"one": []}, paper)
    r = report["paper_blip2_then_deepsolo"]
    assert r["lexical"]["tp"] == 1 and r["lexical"]["parseq_images"] == 1
    assert r["spatial"] is None and r["accepted_t1_images"] == 1


def test_missing_fallback_sample_cannot_disappear_from_denominator():
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_methods([sample()], {"one": {"rows": [], "action": "deepsolo"}}, {}, {})


def test_qualified_json_accepts_string_path_and_checks_bytes(tmp_path):
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps({"test": 1}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checked(str(path), digest) == {"test": 1}
    path.write_text(json.dumps({"test": 2}))
    with pytest.raises(ValueError, match="SHA256"):
        checked(path, digest)
