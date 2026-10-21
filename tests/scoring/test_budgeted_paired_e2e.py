from argparse import Namespace
from pathlib import Path

import pytest

from diagnostics.budgeted_paired_e2e import (
    DEEP_SHA, aggregate, cached, match_words, shard, validate_deep, validate_route,
)
from diagnostics.caption_routing_train import checked, seal_write
from scoring.budgeted_paired import route


def test_word_denominator_duplicates_punctuation_and_extra_predictions():
    r = match_words("Shop shop shop, ROAD extra", ["shop", "road", "missing"])
    assert r["correct_words"] == 2
    assert r["gt_words"] == 3
    assert r["extra_wrong_tokens"] == 3
    assert not r["all_gt_recovered"]
    assert not r["exact_image_word_multiset"]
    summary = aggregate([r, match_words("", ["omitted"])])
    assert summary["word_accuracy"] == .5
    assert summary["complete_image_accuracy"] == 0


def test_complete_image_is_not_any_word_membership():
    a = match_words("B A", ["a","b"])
    b = match_words("B A EXTRA", ["a","b"])
    c = match_words("B", ["a","b"])
    assert a["exact_image_word_multiset"]
    assert b["all_gt_recovered"] and not b["exact_image_word_multiset"]
    assert not c["all_gt_recovered"]
    assert aggregate([a,b,c])["complete_image_accuracy"] == 1/3


def test_empty_population_and_multitoken_gt():
    assert aggregate([])["word_accuracy"] is None
    with pytest.raises(ValueError):
        match_words("word", ["two words"])


def test_train_test_cache_separation_even_when_ids_overlap(tmp_path):
    args = Namespace(output_dir=tmp_path)
    row = {"sample_id":"100","image_sha256":"train-image"}
    path = shard(args,"TRAIN","caption",row)
    seal_write(path,{"contract_sha256":"frozen",**row,"text":"caption"})
    assert cached(args,"TRAIN","caption",row,"frozen")["text"] == "caption"
    assert cached(args,"TEST","caption",row,"frozen") is None
    with pytest.raises(RuntimeError):
        cached(args,"TRAIN","caption",{**row,"image_sha256":"test-image"},"frozen")


def test_seal_resumes_identical_but_rejects_changed_results(tmp_path):
    path = tmp_path/"result.json"
    seal_write(path,{"tau":.23})
    seal_write(path,{"tau":.23})
    with pytest.raises(RuntimeError):
        seal_write(path,{"tau":.24})


def test_tau_applied_without_budget_recalibration_on_test():
    original = route(tau=.8,T1="a",T2="caption",T3="b",S1=.4,S3=.2,L1=.2,L3=1.).to_dict()
    decision = validate_route(original,.23055392728092478)
    assert decision["tau"] == .23055392728092478
    assert not decision["needs_fallback"]
    assert decision["selected_output"] == "a"
    original["C"] += .01
    with pytest.raises(RuntimeError):
        validate_route(original,.23055392728092478)


def test_deepsolo_all_strings_not_collapsed_to_highest_score():
    deep = {"success":True,"error":None,"artifact":{"checkpoint_sha256":DEEP_SHA,
            "inference_options":{"device":"cuda:0","lexicon_type":None,"score_threshold":None}},
            "reduction_policy":"highest_score","selected_output":"first",
            "recognized_strings":["first","second"]}
    validate_deep(deep)
    assert match_words(" ".join(deep["recognized_strings"]),["first","second"])["correct_words"] == 2
    assert match_words(deep["selected_output"],["first","second"])["correct_words"] == 1
    deep["artifact"]["checkpoint_sha256"] = "wrong"
    with pytest.raises(RuntimeError):
        validate_deep(deep)


def test_evaluation_combines_routes_without_oracle_selection(tmp_path, monkeypatch):
    import diagnostics.budgeted_paired_e2e as mod
    from diagnostics.lavis_captioner_candidates import digest
    args = Namespace(output_dir=tmp_path, split="TRAIN")
    gt = {"TRAIN":{"100":["a","b"],"101":["c","d"]},"TEST":{}}
    seal_write(tmp_path/"gt.json",{"contract_sha256":"frozen",**gt})
    rows = []
    for sid,text,s in (("100","a",.9),("101","wrong",.1)):
        dec = route(tau=.23,T1=text,T2="caption",T3="",S1=s,S3=0,L1=s,L3=0).to_dict()
        row = {"sample_id":sid,"image_sha256":sid,"decision":dec}
        rows.append(row)
        deep = {"recognized_strings":gt["TRAIN"][sid],"selected_output":gt["TRAIN"][sid][0]}
        seal_write(shard(args,"TRAIN","deepsolo",row),{"contract_sha256":"frozen",**row,"response":deep})
    monkeypatch.setattr(mod,"load",lambda a: ({"config":{"tau":.23},"gt_digest":digest(gt),
                     "word_protocol":"test","legacy_control":"test"},"frozen",{"TRAIN":rows}))
    monkeypatch.setattr(mod,"cached",lambda a,split,stage,row,contract: checked(shard(a,split,stage,row),contract))
    mod.evaluate(args)
    result = checked(tmp_path/"train-results.json","frozen")
    assert result["fallback_images"] == 1
    assert result["metrics"]["routed_all_strings"]["correct_words"] == 3
    assert result["metrics"]["routed_all_strings"]["word_accuracy"] == .75
    assert result["metrics"]["routed_all_strings"]["complete_image_accuracy"] == .5
    assert result["metrics"]["deepsolo_all_strings"]["correct_words"] == 4
    assert result["metrics"]["routed_highest_score"]["correct_words"] == 2
    assert result["accepted_parseq"]["gt_words"] == 2
    assert result["fallback_deepsolo"]["correct_words"] == 2
