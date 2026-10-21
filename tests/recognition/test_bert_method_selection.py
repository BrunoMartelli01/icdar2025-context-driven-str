"""Selective edits and immutable-KEEP ranking boundaries; no dataset labels."""
import math

import pytest

from recognition.bert_method_selection import (
    MethodPolicy, bounded_candidates, insertion_selection, only_insertions, pll_selection,
)
from recognition.neuspell_correction import SpellPolicy


def method(**kwargs):
    return MethodPolicy(**{**dict(minimum_source_length=4,insertion_max_edits=2,
        minimum_insertion_probability=.8,candidate_minimum_probability=.01,maximum_candidates=6,
        pll_edit_penalty=.5,pll_minimum_gain=1.5,rerank_oov_only=True),**kwargs})


def spell():
    return SpellPolicy(3,6,1,2,.3,.5,1.5,3,.5,8,80)


def neural(predicted="island",reference=(),probability=.9):
    return dict(status="scored",prediction=predicted,reference_label_ids=list(reference),
        probability=probability,special_prediction=False,top_labels=[
        dict(text=predicted,probability=probability,label_id=0),
        dict(text="islands",probability=.05,label_id=1),dict(text="island",probability=.03,label_id=2)])


@pytest.mark.parametrize("original,candidate,expected",[
    ("SLAND","island",True),("parkin","parking",True),("TUDIO","studio",True),
    ("LIVE","love",False),("hong","hung",False),("Grandy","grandly",True),
    ("island","island",False),("slnd","island",True),("abc","axbzcq",False),
])
def test_only_insertions_preserve_every_original_character(original,candidate,expected):
    assert only_insertions(original,candidate,2) is expected


def test_oov_insertion_gate_protects_known_words_probability_and_case():
    assert insertion_selection("SLAND",neural(),method())["text"] == "ISLAND"
    assert insertion_selection("sland",neural(probability=.8),method())["text"] == "island"
    assert insertion_selection("sland",neural(probability=.799),method())["text"] == "sland"
    assert insertion_selection("SLAND",neural(reference=[77]),method())["reason"] == "known_word_KEEP"
    assert insertion_selection("HONG",neural("hung"),method())["text"] == "HONG"
    assert insertion_selection("abc",neural("abcd"),method())["reason"] == "short_source_KEEP"
    with pytest.raises(ValueError,match="probability"):
        insertion_selection("sland",neural(probability=float("nan")),method())


def test_candidates_keep_original_exactly_deduplicate_case_and_bound_edits():
    candidates=bounded_candidates("SLAND",neural(),method(),spell())
    assert candidates[0] == {"text":"SLAND","edit_distance":0,"source":"immutable_OCR_KEEP"}
    assert [c["text"] for c in candidates] == ["SLAND","ISLAND"]
    assert len(bounded_candidates("SLAND",neural(),method(maximum_candidates=1),spell())) == 1
    assert len(bounded_candidates("SLAND",neural(reference=[5]),method(),spell())) == 1
    assert len(bounded_candidates("SLAND",dict(status="protected_word_shape"),method(),spell())) == 1


def test_pll_gate_keeps_ties_requires_scored_reference_and_propagates_errors():
    candidates=[dict(text="sland",edit_distance=0),dict(text="island",edit_distance=1)]
    score=lambda x:dict(status="scored",mean_log_probability=x)
    assert pll_selection(candidates,[score(-4),score(-2)],method())["selected_index"] == 0
    assert pll_selection(candidates,[score(-4),score(-1.99)],method())["selected_index"] == 1
    assert pll_selection(candidates,[dict(status="unknown_target",mean_log_probability=None),score(-1)],method())["text"] == "sland"
    with pytest.raises(ValueError,match="cardinality"):
        pll_selection(candidates,[score(-3)],method())
    with pytest.raises(ValueError,match="Nonfinite"):
        pll_selection(candidates,[score(-4),score(math.inf)],method())


@pytest.mark.parametrize("key,value",[("minimum_source_length",0),("minimum_insertion_probability",1.01),
                                     ("pll_minimum_gain",-1),("rerank_oov_only",None)])
def test_invalid_policy_fails(key,value):
    with pytest.raises(ValueError):
        method(**{key:value})
