"""Device qualification fails on changed words or filter membership."""

import pytest

from scripts.qualify_icdar2015_target70_ocr_device import compare_outputs


def output(text="word",probability=.96):
    import math
    return {"raw_text":text,"token_ids":[1,2,3],"sequence_confidence_native":probability,
        "mean_token_log_probability":math.log(probability)}


POLICY={"confidence":"sequence_confidence","threshold":.95,"minimum_alphanumeric_length":3}


def test_numeric_roundoff_is_measured_while_decisions_are_identical():
    differences=compare_outputs(output(),output(probability=.9600001),POLICY)
    assert differences["sequence_confidence_native"]==pytest.approx(1e-7)


def test_changed_text_fails_even_if_normalized_text_is_equal():
    with pytest.raises(ValueError,match="text differs"):
        compare_outputs(output(),output(text="WORD"),POLICY)


@pytest.mark.parametrize("field",["sequence_confidence","mean_token_confidence"])
def test_threshold_crossing_is_a_failure(field):
    with pytest.raises(ValueError,match="policy decision differs"):
        compare_outputs(output(probability=.950001),output(probability=.949999),{**POLICY,"confidence":field})


def test_token_ids_must_be_identical():
    changed={**output(),"token_ids":[2,1,3]}
    with pytest.raises(ValueError,match="token IDs differ"):compare_outputs(output(),changed,POLICY)


def test_empty_decode_is_not_omitted():
    empty={**output(text=""),"mean_token_log_probability":None}
    assert compare_outputs(empty,empty,POLICY)=={"sequence_confidence_native":0.}
