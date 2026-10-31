import copy
import math

import pytest

from segmentation.icdar2015_multiview import emitted,normalized


def prediction(stream,confidence=.9,box=(0,0,10,10),text='Text'):
    return dict(stream_id=stream,sequence_confidence=confidence,mean_token_log_probability=math.log(confidence),bbox=list(box),text=text)


def run(ps,**kw):return emitted(ps,**dict(dict(field='sequence_confidence',threshold=.7,minimum_length=3,nms_iou=.5),**kw))


def test_single_stream_components_and_input_are_preserved():
    p=[prediction('a'),prediction('a',.8)];before=copy.deepcopy(p)
    assert run(p)==p and p==before


def test_cross_stream_overlap_keeps_stronger_original_hypothesis():
    weak=prediction('a',.8);strong=prediction('b',.95)
    assert run([weak,strong])==[strong]
    strong['bbox']=[20,0,30,10];assert run([weak,strong])==[strong,weak]


def test_exact_overlap_threshold_and_stable_tie():
    a=prediction('a');b=prediction('b',box=(0,0,5,10))
    assert run([a,b])==[a]
    assert run([a,b],nms_iou=.50001)==[a,b]
    assert run([b,a])==[b]


def test_normalization_empty_outputs_and_confidence_boundary():
    assert normalized('Straße-9!')=='strasse9'
    a=prediction('a',.7,text='A-b!');assert run([a],minimum_length=2)==[a]
    assert run([a],threshold=.700001,minimum_length=2)==[]
    blank=prediction('b',.9,text='');blank['mean_token_log_probability']=None
    assert run([blank],field='mean_token_confidence',minimum_length=0)==[]


@pytest.mark.parametrize('overlap',[0,1.1])
def test_invalid_policy_fails(overlap):
    with pytest.raises(ValueError):run([],nms_iou=overlap)
