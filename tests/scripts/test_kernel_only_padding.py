import json

import numpy as np
import pytest

from scripts import evaluate_kernel_only_padding as m


def test_kernel_decoding_uses_only_kernel_and_rgb_canvas():
    p=np.zeros((16,16),np.float32);canvas=np.ones((16,16),bool)
    p[4:8,4:8]=.5;p[13,13]=.9;canvas[13,13]=False
    labels=m.kernel_instances(p,canvas,.5)
    assert np.array_equal(labels>0,(p>=.5)&canvas)
    assert labels.max()==1


@pytest.mark.parametrize('value',[float('nan'),-.1,1.1])
def test_invalid_probabilities_fail(value):
    p=np.zeros((4,4));p[0,0]=value
    with pytest.raises(ValueError):m.kernel_instances(p,np.ones_like(p,dtype=bool),.5)


def test_padding_changes_crop_geometry_without_claiming_raw_detection_gain():
    labels=np.zeros((16,16),np.int32);labels[6:10,6:10]=1
    raw=m.parseq_word_proposals(labels,(16,16),padding=0,min_area=0,max_instances=None)[0]
    padded=m.parseq_word_proposals(labels,(16,16),padding=2,min_area=0,max_instances=None)[0]
    assert raw['bbox']==padded['bbox_original']==[6,6,10,10]
    assert padded['bbox']==[4,4,12,12]
    gt=[{'bbox':[4,4,12,12],'text':'word'}]
    assert m.score_predictions([{**raw,'text':'WORD'}],gt)[0]['geometry_iou_ge_50']==0
    counts,_=m.score_predictions([{**padded,'text':'WORD'}],gt)
    assert counts['geometry_iou_ge_50']==counts['exact_casefold_iou_ge_50']==1
    counts['unpadded_geometry_iou_ge_50']=len(m._geometry_match([{**padded,'bbox':padded['bbox_original']}],gt,.5))
    assert m.add_geometry_metrics([counts],1)['unpadded_geometry_iou_ge_50']==0
    assert m.parseq_word_proposals(labels,(16,16),padding=50,min_area=0,max_instances=None)[0]['bbox']==[0,0,16,16]


def test_ignore_applies_after_decoding_and_partial_instances_remain():
    labels=np.zeros((8,8),np.int32);labels[1:3,1:3]=1;labels[5:7,5:7]=2
    valid=np.zeros((8,8),bool);valid[1,1]=True
    assert m.ignored_instances(labels,valid)==[2]
    assert labels[5,5]==2


def test_full_word_denominator_and_one_to_one_matching():
    p={'bbox':[0,0,10,10],'text':'word'}
    gt=[{'bbox':[0,0,10,10],'text':'WORD'},{'bbox':[20,20,30,30],'text':'missed'}]
    counts,_=m.score_predictions([p,p],gt);counts['unpadded_geometry_iou_ge_50']=1
    result=m.add_geometry_metrics([counts],2)
    assert result['geometry_iou_ge_50']==result['exact_casefold_iou_ge_50']==1
    assert result['geometry_iou_ge_50_recall']==.5
    with pytest.raises(ValueError,match='denominator'):m.add_geometry_metrics([counts],3)


def test_config_isolation_and_frozen_source():
    cfg,source,_=m.read_config(m.DEFAULT_CONFIG,'approximate')
    assert cfg['padding_native_pixels']==[0,2,4,6,8,12]
    assert source['expected_validation_words']==983
    with pytest.raises(ValueError,match='approximate'):m.read_config(m.DEFAULT_CONFIG,'strict')
