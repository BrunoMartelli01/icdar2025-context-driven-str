import copy
import inspect
import json

import cv2
import numpy as np
import pytest

from localize.occlusion_four_tests import (checked_box, correction_choice, expand_box,
    features, instance_pool, mser_pools, row, same_reading_nms, select_groups, decode_instance_labels, bind_baseline)
from scripts.run_occlusion_four_tests import preselect, checked_config
from scripts.run_mask_crop_budget import measure


def config():
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2]/'configs/approximate/occlusion_four_tests_v1.json').read_text())


def reading(text,confidence):
    return dict(raw_text=text,sequence_confidence_native=confidence,
                token_probabilities=[confidence]*len(text),token_log_probabilities=[np.log(confidence)]*len(text))


def test_letterbox_instances_preserved_and_do_not_stretch():
    labels=np.zeros((32,32),np.int32);labels[12:20,4:12]=1;labels[12:20,16:24]=2
    probability=np.where(labels>0,.99,.01).astype(np.float32)
    rgb=np.zeros((20,40,3),np.uint8)
    result=instance_pool(probability,labels,rgb,config()['hypotheses'])
    assert len(result['baseline'])==2
    for r in result['baseline']:
        assert checked_box(r['bbox'],rgb.shape[:2])==r['bbox']
        assert r['bbox'][1] >= 4 and r['bbox'][3] <= 16
    assert all(p in result['pool'] for p in result['baseline'])


def test_torch_canvas_boundary_matches_numpy_decode():
    import torch
    canvas=np.ones((32,32),bool)
    words=np.zeros((2,32,32),np.float32);words[:,8:16,8:16]=.99
    gap=np.zeros((32,32),np.float32)
    pp=dict(method='kernel_bfs',text_threshold=.65,kernel_threshold=.5,center_threshold=.5,gap_threshold=.5)
    a=decode_instance_labels(words,gap,canvas,pp,.9,'preserve')
    b=decode_instance_labels(words,gap,torch.from_numpy(canvas),pp,.9,'preserve')
    np.testing.assert_array_equal(a,b)
    assert np.array_equal(a>0,words[0]>=.65)


def test_split_alternatives_cannot_conflict_and_identity_available():
    base=[{'id':'a'},{'id':'b'}];pool=base+[{'id':'split1'},{'id':'split2'},{'id':'merge'}]
    groups=[dict(id='split',parents=['a'],children=['split1','split2']),dict(id='merge',parents=['a','b'],children=['merge'])]
    scores=dict(a=.1,b=.1,split1=.9,split2=.9,merge=.95)
    result,chosen=select_groups(base,pool,groups,scores,cost=.1,margin=0)
    assert {r['id'] for r in result}=={'split1','split2','b'}
    assert len(chosen)==1
    result,_=select_groups(base,pool,groups,scores,cost=0,margin=100)
    assert result==base


def test_proposal_budget_never_discards_baseline():
    labels=np.zeros((32,32),np.int32);labels[5:9,3:6]=1;labels[5:9,8:11]=2
    probability=np.full(labels.shape,.99,np.float32);rgb=np.zeros((32,32,3),np.uint8)
    c=copy.deepcopy(config()['hypotheses']);c['maximum_extra']=0
    result=instance_pool(probability,labels,rgb,c)
    assert result['pool']==result['baseline'] and not result['groups']


def test_serialized_baseline_rebinds_to_recognized_pool():
    item=json.loads(json.dumps(dict(baseline=[dict(id='a')],pool=[dict(id='a')])))
    item['pool'][0]['crop_array_sha256']='real-crop-hash'
    bind_baseline(item)
    assert item['baseline'][0]['crop_array_sha256']=='real-crop-hash'
    assert item['baseline'][0] is item['pool'][0]


def test_mser_is_deterministic_gt_free_and_original_coordinates():
    rgb=np.full((70,240,3),255,np.uint8)
    cv2.putText(rgb,'TEST WORD',(5,45),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    c=config()['mser'];a=mser_pools(rgb,c);b=mser_pools(rgb,c)
    assert a==b and set(a)=={'mser','ss_mser_inspired'}
    assert len(a['mser'])>0
    for pool in a.values():
        assert len(pool)<=c['maximum_word_candidates']
        for p in pool:checked_box(p['bbox'],rgb.shape[:2])
    assert 'gt' not in inspect.signature(mser_pools).parameters


def test_empty_rgb_proposals_are_valid_empty_outputs():
    assert all(not pool for pool in mser_pools(np.zeros((32,64,3),np.uint8),config()['mser']).values())


def test_pre_features_have_no_ocr_dependency():
    rgb=np.zeros((20,40,3),np.uint8);r=row([0,0,10,10],rgb.shape[:2],'a','baseline')
    before=features(r,rgb);r.update(raw_text='SECRET',confidence=.999)
    assert features(r,rgb)==before
    assert np.isfinite(before).all()
    assert len(features(r,rgb,np.ones((32,32),np.float32),text_threshold=.65))==len(before)
    with pytest.raises(ValueError):features(r,rgb,np.ones((32,32),np.float32))


def test_preselection_budget_counts_unique_crops_and_does_not_read_ocr():
    values=[dict(id='a',crop_array_sha256='x'),dict(id='b',crop_array_sha256='x'),dict(id='c',crop_array_sha256='y')]
    r={'arms':{'mk':{'pool':values}}}
    assert preselect(r,'mk',dict(a=.9,b=.8,c=.7),1)=={'a','b'}
    assert 'cache' not in inspect.signature(preselect).parameters


def test_view_expansion_clamps_and_keeps_original_box():
    original=[0,2,10,12];before=original.copy()
    assert expand_box(original,(15,20),.25)==[0,0,13,15]
    assert original==before
    with pytest.raises(ValueError):expand_box(original,(15,20),float('nan'))


def test_correction_votes_context_and_keep_boundary():
    trace={'views':[{'crop_array_sha256':v} for v in 'abc']}
    readings=dict(a=reading('CAT',.9),b=reading('CAR',.99),c=reading('CAR',.99))
    assert correction_choice(trace,readings,2,.05,0)=='CAR'
    assert correction_choice(trace,readings,2,.2,0)=='CAT'
    assert correction_choice(trace,readings,2,.05,.1)=='CAT'  # unavailable context KEEP
    trace['context']={'CAT':-1,'CAR':-3}
    assert correction_choice(trace,readings,2,.05,.1)=='CAT'
    trace['context']={'CAT':-3,'CAR':-1}
    assert correction_choice(trace,readings,2,.05,.1)=='CAR'


def test_nms_deduplicates_only_same_reading_and_spatial_overlap():
    values=[dict(id='a',bbox=[0,0,10,10],crop_array_sha256='a'),dict(id='b',bbox=[0,0,10,10],crop_array_sha256='b'),
            dict(id='c',bbox=[0,0,10,10],crop_array_sha256='c')]
    kept=same_reading_nms(values,dict(a=reading('CAT',.9),b=reading('cat',.9),c=reading('CAR',.9)),dict(a=.9,b=.8,c=.7),.5)
    assert [r['id'] for r in kept]==['a','c']


def test_matching_retains_misses_empty_duplicates_and_punctuation():
    p=[dict(bbox=[0,0,10,10],crop_array_sha256=h) for h in 'abc']
    gt=[dict(bbox=[0,0,10,10],text='CAT'),dict(bbox=[20,0,30,10],text='DOG')]
    result=measure(p,gt,dict(a={'raw_text':'CAT'},b={'raw_text':'cat'},c={'raw_text':''}))
    assert result['correct']==1 and result['proposals']==3 and result['gt_words']==2
    assert measure(p[:1],gt,dict(a={'raw_text':'CAT!'}))['correct']==0


def test_config_rejects_non_approximate_namespace(tmp_path):
    path=tmp_path/'configuration.json';path.write_text(json.dumps({'reproduction_mode':'approximate'}))
    with pytest.raises(ValueError,match='isolated'):checked_config(path,'approximate')
