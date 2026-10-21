import json
import numpy as np
import pytest

from diagnostics.fixed_group_curve_geometry import fit, render, bernstein, sampling_map
from diagnostics.ocr_context_evidence_next_train import parse_patch, fixtures, pilot_blocks, messages


CFG = {'pad':6,'envelope_bins':8,'minimum_occupied_bins':6,'minimum_pixels':16,'max_condition_number':10000}
PC = {'max_patch_edits':2,'max_patch_distance':2}


@pytest.mark.parametrize('kind',['ORIENTED','BEZIER'])
def test_straight_fixed_group_and_determinism(kind):
    mask=np.zeros((30,100),np.uint8);mask[10:20,10:90]=1; before=mask.copy()
    s,reason=fit(mask,(30,30),(200,200),kind,CFG)
    assert reason=='applied' and s['encloses_assigned_foreground']
    assert fit(mask,(30,30),(200,200),kind,CFG)[0]==s
    assert np.array_equal(mask,before)
    mx,my=sampling_map(s)
    assert np.allclose(my[0],my[0,0])
    rgb=np.zeros((200,200,3),np.uint8);rgb[40:50,40:120]=255
    assert render(rgb,s).shape==(s['height'],s['width'],3)
    assert (mx[0,0]<40 and mx[0,-1]>119)


def test_curve_has_real_warp_and_encloses_foreground():
    mask=np.zeros((80,100),np.uint8)
    for x in range(10,90):
        y=int(20+.012*(x-50)**2);mask[y:y+8,x]=1
    s,_=fit(mask,(30,30),(200,200),'BEZIER',CFG)
    assert s and s['encloses_assigned_foreground']
    _,my=sampling_map(s)
    assert np.ptp(my[0])>5
    assert np.all(bernstein(np.linspace(0,1,200))@(np.array(s['lower'])-s['upper'])>0)


def test_sparse_degenerate_and_border_fallback():
    assert fit(np.eye(3),(0,0),(20,20),'BEZIER',CFG)[0] is None
    assert fit(np.ones((20,20)),(30,30),(100,100),'ORIENTED',CFG)[1]=='ambiguous_axis'
    assert fit(np.ones((10,40)),(0,0),(50,50),'BEZIER',CFG)[1]=='outside_image_fallback'


@pytest.mark.parametrize('fmt,edit',[('NAMED_EDITS',{'start':0,'end':1,'text':'b'}),('TRIPLE_EDITS',[0,1,'b'])])
def test_literal_edits(fmt,edit):
    r=parse_patch(json.dumps({'target_id':'x','edits':[edit]}),'cat','x',fmt,PC)
    assert r['valid'] and r['text']=='bat'


def test_insertion_is_not_replacement():
    raw=json.dumps({'target_id':'x','edits':[[0,0,'cat']]})
    r=parse_patch(raw,'cat','x','TRIPLE_EDITS',PC)
    assert not r['valid'] and r['text']=='cat'


def test_anchor_unicode_and_keep():
    for old,new,expect in [('ü','u','Zurich'),('','','Zürich')]:
        r=parse_patch(json.dumps({'target_id':'x','old':old,'new':new}),'Zürich','x','ANCHORED_REPLACE',PC)
        assert r['valid'] and r['text']==expect
    assert not parse_patch('{"target_id":"x","old":"a","new":"e"}','banana','x','ANCHORED_REPLACE',PC)['valid']


@pytest.mark.parametrize('raw',[
    '{"target_id":"x","edits":[[0,2,"A"],[1,3,"B"]]}',
    '{"target_id":"x","edits":[[true,1,"A"]]}',
    '{"target_id":"y","edits":[]}',
    '{"target_id":"x","edits":[],"extra":0}',
    '{"target_id":"x","edits":[],"edits":[]}',
    '{"target_id":"x","edits":[[0,1,"a "]]}'])
def test_invalid_patch_keeps_original(raw):
    r=parse_patch(raw,'cat','x','TRIPLE_EDITS',PC)
    assert not r['valid'] and r['text']=='cat'


def test_no_gt_selection_or_prompt_and_synthetic_count():
    assert len(fixtures())==24
    cfg={'pilot_min_length':2,'pilot_max_length':25,'pilot_crops_per_confidence_extreme':1}
    blocks=[{'candidate_id':str(i),'text':'word','confidence':{'mean_token_log_probability':-i}} for i in range(3)]
    d={'samples':[{'blocks':blocks,'gt_text':'SECRET'}]}
    assert [b['candidate_id'] for _,b in pilot_blocks(d,cfg)]==['2','0']
    prompt=messages('word','0','TRIPLE_EDITS','correct','caption')
    assert 'SECRET' not in json.dumps(prompt)
