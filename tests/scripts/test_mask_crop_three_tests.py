import copy
import json

import numpy as np
import pytest
import torch

from localize.mask_crop_hypotheses import baseline, hypotheses, select_groups
from scripts.run_mask_crop_budget import aggregate, config, measure
from scripts.run_mask_crop_learned_tests import context_geometry, normalized_box, pixel_box
from scripts.evaluate_cached_totaltext_padding import proposals_for_padding
from segmentation.recognition_crop_refiner import RecognitionCropRefiner, box_loss, sample_boxes


@pytest.fixture
def cfg():
    with open('configs/approximate/mask_crop_three_tests_v1.json') as stream:
        return json.load(stream)


def test_baseline_replays_historical_stretch_coordinates_and_ties(cfg):
    p=np.zeros((128,128),np.float32)
    p[10:20,15:30]=1;p[40:50,70:85]=1;p[0:3,0:4]=1
    actual=baseline(p,(777,1234),cfg['hypotheses'])
    plan={'padding_native_pixels':[1], 'min_unpadded_bbox_area_native':0,'max_proposals_per_image':10}
    expected=proposals_for_padding((p>=.5).astype('uint8'),(777,1234),plan)['1']['proposals']
    fields=['bbox_native','bbox','padded_bbox_native','padded_bbox_original','contour_index']
    assert [[r[k] for k in fields] for r in actual]==[[r[k] for k in fields] for r in expected]


def test_seed_split_reconstructs_support_and_keeps_baseline(cfg):
    p=np.zeros((128,128),np.float32)
    p[30:50,20:40]=.95;p[30:50,50:70]=.95;p[38:42,40:50]=.55
    raw,pool,groups=hypotheses(p,np.zeros((256,512,3),np.uint8),cfg['hypotheses'])
    assert len(raw)==1 and pool[0]==raw[0]
    split=next(g for g in groups if g['method']=='threshold_split')
    assert split['parents']==[raw[0]['id']] and len(split['children'])==2
    assert len(pool)<=cfg['hypotheses']['maximum_pool']
    lookup={r['id']:r for r in pool}
    assert all(lookup[k]['bbox'][2]<=512 and lookup[k]['bbox'][3]<=256 for k in split['children'])


def test_no_conflicting_replacements_and_identity_is_available():
    base=[{'id':'a'},{'id':'b'}]
    pool=base+[{'id':'x'},{'id':'y'},{'id':'z'}]
    groups=[{'id':'s','parents':['a'],'children':['x','y']},
            {'id':'m','parents':['a','b'],'children':['z']}]
    scores={'a':.1,'b':.2,'x':.9,'y':.9,'z':.8}
    result,chosen=select_groups(base,pool,groups,scores,cost=0,margin=0)
    assert {r['id'] for r in result}=={'b','x','y'} and len(chosen)==1
    result,chosen=select_groups(base,pool,groups,scores,cost=0,margin=100)
    assert result==base and not chosen


def test_empty_outputs_keep_gt_and_punctuation_is_retained():
    words=[{'bbox':[0,0,10,10],'text':'WORD!'}]
    assert measure([],words,{})['gt_words']==1
    p=[{'bbox':[0,0,10,10],'crop_array_sha256':'a'}]
    assert measure(p,words,{'a':{'raw_text':'word'}})['correct']==0
    assert measure(p,words,{'a':{'raw_text':'word!'}})['correct']==1
    assert aggregate([measure([],words,{})])['recall']==0


def test_duplicate_crops_cannot_count_one_word_twice():
    p=[{'bbox':[0,0,10,10],'crop_array_sha256':'a'}]*2
    m=measure(p,[{'bbox':[0,0,10,10],'text':'a'}],{'a':{'raw_text':'a'}})
    assert m['proposals']==2 and m['correct']==1


def test_strict_mode_is_rejected_before_loading_weights():
    with pytest.raises(ValueError,match='APPROXIMATE'):
        config('configs/approximate/mask_crop_three_tests_v1.json','strict')


def test_differentiable_sampler_identity_and_gradient():
    x=torch.arange(12,dtype=torch.float32).reshape(1,1,3,4)
    boxes=torch.tensor([[0.,0.,1.,1.]],requires_grad=True)
    sampled=sample_boxes(x,boxes,(3,4))
    torch.testing.assert_close(sampled,x)
    sampled.square().sum().backward()
    assert boxes.grad is not None and torch.isfinite(boxes.grad).all() and boxes.grad.abs().sum()>0


def test_refiner_starts_as_identity_and_can_learn(cfg):
    c=cfg['refiner'];model=RecognitionCropRefiner(translation=c['translation'],log_scale=c['log_scale'])
    context=torch.rand(2,4,64,256);boxes=torch.tensor([[.2,.2,.7,.8],[.1,.1,.9,.9]])
    result=model(context,boxes)
    torch.testing.assert_close(result,boxes)
    loss=(result-torch.tensor([[.1,.1,.8,.8],[.2,.2,.8,.8]])).square().mean()
    loss.backward()
    assert model.head[-1].weight.grad.abs().sum()>0


def test_coverage_and_neighbor_contamination_are_distinct():
    truth=torch.tensor([[.1,.1,.4,.5]])
    neighbor=torch.tensor([[[.5,.1,.8,.5]]]);valid=torch.tensor([[True]])
    coverage,contamination,_=box_loss(truth,truth,neighbor,valid)
    assert coverage==0 and contamination==0
    _,contamination,_=box_loss(torch.tensor([[.1,.1,.8,.5]]),truth,neighbor,valid)
    assert contamination>0


def test_nonfinite_masks_fail_explicitly(cfg):
    p=np.zeros((128,128),np.float32);p[0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        baseline(p,(200,400),cfg['hypotheses'])


@pytest.mark.parametrize('box,hw', [([100,20,400,80],[200,1000]),
                                   ([0,0,7,11],[199,333]),
                                   ([299,150,333,199],[199,333])])
def test_context_roundtrip_keeps_identity_rgb_pixels(box,hw):
    context=context_geometry(box,hw,2.0)
    assert pixel_box(normalized_box(box,context),context,hw)==box


def test_refiner_extreme_parameters_keep_nonempty_bounded_crops(cfg):
    c=cfg['refiner'];model=RecognitionCropRefiner(translation=c['translation'],log_scale=c['log_scale'])
    with torch.no_grad():model.head[-1].bias.fill_(100)
    result=model(torch.rand(2,4,64,256),torch.tensor([[0.,0.,.1,.2],[.9,.9,1.,1.]]))
    assert torch.isfinite(result).all() and torch.all(result[:,:2]<result[:,2:])
    assert result.min()>=0 and result.max()<=1
