import numpy as np
import pytest

from recognition.residual_text_vqa import residual_views
from scripts.run_yolo_residual_vqa import raw_signals, score_features


def test_mask_union_preserves_outside_pixels_and_overlap_order():
    rgb=np.random.default_rng(0).integers(0,256,(20,30,3),dtype=np.uint8)
    boxes=[[2,3,12,13],[8,5,18,15]]
    a,meta=residual_views(rgb,boxes,mask_value=128,blur_fraction=.05)
    b,_=residual_views(rgb,list(reversed(boxes)),mask_value=128,blur_fraction=.05)
    mask=np.zeros(rgb.shape[:2],bool)
    for x1,y1,x2,y2 in boxes:mask[y1:y2,x1:x2]=True
    assert np.array_equal(a['original'],rgb)
    for key in ('hard_mask','blur_mask'):
        assert np.array_equal(a[key][~mask],rgb[~mask])
        assert np.array_equal(a[key],b[key])
    assert np.all(a['hard_mask'][mask]==128)
    assert meta['masked_pixels']==mask.sum()
    assert meta['masking_may_cover_unrecognized_text_inside_boxes'] is True
    with pytest.raises(ValueError,match='Out-of-image'):
        residual_views(rgb,[[-1,0,4,4]],mask_value=128,blur_fraction=.05)


def test_zero_boxes_leave_both_residuals_identical_to_original():
    rgb=np.zeros((5,6,3),np.uint8)
    views,meta=residual_views(rgb,[],mask_value=128,blur_fraction=.05)
    assert all(np.array_equal(rgb,x)for x in views.values())
    assert meta['mask_fraction']==0


def test_risk_uses_strongest_mask_and_same_context_control():
    def q(plain,ctx):return {'plain':{'yes_probability':plain},'context':{'yes_probability':ctx}}
    scores=raw_signals({'original':q(.9,.8),'hard_mask':q(.1,.4),'blur_mask':q(.2,.3),'blank':q(.01,.2)})
    assert scores['vqa_residual_plain']==-.2
    assert scores['vqa_visual_drop_plain']==pytest.approx(.7)
    assert scores['vqa_visual_drop_context']==pytest.approx(.4)
    assert scores['vqa_context_specific']==pytest.approx(-.3)


def test_empty_ocr_cannot_be_rescued_by_high_visual_score(tmp_path):
    import json
    (tmp_path/'resolved_configuration.json').write_text(json.dumps({'smoke':True}))
    fields=('vqa_residual_plain','vqa_residual_context','vqa_visual_drop_plain','vqa_visual_drop_context','vqa_context_specific')
    names=['coverage_baseline','all_checks_softcrop',*fields,*('base_'+f for f in fields)]
    r={'eligible':False,'signals':dict.fromkeys(fields,1.),
       'source_scores':{'coverage_baseline':None,'all_checks_softcrop':None}}
    features,_=score_features({'families':names},{'selection_ids':['empty']},tmp_path,'train',{'empty':r})
    assert set(features['empty']['scores'].values())=={None}


def test_gpu_oom_propagates_with_explicit_failure_receipt(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from scripts import run_yolo_residual_vqa as runner
    cfg={'execution':{'cpu_threads':1,'seed':0,'device':'cpu'}}
    monkeypatch.setattr(runner,'prepare',lambda args:(cfg,{'code_sha256':{}},tmp_path,
        {'train':{}},{'train':{}}))
    def fail(*args):raise torch.cuda.OutOfMemoryError('injected qualification OOM')
    monkeypatch.setattr(runner,'probe_split',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError,match='qualification OOM'):
        runner.run(SimpleNamespace(preflight=False))
    receipt=json.loads((tmp_path/'failure.json').read_text())
    assert receipt['status']=='FAILED' and receipt['out_of_memory'] is True
    assert receipt['optimizer_steps']==0 and receipt['partial_results_are_not_final'] is True
    assert not(tmp_path/'results.json').exists()
