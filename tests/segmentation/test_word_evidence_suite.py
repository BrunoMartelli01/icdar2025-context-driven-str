"""Scientific boundaries for the three frozen Text+Kernel evidence tests."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from localize.word_evidence import candidates,associate,adjust_boxes
from segmentation.word_evidence import supervision,crop_loss,fit_head,predict_head
from scripts.run_word_evidence_suite import read_config,summarize,proposals,bind


@pytest.fixture
def sample():
    labels=np.zeros((16,16),np.int32);labels[4:8,4:8]=1;labels[4:8,11:14]=2
    p=np.zeros((2,16,16),np.float32);p[0]=labels>0;p[1,5:7,5:7]=1;p[1,5:7,12:13]=1
    g=np.zeros((16,16),np.float32);g[4:8,2:4]=1;g[4:8,5:7]=1;g[4:8,12:14]=1
    cfg={'glyph_threshold':.5,'kernel_threshold':.5,'nearest_instances':4,'owner_majority':.5,'association_threshold':.8}
    return labels,p,g,np.ones((16,16),bool),cfg


def test_inference_interfaces_have_no_ground_truth_inputs(sample):
    for func in (candidates,associate,adjust_boxes,predict_head):
        assert not {'owners','ground_truth','validity','transcriptions'} & set(inspect.signature(func).parameters)
    labels,p,g,canvas,c=sample
    out=candidates(p,g,labels,canvas,c)
    assert out['pair_features'].shape[1]==12 and out['box_features'].shape==(2,12)
    assert np.isfinite(out['pair_features']).all()


def test_gt_changes_supervision_but_never_candidate_features(sample):
    labels,p,g,canvas,c=sample;can=candidates(p,g,labels,canvas,c)
    owners=labels.copy();owners[4:8,2:4]=1
    before=copy.deepcopy(can)
    sup=supervision(can,labels,owners,canvas,c)
    assert sup['pair_target'].sum()>0 and np.any(sup['pair_target']==0)
    for key in before:np.testing.assert_array_equal(before[key],can[key])
    other=supervision(can,labels,np.zeros_like(owners),canvas,c)
    assert other['pair_target'].sum()==0


def test_fragment_extension_is_single_owner_and_preserves_existing_pixels(sample):
    labels,p,g,canvas,c=sample;can=candidates(p,g,labels,canvas,c)
    scores=np.array([.99 if sid==1 else .1 for _,sid in can['pair_ids']],np.float32)
    new,audit=associate(labels,can['fragments'],can['pair_ids'],scores,.8)
    np.testing.assert_array_equal(new[labels>0],labels[labels>0])
    assert np.all(new[4:8,2:4]==1) and audit['attached_fragments']==1
    assert set(np.unique(new))==set(np.unique(labels))


def test_conflicting_fragment_cannot_merge_two_words(sample):
    labels,p,g,canvas,c=sample
    fragments=np.zeros_like(labels);fragments[4:8,4:14]=1
    new,audit=associate(labels,fragments,np.array([[1,1]]),np.array([.99]),.8)
    np.testing.assert_array_equal(new,labels)
    assert audit['conflicting_fragments_preserved']==1


def test_association_rejects_nonfinite_scores(sample):
    labels,p,g,canvas,c=sample
    with pytest.raises(ValueError,match='scores'):
        associate(labels,np.zeros_like(labels),np.array([[1,1]]),np.array([np.nan]),.8)


def test_unseeded_instance_is_not_fabricated_as_a_kernel(sample):
    labels,p,g,canvas,c=sample;p[1]=0
    out=candidates(p,g,labels,canvas,c)
    assert len(out['pair_ids'])==0 and len(out['boxes'])==2


@pytest.mark.parametrize('bad',[np.nan,-.1,1.1])
def test_invalid_evidence_fails(sample,bad):
    labels,p,g,canvas,c=sample;p[0,0,0]=bad
    with pytest.raises(ValueError):candidates(p,g,labels,canvas,c)


def test_empty_foreground_has_valid_empty_features(sample):
    labels,p,g,canvas,c=sample
    out=candidates(np.zeros_like(p),np.zeros_like(g),np.zeros_like(labels),canvas,c)
    assert out['box_features'].shape==(0,12) and out['pair_features'].shape==(0,12)
    assert adjust_boxes(out['boxes'],np.zeros((0,4)),labels.shape).shape==(0,4)


def test_zero_correction_and_clipping_exclusive_geometry():
    boxes=np.array([[0,0,4,4],[10,10,16,16]],np.float32)
    np.testing.assert_array_equal(adjust_boxes(boxes,np.zeros((2,4)),(16,16)),boxes)
    out=adjust_boxes(boxes,np.array([[-.25,-.25,.25,.25]]*2),(16,16))
    assert out.min()==0 and out.max()==16 and np.all(out[:,2:]>out[:,:2])


def test_crop_loss_zero_for_exact_boxes_and_gradient_for_missing_edge():
    config={'regression':1.,'coverage':1.,'excess':.25,'foreign':.25}
    box=torch.tensor([[2.,2.,10.,6.]])
    zero=torch.zeros((1,4),requires_grad=True)
    assert float(crop_loss(zero,box,box,[[]],.25,config))==0
    target=torch.tensor([[1.,2.,11.,6.]])
    loss=crop_loss(zero,box,target,[[]],.25,config);loss.backward()
    assert zero.grad[0,0]>0 and zero.grad[0,2]<0


def test_neighbor_penalty_does_not_penalize_gt_bbox_overlap():
    config={'regression':0.,'coverage':0.,'excess':0.,'foreign':1.}
    box=torch.tensor([[2.,2.,10.,6.]])
    assert float(crop_loss(torch.zeros((1,4)),box,box,[[[8,2,12,6]]],.25,config))==0
    pred=torch.tensor([[0.,0.,1.,0.]])
    assert float(crop_loss(pred,box,box,[[[8,2,12,6]]],.25,config))>0


def training_config():
    return {'seed':7,'epochs':6,'hidden':4,'learning_rate':.01,'weight_decay':.0001,'max_box_delta':.25,
            'crop_loss':{'regression':1.,'coverage':1.,'excess':.25,'foreign':.25}}


def test_small_head_training_is_deterministic_and_roundtrips():
    x=np.random.default_rng(1).normal(size=(12,12)).astype(np.float32);y=np.array([0,1]*6,np.float32)
    a=fit_head(x,y,training_config(),kind='affinity');b=fit_head(x,y,training_config(),kind='affinity')
    assert a==b and len(a['history'])==6
    np.testing.assert_array_equal(predict_head(a,x),predict_head(json.loads(json.dumps(a)),x))
    assert predict_head(a,np.zeros((0,12),np.float32)).shape==(0,)


def test_head_rejects_missing_positive_supervision():
    with pytest.raises(ValueError,match='positive AND negative'):
        fit_head(np.zeros((5,12),np.float32),np.zeros(5),training_config(),kind='affinity')


def test_geometry_report_never_fabricates_ocr_measurements():
    result=summarize([{'gt_words':20,'proposals':10,'geometry_correct':5}])
    assert result['geometry_recall']==.25 and 'ocr_correct' not in result
    assert 'ocr_hmean' not in result


def test_empty_ocr_keeps_full_denominator():
    result=summarize([{'gt_words':20,'proposals':10,'geometry_correct':5,'ocr_correct':0}])
    assert result['ocr_hmean']==0 and result['gt_words']==20 and result['proposals']==10


def test_correction_preserves_source_ignore_identity(sample):
    labels,p,g,canvas,c=sample;m={'image_wh':np.array([16,16]),'validity':np.zeros_like(canvas)}
    m['validity'][4:8,2:4]=True
    extended=labels.copy();extended[4:8,2:4]=1
    out=proposals(extended,m,ignore_labels=labels)
    assert all(row['ignored'] for row in out)


def test_immutable_contract_cannot_be_rebound(tmp_path):
    path=tmp_path/'contract.json';bind(path,{'a':1});bind(path,{'a':1})
    with pytest.raises(ValueError,match='changed'):bind(path,{'a':2})


def test_strict_and_cross_namespace_configs_fail():
    config=Path(__file__).resolve().parents[2]/'configs/approximate/word_evidence_suite_v1.json'
    with pytest.raises(ValueError):read_config(config,'strict')
    assert read_config(config,'approximate')[0]['claim_label']=='approximate reproduction'


def test_oom_is_recorded_and_propagated_without_fallback(tmp_path,monkeypatch):
    import scripts.run_word_evidence_suite as runner
    c={'cpu_threads':1,'seed':1,'source_receipt_sha256':'same','control_checkpoint_sha256':'same',
       'validation_count':1,'training_count':1,'export_batch_size':1,'ocr_batch_size':1}
    source={'train_manifest':'manifest','word_manifest_sha256':'same','glyph_manifest_sha256':'same','validation_ids_sha256':'split'}
    args=SimpleNamespace(output_root=tmp_path/'run',mode='approximate',config=tmp_path/'config',device='cpu',action='all')
    for name in ('word_root','glyph_root','glyph_run','control_run','parseq_source','parseq_checkpoint','parseq_archive'):
        setattr(args,name,tmp_path/name)
    monkeypatch.setattr(runner,'read_config',lambda *a:(c,source))
    monkeypatch.setattr(runner,'sha256_file',lambda p:'same')
    monkeypatch.setattr(runner,'load_segmentation_manifest',lambda *a,**k:object())
    monkeypatch.setattr(runner,'_split_indices',lambda *a:([0],[1],'split'))
    monkeypatch.setattr(runner,'runtime_environment',lambda:{})
    monkeypatch.setattr(runner,'validation_maps',lambda *a:(_ for _ in ()).throw(torch.cuda.OutOfMemoryError('simulated OOM')))
    with pytest.raises(torch.cuda.OutOfMemoryError,match='simulated OOM'):runner.run(args)
    failure=json.loads((args.output_root/'failure.json').read_text())
    assert failure['status']=='FAILED' and failure['memory']['oom'] is True
    assert 'simulated OOM' in failure['message']


def test_ignored_fragments_have_zero_training_supervision(sample):
    labels,p,g,canvas,c=sample;can=candidates(p,g,labels,canvas,c)
    validity=np.zeros_like(canvas)
    result=supervision(can,labels,labels,validity,c)
    assert not result['pair_keep'].any() and not result['box_keep'].any()
    assert len(result['pair_keep'])==len(can['pair_ids'])
