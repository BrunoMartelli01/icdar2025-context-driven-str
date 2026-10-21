from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.run_text_recall_resolution import DEFAULT_CONFIG, read_config
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.text_recall import (
    TextRecallDataset, make_transform, masked_tversky, recall_loss, train_recall_epoch,
    canonical_probability, coverage_counts, aggregate_coverage, operating_point, selection_key,
)
from segmentation.transforms import SegmentationTransformConfig
from segmentation.word_multitask import masked_bce_dice


@pytest.fixture
def config():
    return read_config(DEFAULT_CONFIG,"approximate")


def test_512_transform_exact_replay_and_original_checks_stay_strict(config):
    cfg,source=config
    old=_transform(source,"letterbox",train=True)
    new=make_transform(source,input_size=512,train=True)
    old.set_epoch(8);new.set_epoch(8)
    image=Image.fromarray(np.arange(33*57*3,dtype=np.uint8).reshape(33,57,3))
    mask=Image.new("L",image.size,0);mask.paste(255,(3,6,24,21))
    valid=Image.new("L",image.size,255);valid.paste(0,(25,0,57,8))
    assert all(torch.equal(a,b) for a,b in zip(old.transform_with_validity(image,mask,valid,5),
        new.transform_with_validity(image,mask,valid,5)))
    values=asdict(old.config);values.update(image_size=(768,768),mask_size=(384,384))
    with pytest.raises(ValueError,match="sizes"):SegmentationTransformConfig(**values)


def test_768_uses_original_targets_paired_geometry_and_ignore(tmp_path,config):
    cfg,source=config
    transform=make_transform(source,input_size=768,train=False)
    owners=np.zeros((31,79),np.int32);owners[4:18,5:27]=1;owners[21:24,52:58]=2
    Image.fromarray(owners).save(tmp_path/"owners.tiff")
    image=Image.new("RGB",(79,31),(121,85,144))
    mask=Image.fromarray((owners>0).astype(np.uint8)*255);valid=Image.new("L",image.size,255)
    valid.paste(0,(0,0,79,3))
    transformed=transform.transform_with_validity_and_separation(image,mask,valid,
        Image.new("L",image.size,0),Image.fromarray(owners),0)
    class Base:
        data_root=tmp_path
        manifest=SimpleNamespace(samples=[SimpleNamespace(word_owner_annotation_path=Path("owners.tiff"))])
        def __len__(self):return 1
        def __getitem__(self,index):
            return dict(zip(("image","mask","validity","separation"),transformed),sample_id="word:1")
    base=Base();base.transform=transform;sample=TextRecallDataset(base)[0]
    assert sample["image"].shape==(3,768,768) and sample["mask"].shape==(1,384,384)
    assert sample["expected_words"]==2 and set(torch.unique(sample["owners_native"]).tolist())=={0,1,2}
    assert torch.equal(sample["mask"][0].bool(),sample["owners_native"]>0)
    assert not sample["owners_native"][sample["validity"][0]==0].any()


def test_tversky_reference_fn_cost_and_ignore_gradient():
    target=torch.tensor([[[[1.,1.,0.,0.]]]])
    valid=torch.tensor([[[[1.,1.,1.,0.]]]])
    logits=torch.zeros_like(target,requires_grad=True)
    loss=masked_tversky(logits,target,valid,fp_weight=.3,fn_weight=.7,smooth=1)
    assert loss.item()==pytest.approx(1-2/(1+.3*.5+.7*1+1))
    loss.backward();assert logits.grad[0,0,0,3]==0 and logits.grad[0,0,0,0]<0
    p_fn=torch.tensor([[[[1.,0.,0.]]]]).clamp(.000001,.999999).logit()
    p_fp=torch.tensor([[[[1.,1.,1.]]]]).clamp(.000001,.999999).logit()
    gt=target[:,:,:,:3];valid=torch.ones_like(gt)
    fn=masked_tversky(p_fn,gt,valid,fp_weight=.3,fn_weight=.7,smooth=1)
    fp=masked_tversky(p_fp,gt,valid,fp_weight=.3,fn_weight=.7,smooth=1)
    # Different TP counts also affect the ratio; compare weights on the SAME FN.
    symmetric=masked_tversky(p_fn,gt,valid,fp_weight=.5,fn_weight=.5,smooth=1)
    assert fn>symmetric and fn>fp


def test_tversky_equals_dice_at_symmetric_weights():
    torch.manual_seed(7)
    logits=torch.randn(2,1,5,8);gt=(torch.rand_like(logits)>.4).float();valid=torch.ones_like(gt)
    t=masked_tversky(logits,gt,valid,fp_weight=.5,fn_weight=.5,smooth=.5)
    d=masked_bce_dice(logits,gt,valid,bce_weight=0,dice_weight=1,smooth=1)
    assert t==pytest.approx(float(d),abs=1e-6)


@pytest.mark.parametrize("fp,fn",[(0,1),(.5,.7),(float('nan'),.7)])
def test_invalid_tversky_coefficients_fail(fp,fn):
    x=torch.zeros(1,1,2,2)
    with pytest.raises(ValueError):masked_tversky(x,x,torch.ones_like(x),fp_weight=fp,fn_weight=fn,smooth=1)


def test_control_loss_exact_and_all_ignore_skips_optimizer(config):
    cfg,_=config;target=torch.zeros(1,1,5,5);target[:,:,1:4,1:4]=1
    batch={"mask":target,"validity":torch.ones_like(target),"separation":torch.zeros_like(target)}
    logits=torch.randn_like(target)
    expected=masked_bce_dice(logits,target,batch["validity"],bce_weight=1,dice_weight=1,smooth=1)
    assert torch.equal(recall_loss(logits,batch,loss_name="bce_dice",config=cfg["loss"]),expected)
    model=torch.nn.Conv2d(3,1,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.5)
    before={k:v.clone() for k,v in model.state_dict().items()}
    batch.update(image=torch.ones(1,3,5,5),sample_id=["ignored:1"],validity=torch.zeros_like(target))
    metrics=train_recall_epoch(model,[batch],optimizer,torch.device('cpu'),loss_name="bce_tversky",loss_config=cfg["loss"])
    assert metrics['empty_batches_without_update']==1 and metrics['steps']==0 and not optimizer.state
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())


def reference():
    owners=np.array([[1,1,0,0],[0,0,2,2]],np.int32)
    return dict(owners=owners,target=owners>0,validity=np.ones_like(owners,bool),
                canvas=np.ones_like(owners,bool),expected_owner_ids=[1,2,3])


def test_coverage_union_fusion_missing_words_and_ignore():
    ref=reference();ref['validity'][0,3]=False
    p=np.array([[1,1,1,1],[0,0,1,1]],np.float32)
    row=coverage_counts(p,ref,.5,complete_coverage=.9)
    assert (row['tp'],row['fp'],row['fn'],row['gt_words'],row['lost_words'])==(4,1,0,3,1)
    assert row['complete_words']==2 and row['coverage_sum']==2
    aggregate=aggregate_coverage([row])
    assert aggregate['mean_word_coverage']==pytest.approx(2/3)
    assert aggregate['pixel_f2']==pytest.approx(20/21)


def test_canonical_resampling_before_threshold_no_gt_modification():
    p=np.array([[.3,.3,.9,.9],[.3,.3,.9,.9],[.9,.9,.3,.3],[.9,.9,.3,.3]],np.float32)
    actual=canonical_probability(p,size=2)
    assert np.allclose(actual,[[.3,.9],[.9,.3]])
    assert np.array_equal(canonical_probability(p,size=4),p)
    assert not np.shares_memory(canonical_probability(p,size=4),p)
    with pytest.raises(ValueError):canonical_probability(p*np.nan,size=2)


def test_precision_floor_prevents_full_canvas_winning_and_ties_are_early():
    base=aggregate_coverage([coverage_counts(np.array([[1,1,0,0],[0,0,0,0]],np.float32),reference(),.5,complete_coverage=.9)])
    all_foreground=aggregate_coverage([coverage_counts(np.ones((2,4),np.float32),reference(),.5,complete_coverage=.9)])
    point=operating_point({'0.35':all_foreground,'0.65':base},minimum_precision=.9)
    assert point['threshold']==.65 and point['precision_requirement_met']
    assert selection_key(point,1)>selection_key(point,2)
    unqualified=operating_point({'0.35':all_foreground},minimum_precision=.9)
    assert not unqualified['precision_requirement_met']
    assert selection_key(point,9)>selection_key(unqualified,1)
