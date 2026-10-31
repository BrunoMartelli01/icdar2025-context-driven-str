"""Target70 preserves metrics and qualifies new supervision/update semantics."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.icdar2015_improvements import make_transform
from segmentation.icdar2015_target70 import (TrainingSupervision,calibrate_emission,
    emitted_predictions,load_profile,set_learning_rates,supervised_loss,train_epoch,
    training_config,word_selection_key)
from segmentation.training import separation_weighted_bce_with_logits


PROFILE=Path(__file__).resolve().parents[2]/"configs/approximate/icdar2015_target70_v1.json"


def test_profile_isolation_and_explicit_physical_batch(tmp_path):
    config=load_profile(PROFILE)
    training=training_config(config,config["branches"]["adapt1280"])
    assert training.batch_size==4 and training.learning_rate==1e-5
    path=tmp_path/"strict"/"profile.json";path.parent.mkdir()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="approximate"):load_profile(path)


def test_ignore_negative_gradients_exclude_padding_and_keep_baseline_loss():
    logits=torch.zeros(1,1,1,3,requires_grad=True)
    batch={"mask":torch.tensor([[[[1.,0.,0.]]]]),"validity":torch.tensor([[[[1.,1.,0.]]]]),
        "ignored_negative":torch.tensor([[[[0.,1.,0.]]]]),"separation":torch.zeros_like(logits)}
    supervised_loss(logits,batch,separation_weight=3,ignore_negative_weight=3).backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[[[-.125,.375,0.]]]]),rtol=0,atol=0)
    batch["ignored_negative"].zero_();batch["separation"][0,0,0,1]=1
    actual=supervised_loss(logits,batch,separation_weight=3,ignore_negative_weight=0)
    expected=separation_weighted_bce_with_logits(logits,batch["mask"],batch["validity"],batch["separation"],separation_weight=3)
    assert torch.equal(actual,expected)
    batch["ignored_negative"][0,0,0,0]=1
    with pytest.raises(ValueError,match="Invalid positive"):supervised_loss(logits,batch,separation_weight=3,ignore_negative_weight=3)


def test_ignore_supervision_replays_geometry_without_training_on_padding(tmp_path):
    config=load_profile(PROFILE)
    # Deterministic paired rotation/flip exercises the replay, not just resize.
    config["training"].update(rotation_probability=1.,horizontal_flip_probability=1.)
    transform=make_transform(config,[64,96],augment=True);transform.set_epoch(57)
    image=Image.new("RGB",(96,48));image.save(tmp_path/"rgb.png")
    valid=np.full((48,96),255,dtype=np.uint8);valid[10:24,30:50]=0
    _,mask,validity=transform.transform_with_validity(image,Image.new("L",image.size),Image.fromarray(valid),0)
    base=SimpleNamespace(data_root=tmp_path,manifest=SimpleNamespace(samples=[SimpleNamespace(image_path=Path("rgb.png"))]),transform=transform)
    class Owner:
        def __init__(self):self.base=base
        def __len__(self):return 1
        def __getitem__(self,index):
            return {"mask":mask.clone(),"validity":validity.clone(),"separation":torch.zeros_like(mask)}
    item=TrainingSupervision(Owner(),negative_ignore=True)[0]
    assert item["ignored_negative"].sum()>0
    assert (item["validity"]==0).any()
    assert torch.equal(item["mask"],mask)
    assert torch.equal(item["ignored_negative"],(item["validity"]-validity).clamp_min(0))


def test_accumulation_includes_last_partial_window():
    torch.manual_seed(17)
    net=torch.nn.Conv2d(1,1,1,bias=False);reference=copy.deepcopy(net)
    batches=[]
    for i in range(3):
        image=torch.tensor([[[[float(i+1)]]]])
        batches.append({"image":image,"sample_id":[str(i)],"mask":torch.ones_like(image),
            "validity":torch.ones_like(image),"separation":torch.zeros_like(image),"ignored_negative":torch.zeros_like(image)})
    config={"training":{"separation_weight":3}}
    branch={"gradient_accumulation":2,"gradient_clip_norm":0,"ignore_negative_weight":0}
    opt=torch.optim.SGD(net.parameters(),lr=.1);refopt=torch.optim.SGD(reference.parameters(),lr=.1)
    actual=train_epoch(net,batches,opt,config,branch,device="cpu")
    for window in (batches[:2],batches[2:]):
        refopt.zero_grad()
        loss=sum(supervised_loss(reference(b["image"]),b,separation_weight=3,ignore_negative_weight=0) for b in window)/len(window)
        loss.backward();refopt.step()
    torch.testing.assert_close(net.weight,reference.weight,rtol=0,atol=0)
    assert (actual["samples"],actual["steps"],actual["microbatches"])==(3,2,3)


def test_cosine_schedule_has_declared_endpoints():
    config=load_profile(PROFILE)
    opt=torch.optim.SGD([{"params":[torch.nn.Parameter(torch.ones(()))],"group_name":"decoder","initial_lr":1e-5}],lr=1e-5)
    rates=[set_learning_rates(opt,config,epoch)["decoder"] for epoch in range(1,101)]
    assert rates[0]==1e-5 and rates[-1]==pytest.approx(1e-6)
    assert all(a>=b for a,b in zip(rates,rates[1:]))


def test_output_filter_preserves_all_ground_truth_and_raw_predictions():
    pred=[{"bbox":[0,0,10,10],"text":"A!","sequence_confidence":.99,"mean_token_log_probability":-.01},
        {"bbox":[20,0,30,10],"text":"WOrd!","sequence_confidence":.99,"mean_token_log_probability":-.01},
        {"bbox":[40,0,50,10],"text":"noise","sequence_confidence":.1,"mean_token_log_probability":-3}]
    gt=[{"bbox":[0,0,10,10],"text":"a"},{"bbox":[20,0,30,10],"text":"word"}]
    rows=[{"predictions":pred,"ground_truth":gt}]
    config={"matching_iou_threshold":.5,"emission_policy_grid":{"sequence_confidence":[0,.8],
        "mean_token_confidence":[0,.8],"minimum_alphanumeric_length":[0,3]}}
    result=calibrate_emission(rows,config)
    assert result["aggregate"]["alphanumeric_casefold"]["f1"]==1
    assert result["policy"]["minimum_alphanumeric_length"]==0
    assert len(pred)==3 and len(gt)==2
    short_reject={"confidence":"sequence_confidence","threshold":.8,"minimum_alphanumeric_length":3}
    filtered=emitted_predictions(pred,short_reject)
    assert [p["text"] for p in filtered]==["WOrd!"]
    candidate=next(c for c in result["candidates"] if c["policy"]==short_reject)
    assert candidate["aggregate"]["alphanumeric_casefold"]["fn"]==1
    assert candidate["aggregate"]["gt_words"]==2
    geometry={"proposal_f1_iou_50":.9}
    raw={"alphanumeric_casefold":{"f1":.5}};emitted={"alphanumeric_casefold":{"f1":.6}}
    assert word_selection_key(raw,emitted,geometry,1)>word_selection_key(raw,emitted,geometry,2)


def test_empty_ocr_decode_remains_a_prediction_at_zero_threshold():
    empty={"text":"","sequence_confidence":.9,"mean_token_log_probability":None}
    policy={"confidence":"mean_token_confidence","threshold":0.,"minimum_alphanumeric_length":0}
    assert emitted_predictions([empty],policy)==[empty]
    assert emitted_predictions([empty],{**policy,"threshold":.1})==[]
