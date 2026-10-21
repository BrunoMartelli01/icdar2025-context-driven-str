from types import SimpleNamespace as NS
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.train_adaspot_gap_fulltrain import ignored_instances,validate_epoch,validate_population


def manifests():
    train=NS(dataset="total_text",split_role="train",samples=[NS(sample_id="a",image_sha256="1"),NS(sample_id="b",image_sha256="2")])
    test=NS(dataset="total_text",split_role="test",samples=[NS(sample_id="c",image_sha256="3")])
    return train,test,{"train_images":2,"test_images":1,"known_official_rgb_overlap":[]}


def test_all_official_images_and_roles_required():
    train,test,c=manifests()
    assert validate_population(train,test,c)==(["a","b"],["c"])
    c["train_images"]=3
    with pytest.raises(ValueError,match="Full official population"):validate_population(train,test,c)


def test_unknown_overlap_fails_and_only_explicit_exact_pair_can_pass():
    train,test,c=manifests();test.samples[0].image_sha256="1"
    with pytest.raises(ValueError,match="Unknown TRAIN/TEST"):validate_population(train,test,c)
    c["known_official_rgb_overlap"]=[{"test_id":"c","train_ids":["a"],"sha256":"1"}]
    assert validate_population(train,test,c)==(["a","b"],["c"])
    c["known_official_rgb_overlap"][0]["train_ids"]=["b"]
    with pytest.raises(ValueError,match="Unknown TRAIN/TEST"):validate_population(train,test,c)


@pytest.mark.parametrize("seen",[["a"],["a","a"],["a","b","c"],["a","c"]])
def test_no_sample_may_be_skipped_or_repeated(seen):
    with pytest.raises(ValueError,match="omitted or duplicated"):validate_epoch(seen,["a","b"])
    validate_epoch(["b","a"],["a","b"])


def test_ignore_only_wholly_invalid_component_keep_partial_care():
    labels=np.array([[1,1,0],[0,2,2]],np.int32)
    valid=np.array([[False,False,False],[False,False,True]])
    assert ignored_instances(labels,valid)==[1]


def test_fulltrain_numerics_preserve_original_and_fixed_prior_choice():
    root=Path(__file__).resolve().parents[2]
    a=json.loads((root/"configs/approximate/adaspot_gap_decoder_v1.json").read_text())
    b=json.loads((root/"configs/approximate/adaspot_gap_fulltrain_v1.json").read_text())
    for k in ("seed","epochs","loss","learning_rate","weight_decay","image_batch_size","initial_bias"):
        assert a[k]==b[k]
    assert b["train_images"]==1255 and b["test_images"]==300
    assert b["evaluation_epochs"]==[b["selected_epoch"],b["epochs"]]==[1,10]
    assert len(b["known_official_rgb_overlap"])==2
