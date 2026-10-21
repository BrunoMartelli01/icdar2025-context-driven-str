from dataclasses import replace

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.guarded_word_gt import WordGuardConfig, guard_native_owners, contact_pairs, GuardedWordTransform
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig
from segmentation.data import SegmentationManifest, SegmentationSample
from pathlib import PurePosixPath
from segmentation.data import file_sha256


CFG=WordGuardConfig(max_growth_native=2,growth_steps=8,min_area_fraction=.2,
    min_span_fraction=.8,crop_padding_native=2,thin_bbox_side_native=2,
    unresolved_policy='retain_all_original_fragments_and_report')


def test_repairs_split_word_without_discarding_original_fragments():
    ref=np.zeros((20,30),np.int32);ref[5:10,3:23]=1
    compact=ref.copy();compact[:,12:14]=0
    before=compact.copy();out,a=guard_native_owners(compact,ref,np.ones_like(ref,bool),[1],CFG)
    assert a['stats']['fragmented_before']==1 and a['stats']['fragmented_after']==0
    assert np.array_equal(compact,before) and np.all(out[compact>0]==compact[compact>0])
    assert np.all(ref[out>0]==out[out>0]) and contact_pairs(out)==0


def test_thin_words_restore_full_source_extent():
    ref=np.zeros((10,30),np.int32);ref[4:6,2:27]=1
    compact=np.zeros_like(ref);compact[4,5:24]=1
    out,a=guard_native_owners(compact,ref,np.ones_like(ref,bool),[1],CFG)
    np.testing.assert_array_equal(out,ref)
    assert a['stats']['changed_words']==1 and a['stats']['unresolved_words']==0


def test_source_restoration_never_merges_neighbours_or_erases_them():
    ref=np.zeros((15,20),np.int32);ref[5:9,2:10]=1;ref[5:9,10:17]=2
    compact=np.zeros_like(ref);compact[6,4:7]=1;compact[5:9,11:17]=2
    out,a=guard_native_owners(compact,ref,np.ones_like(ref,bool),[1,2],CFG)
    assert contact_pairs(out)==0 and np.all(out[compact==2]==2)
    assert np.all(out[compact==1]==1) and a['stats']['contact_pairs_after']==0


def test_unresolvable_split_is_retained_and_reported_not_trimmed_to_largest():
    ref=np.zeros((10,20),np.int32);ref[3:6,2:6]=1;ref[3:6,12:16]=1
    out,a=guard_native_owners(ref,ref,np.ones_like(ref,bool),[1],CFG)
    np.testing.assert_array_equal(out,ref)
    assert a['stats']['fragmented_after']==1 and 'fragmented' in a['per_word'][0]['unresolved']


def test_missing_reference_words_remain_in_denominator():
    ref=np.zeros((8,8),np.int32);ref[2:5,2:5]=1
    _,a=guard_native_owners(ref,ref,np.ones_like(ref,bool),[1,2],CFG)
    assert a['stats']['words']==2 and a['stats']['missing_after']==1
    assert a['per_word'][1]['owner']==2 and a['per_word'][1]['unresolved']


def test_crop_coverage_checks_both_axes():
    ref=np.zeros((25,40),np.int32);ref[3:20,2:35]=1
    compact=np.zeros_like(ref);compact[9:15,4:33]=1
    out,a=guard_native_owners(compact,ref,np.ones_like(ref,bool),[1],replace(CFG,crop_padding_native=0))
    y,x=np.where(out);assert (x.min(),x.max(),y.min(),y.max())==(2,34,3,19)
    assert a['stats']['coverage_failures_after']==0


def test_validity_is_preserved_and_repair_is_deterministic():
    ref=np.zeros((12,20),np.int32);ref[3:9,2:16]=1
    valid=np.ones_like(ref,bool);valid[:4]=False
    compact=np.where(valid,ref,0);compact[:,7:10]=0
    x,a=guard_native_owners(compact,ref,valid,[1],CFG);y,b=guard_native_owners(compact,ref,valid,[1],CFG)
    np.testing.assert_array_equal(x,y);assert a==b and not x[~valid].any()


def test_empty_ignore_only_sample_is_valid():
    z=np.zeros((10,10),np.int32)
    out,a=guard_native_owners(z,z,np.zeros_like(z,bool),[],CFG)
    assert not out.any() and a['stats']['words']==0


def test_refuses_mislabeled_compact_pixels_and_inherited_merges():
    ref=np.zeros((10,10),np.int32);ref[2:5,1:4]=1;ref[2:5,4:7]=2
    with pytest.raises(ValueError,match='cross-owner'):
        guard_native_owners(ref,ref,np.ones_like(ref,bool),[1,2],CFG)
    wrong=np.zeros_like(ref);wrong[2,2]=2
    with pytest.raises(ValueError,match='same-ID'):
        guard_native_owners(wrong,ref,np.ones_like(ref,bool),[1,2],CFG)


@pytest.mark.parametrize('field,value',[('growth_steps',0),('crop_padding_native',-1),('min_span_fraction',0),
    ('unresolved_policy','discard_fragments')])
def test_invalid_guard_configuration_fails(field,value):
    with pytest.raises(ValueError):replace(CFG,**{field:value})


@pytest.mark.parametrize('namespace,mode',[('strict','approximate'),('demo','approximate'),('approximate','strict')])
def test_factory_cannot_enter_strict_or_demo_namespaces(tmp_path,namespace,mode):
    import json
    from segmentation.guarded_word_gt import make_guarded_dataset
    folder=tmp_path/namespace;folder.mkdir();path=folder/'plan.json'
    path.write_text(json.dumps({'reproduction_mode':mode,'variant':'guarded-word-targets-v3'}))
    with pytest.raises(ValueError,match='isolated APPROXIMATE'):
        make_guarded_dataset(path,augmentation_enabled=True)


def test_reference_order_mismatch_fails_before_any_training(tmp_path):
    from types import SimpleNamespace
    a=SimpleNamespace(samples=[SimpleNamespace(sample_id='a')])
    b=SimpleNamespace(samples=[SimpleNamespace(sample_id='b')])
    with pytest.raises(ValueError,match='ordering'):
        GuardedWordTransform(None,a,b,tmp_path,CFG)


def test_audited_dataset_returns_collatable_evidence_without_changing_targets(tmp_path):
    import json
    from types import SimpleNamespace
    from torch.utils.data._utils.collate import default_collate
    from segmentation.guarded_word_dataset import AuditedGuardedWordDataset
    target=torch.ones((1,2,2));owner=np.ones((2,2),np.int32)
    audit={'stats':{'words':1,'unresolved_words':0},'per_word':[{'owner':1}]}
    class Source:
        transform=SimpleNamespace(last_audit=audit,last_owners=owner)
        manifest=None
        paths=[]
        def __len__(self):return 2
        def __getitem__(self,index):return {'mask':target,'sample_id':str(index)}
    recipe=tmp_path/'recipe.json';recipe.write_text('{}')
    dataset=AuditedGuardedWordDataset(Source(),recipe)
    a,b=dataset[0],dataset[1]
    assert a['mask'] is target and a['target_variant']=='guarded-word-targets-v3'
    assert json.loads(a['gt_guard_audit_json'])==audit
    batch=default_collate([a,b])
    assert batch['native_owners'].shape==(2,2,2) and batch['gt_guard_stats']['words'].tolist()==[1,1]


@pytest.mark.parametrize('angle',[0,13])
def test_adapter_preserves_rgb_validity_and_recomputes_separator_after_augmentation(tmp_path,angle):
    from types import SimpleNamespace
    ref=np.zeros((160,240),np.int32);ref[40:70,30:180]=1
    compact=np.zeros_like(ref);compact[45:65,40:170]=1;compact[:,100:110]=0
    path=tmp_path/'ref.png';Image.fromarray(ref).save(path)
    record=SimpleNamespace(sample_id='test:1',image_sha256='same',word_owner_annotation_path=PurePosixPath('ref.png'),
        word_owner_annotation_sha256=file_sha256(path))
    manifest=SimpleNamespace(samples=[record])
    c=SegmentationTransformConfig(horizontal_flip_probability=1.,rotation_probability=float(angle!=0),
        rotation_degrees=(angle,angle),image_interpolation='bilinear',mask_interpolation='bilinear',
        image_fill=(0,0,0),mask_fill=0,normalization='none',resize_geometry='stretch',operation_order='flip_rotate_resize')
    base=PairedSegmentationTransform(c,base_seed=13)
    adapter=GuardedWordTransform(base,manifest,manifest,tmp_path,CFG);adapter.set_epoch(2)
    image=Image.fromarray(np.repeat((ref>0)[...,None]*np.uint8(255),3,axis=2))
    mask=Image.fromarray((compact>0).astype(np.uint8)*255);valid=Image.new('L',image.size,255)
    sep=Image.fromarray(((ref>0)&(compact==0)).astype(np.uint8)*255);owners=Image.fromarray(compact)
    old=base.transform_with_validity_and_separation(image,mask,valid,sep,owners,0)
    new=adapter.transform_with_validity_and_separation(image,mask,valid,sep,owners,0)
    assert torch.equal(old[0],new[0]) and torch.equal(old[2],new[2])
    assert torch.all(new[1]>=old[1]) and not torch.any(new[1].bool()&new[3].bool())
    assert adapter.last_audit['stats']['words']==1
