import numpy as np
from PIL import Image
import pytest

from segmentation.tighter_word_gt import compact_word_owners, native_training_owners
from segmentation.separated_word_gt import audit_word_owners
from segmentation.config import load_segmentation_config
from segmentation.config import SegmentationConfigError
from segmentation.train import _transform
import yaml
from scripts.prepare_separated_word_training import VARIANTS


PARAMS = dict(distance_fraction=0.55, max_inset_reference=2,
              reference_wh=(128,128), crop_padding_reference=6,
              min_area_fraction=0.2, min_span_fraction=0.8, backoff_steps=8)


def sample():
    own=np.zeros((128,128),np.uint16)
    own[20:40,10:90]=1
    own[44:60,15:85]=2
    return own,np.full_like(own,255,dtype=np.uint8),np.zeros_like(own,dtype=np.uint8)


def test_cores_are_smaller_connected_separate_subsets_and_deterministic():
    own,valid,sep=sample()
    after,gap,audit=compact_word_owners(own,valid,sep,**PARAMS)
    assert np.array_equal(own,sample()[0])
    assert audit['changed_words']==2 and audit['removed_pixels']>0
    assert audit['missing_word_ids']==audit['fragmented_word_ids']==[]
    assert not np.any((after>0)&(after!=own))
    assert not np.any((after>0)&(gap>0))
    repeated=compact_word_owners(own,valid,sep,**PARAMS)
    assert np.array_equal(after,repeated[0]) and audit==repeated[2]


def test_crop_padding_guard_backs_off_instead_of_cutting_word_bounds():
    own,valid,sep=sample()
    after,_,_=compact_word_owners(own,valid,sep,**{**PARAMS,'crop_padding_reference':0})
    for index in (1,2):
        a=np.where(after==index); b=np.where(own==index)
        assert tuple((x.min(),x.max()) for x in a)==tuple((x.min(),x.max()) for x in b)


def test_effective_native_word_survival_is_preserved_for_small_words():
    own=np.zeros((512,512),np.uint16)
    own[40:50,40:100]=1
    own[200:240,40:220]=2
    valid=np.full(own.shape,255,np.uint8); sep=np.zeros(own.shape,np.uint8)
    before=native_training_owners(own,valid,sep,(128,128))
    after,gap,audit=compact_word_owners(own,valid,sep,**PARAMS)
    native=native_training_owners(after,valid,gap,(128,128))
    assert set(np.unique(before)) <= set(np.unique(native))
    assert audit['new_words_lost_at_training_resolution']==[]


def test_native_helper_agrees_with_actual_training_target_transform():
    own=np.zeros((233,511),np.uint16)
    own[20:47,35:219]=1; own[55:80,35:120]=2
    valid=np.full(own.shape,255,np.uint8); sep=np.zeros(own.shape,np.uint8)
    valid[:10]=0
    config=load_segmentation_config('configs/approximate/segmentation_total_text_separated_words_v1.yaml')
    transform=_transform(config.transforms,seed=202503,augmentation_enabled=False)
    mask=Image.fromarray((own>0).astype(np.uint8)*255)
    _,target,_,_=transform.transform_with_validity_and_separation(Image.new('RGB',(511,233)),
        mask,Image.fromarray(valid),Image.fromarray(sep),Image.fromarray(own.astype(np.int32),mode='I'),0)
    helper=native_training_owners(own,valid,sep,(128,128))
    assert np.array_equal(helper>0,np.asarray(target[0])>0)


def test_empty_ignore_only_target_is_retained():
    own=np.zeros((50,100),np.uint16)
    out,_,audit=compact_word_owners(own,np.zeros(own.shape,np.uint8),np.zeros(own.shape,np.uint8),**PARAMS)
    assert not out.any() and audit['words']==0


@pytest.mark.parametrize('key,value',[('distance_fraction',0),('max_inset_reference',0),
                                     ('min_area_fraction',0),('backoff_steps',-1)])
def test_invalid_scientific_parameters_fail(key,value):
    with pytest.raises(ValueError):
        compact_word_owners(*sample(),**{**PARAMS,key:value})


@pytest.mark.parametrize('mutation',[None,'wrong_target','short_run','v1_protocol'])
def test_compact_training_namespace_is_isolated(tmp_path,mutation):
    source=load_segmentation_config('configs/approximate/segmentation_total_text_separated_words_v1.yaml')
    config=yaml.safe_load(open('configs/approximate/segmentation_total_text_separated_words_v1.yaml'))
    identity=VARIANTS['compact_v2']
    config['run']['name']=identity['run_name'];config['training']['protocol']=identity['protocol']
    config['data']['train'].update({'artifact_id':identity['artifact'],'target_source':identity['target']})
    if mutation=='wrong_target':config['data']['train']['target_source']=VARIANTS['separated_v1']['target']
    elif mutation=='short_run':config['training']['epochs']=100
    elif mutation=='v1_protocol':config['training']['protocol']=VARIANTS['separated_v1']['protocol']
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
    if mutation is not None:
        with pytest.raises(SegmentationConfigError):load_segmentation_config(path)
    else:
        loaded=load_segmentation_config(path)
        assert loaded.architecture==source.architecture and loaded.transforms==source.transforms
        assert loaded.training.epochs==source.training.epochs==500
