import json

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.data import file_sha256
from segmentation.icdar2015_improvements import make_transform
from segmentation.icdar2015_synthtext_cache import CachedSynthTextBCE,prepare_sample
from segmentation.synthtext_words import SynthTextWordDataset


def fixture(tmp_path):
    rgb=tmp_path/'image.png';Image.fromarray(np.random.default_rng(17).integers(0,256,(70,110,3),dtype=np.uint8)).save(rgb)
    row=dict(sample_id='synthtext:7',image_relative='image.png',image_sha256=file_sha256(rgb),words=[
        dict(polygon_xy=[[8,12],[45,12],[45,34],[8,34]],transcription='Word'),
        dict(polygon_xy=[[60,40],[100,40],[100,62],[60,62]],transcription='two')])
    return row,rgb


@pytest.mark.parametrize('epoch',[0,7])
def test_cache_preserves_actual_paired_tensors_and_audits(tmp_path,epoch):
    row,rgb=fixture(tmp_path);cache=tmp_path/'cache';saved=prepare_sample(row,tmp_path,cache,'preserve_visible_words_v1',set())
    assert prepare_sample(row,tmp_path,cache,'preserve_visible_words_v1',set())==saved
    config=dict(canvas_fill_rgb=[124,116,104],probability_threshold=.5,training=dict(seed=202503,horizontal_flip_probability=.5,rotation_probability=.5,rotation_degrees=[-15.,15.]))
    tx=make_transform(config,[512,512],augment=True);tx.set_epoch(epoch)
    new=CachedSynthTextBCE(dict(samples=[saved],target_policy='preserve_visible_words_v1'),image_root=tmp_path,cache_root=cache,transform=tx)[0]
    oldrow=dict(sample_id=row['sample_id'],image_path=str(rgb),image_sha256=row['image_sha256'],words=row['words'],target_audit=saved['target_audit'])
    old=SynthTextWordDataset(dict(samples=[oldrow],target_policy='preserve_visible_words_v1'),tx,[.5,.3,.1],target_policy='preserve_visible_words_v1')[0]
    for key in ['image','mask','validity','separation']:assert torch.equal(new[key],old[key])
    assert new['native_words_lost']==old['native_words_lost'] and new['visible_words']==2


def test_overlap_and_cache_corruption_fail(tmp_path):
    row,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='overlap'):prepare_sample(row,tmp_path,tmp_path/'cache','preserve_visible_words_v1',{row['image_sha256']})
    saved=prepare_sample(row,tmp_path,tmp_path/'cache','preserve_visible_words_v1',set())
    (tmp_path/'cache'/saved['target_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='cache changed'):prepare_sample(row,tmp_path,tmp_path/'cache','preserve_visible_words_v1',set())


def test_raw_words_are_not_mutated_and_source_changes_fail(tmp_path):
    row,rgb=fixture(tmp_path);before=json.dumps(row,sort_keys=True);prepare_sample(row,tmp_path,tmp_path/'cache','preserve_visible_words_v1',set());assert json.dumps(row,sort_keys=True)==before
    rgb.write_bytes(b'changed')
    with pytest.raises(ValueError,match='RGB bytes changed'):prepare_sample(row,tmp_path,tmp_path/'cache','preserve_visible_words_v1',set())
