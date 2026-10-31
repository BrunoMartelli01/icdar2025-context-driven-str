import numpy as np
import pytest
import torch

from recognition.encoder_family_router import EncoderFamilyRouter, collate_observations, count_targets


SETTINGS = dict(hidden=16, heads=2, blocks=1, mlp=24, dropout=.1, byte_embedding=8,
                byte_channels=8, byte_kernel=3, geometry_dim=12, scalar_dim=30, clip_dim=512,
                query_dim=768, query_count=32, global_patches=196, initialization_std=.02)


def row(texts):
    n = len(texts)
    return dict(texts=texts, geometry=np.zeros((n,12),np.float32), scalars=np.zeros(30,np.float32),
        word_text=np.ones((n,512),np.float32), word_image=np.ones((n,512),np.float32),
        patches=np.ones((196,512),np.float32), patch_geometry=np.zeros((196,4),np.float32),
        patch_valid=np.zeros(196,bool), queries=np.ones((32,768),np.float32))


@pytest.mark.parametrize('arm',['none','clip','blip','both'])
def test_shapes_empty_words_and_masked_context(arm):
    torch.manual_seed(123); model=EncoderFamilyRouter(arm,SETTINGS,123)
    batch=collate_observations([row([]),row(['a','città'])],arm,'cpu')
    out=model(batch)
    assert out.shape==(2,2) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_bytes_preserve_long_utf8_and_inactive_features_absent():
    r=row(['à'*130]); del r['queries'],r['patches'],r['word_text'],r['word_image']
    batch=collate_observations([r],'none','cpu')
    assert batch['bytes'].shape==(1,1,260)
    assert batch['bytes'][0,0].tolist()==list(('à'*130).encode('utf8'))
    assert 'queries' not in batch and 'patches' not in batch
    assert torch.isfinite(EncoderFamilyRouter('none',SETTINGS,123)(batch)).all()


def test_shared_initialization_and_optional_dependencies():
    models={}
    for arm in ['none','clip','blip','both']:
        torch.manual_seed(123); models[arm]=EncoderFamilyRouter(arm,SETTINGS,123).eval()
    for key,value in models['none'].state_dict().items():
        assert all(torch.equal(value,m.state_dict()[key]) for m in models.values())
    for key,value in models['both'].state_dict().items():
        if key.startswith('clip_'):assert torch.equal(value,models['clip'].state_dict()[key])
        if key.startswith('blip_') or key=='query_position':assert torch.equal(value,models['blip'].state_dict()[key])
    r=row(['word']); b=collate_observations([r],'blip','cpu'); original=models['blip'](b)
    b['patches']=torch.full((1,196,512),float('nan'))
    assert torch.equal(original,models['blip'](b))
    b['queries']=torch.randn_like(b['queries'])
    assert not torch.equal(original,models['blip'](b))


def test_count_target_direction():
    branches={'x':dict(light=dict(tp=2,detected_words=5),deepsolo=dict(tp=3,detected_words=4))}
    np.testing.assert_array_equal(count_targets(branches,['x']),[[-1,1]])
