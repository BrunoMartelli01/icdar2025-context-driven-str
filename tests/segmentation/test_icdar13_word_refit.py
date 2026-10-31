"""Word-mask identity, strict source compatibility and partial-encoder gradients."""
import numpy as np
import pytest
import torch
from models.adaspot_gap import AdaSpotGapAGUNet
from segmentation.icdar13_word_refit import WordRefitUNet,strict_fallback_count,padded_rows,word_loss
from segmentation.separated_word_gt import parse_icdar13_character_groups,separated_word_regions


def test_character_groups_become_one_filled_component_per_word(tmp_path):
    p=tmp_path/'labels.txt';p.write_text('1 2 3 0 0 2 3 6 12 "A"\n4 5 6 0 0 11 3 15 12 "B"\n\n7 8 9 0 0 24 3 28 12 "C"\n')
    words=parse_icdar13_character_groups(p);mask,valid,separator,owners,audit=separated_word_regions((40,20),words,ignore_policy='care_region_precedence')
    assert [w.transcription for w in words]==['AB','C']
    assert np.asarray(mask)[6,8]==255 and np.asarray(owners)[6,8]==1
    assert audit['components_per_word']=={'1':1,'2':1} and not audit['interword_contact_pairs']


def test_strict_budget_stays_below_thirty_percent_even_on_small_validation():
    for n,expected in [(233,69),(23,6),(10,2),(500,149)]:
        assert strict_fallback_count(n,.3)==expected and expected/n<.3
    assert padded_rows([dict(bbox=[1,1,5,6])],[8,8],2)[0]['bbox']==[0,0,7,8]
    with pytest.raises(ValueError):strict_fallback_count(0,.3)


def test_partial_encoder_has_real_gradients_and_strict_source_state():
    torch.set_num_threads(2);torch.manual_seed(4)
    model=WordRefitUNet(partial_encoder=True,skip_floor=.25,initial_gap_bias=-2.)
    source=AdaSpotGapAGUNet('control',skip_floor=.25,initial_gap_bias=-2.)
    model.load_state_dict(source.state_dict(),strict=True);model.eval();source.eval();x=torch.rand(2,3,64,64)
    with torch.inference_mode():assert torch.equal(model(x),source(x))
    del source
    before=model.encoder.layer3[0].bn1.running_mean.clone();model.train()
    batch=dict(mask=torch.ones(2,1,32,32),validity=torch.ones(2,1,32,32),separation=torch.zeros(2,1,32,32),kernel=torch.ones(2,1,32,32))
    batch['kernel'][:,:,:8]=0
    word_loss(model(x),batch,dict(separation_weight=3.,kernel_loss_weight=1.)).backward()
    assert model.encoder.layer4[0].conv1.weight.grad is not None
    assert model.encoder.conv1.weight.grad is None and not model.encoder.layer3[0].bn1.training
    assert torch.equal(before,model.encoder.layer3[0].bn1.running_mean)
    assert model.head.weight.grad is not None
