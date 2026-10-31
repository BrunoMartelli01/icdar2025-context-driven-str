from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from segmentation.data import file_sha256
from segmentation.icdar2015_synthetic_training import stage_config,training_config,optimizer,CountedBatches,load_source
from segmentation.icdar2015_target70 import set_learning_rates


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/approximate/icdar2015_synthtext_pretrain_v1.json').read_text())


@pytest.mark.parametrize('phase,epochs,batch,selection',[('pretrain',5,8,'last'),('finetune',50,4,'validation_word_f1')])
def test_stage_metadata_and_cosine_rates_are_actual(phase,epochs,batch,selection):
    original=config();c=stage_config(original,phase);b=c['branches'][phase];t=training_config(c,b)
    assert asdict(t)['optimizer']=='AdamW' and t.epochs==epochs and t.batch_size==batch and t.checkpoint_selection==selection
    assert original==config()
    model=torch.nn.Module();model.decoder=torch.nn.Linear(2,1);model.encoder=torch.nn.Linear(2,2)
    opt=optimizer(model,c,b);assert isinstance(opt,torch.optim.AdamW) and not opt.state
    assert [g['weight_decay'] for g in opt.param_groups]==[.01,0.]
    start=set_learning_rates(opt,c,1);end=set_learning_rates(opt,c,epochs)
    assert start['decoder']==b['decoder_learning_rate'] and end['decoder']==pytest.approx(b['decoder_learning_rate']*.1)
    assert end['encoder_layer3_layer4']==pytest.approx(b['encoder_learning_rate']*.1)


def test_counted_batches_does_not_count_unused_smoke_tail():
    loader=[dict(visible_words=torch.tensor([3,4]),native_words_lost=torch.tensor([0,1]),outside_image_words=torch.tensor([0,0])) for _ in range(4)]
    batches=CountedBatches(loader,2);assert len(list(batches))==2
    assert batches.counts==dict(visible_words=14,native_words_lost=2,outside_image_words=0)


@pytest.mark.parametrize('smoke',[False,True])
def test_incomplete_pretraining_only_enters_explicit_smoke(tmp_path,monkeypatch,smoke):
    import segmentation.icdar2015_synthetic_training as module
    c=stage_config(config(),'finetune')
    checkpoint=tmp_path/'source.pt';checkpoint.write_bytes(b'fixture');words=tmp_path/'words.json';words.write_text('{}');profile=tmp_path/'config.json';profile.write_text('{}')
    report=dict(status='SUCCESS',training_complete=False,result_class='smoke-test result',completed_epochs=1,
        checkpoint=dict(sha256=file_sha256(checkpoint),epoch=1),selected_validation_words_sha256=file_sha256(words),
        run_contract=dict(branch='pretrain',source_sha256={},config_sha256=file_sha256(profile),synthetic_manifest_sha256='fixture-manifest'))
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    args=SimpleNamespace(branch='finetune',source_checkpoint=checkpoint,source_report=path,source_validation=words,config=profile,smoke_steps=2 if smoke else None)
    monkeypatch.setattr(module,'load_checkpoint',lambda *a,**kw:kw['expected_training_config'])
    model=SimpleNamespace(architecture_identity='test')
    if smoke:assert load_source(model,args,c,'fixture-manifest').checkpoint_selection=='last'
    else:
        with pytest.raises(ValueError,match='Incomplete pretraining'):load_source(model,args,c,'fixture-manifest')
