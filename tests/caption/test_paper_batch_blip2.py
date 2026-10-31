from types import SimpleNamespace

import numpy as np
import pytest

from caption.blip2_captioner import BLIP2ApproxConfig
from caption.paper_batch_blip2 import generate_batch


def fake_captioner(monkeypatch,texts):
    import torch
    calls={}
    for name in ('init','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:20)
    class Inputs(dict):
        def to(self,**kwargs):calls['placement']=kwargs;return self
    class Processor:
        def __call__(self,**kwargs):calls['processor']=kwargs;return Inputs(pixel_values='native images')
        def batch_decode(self,outputs,**kwargs):return texts
    class Model:
        def generate(self,**kwargs):calls['generate']=kwargs;return 'native sequences'
    captioner=SimpleNamespace(config=BLIP2ApproxConfig(device='cuda:0',dtype='float16'),
        device='cuda:0',_processor=Processor(),_model=Model(),_load=lambda:None,
        _device_placement_summary=lambda:dict(target='cuda:0'))
    return captioner,calls


def profile():return dict(assembly='single',prompts=['Describe the image.'],generation=dict(min_length=40,max_length=80,num_beams=3,do_sample=False))


def test_native_batch_image_only_prompt_generation_and_shared_memory(monkeypatch):
    captioner,calls=fake_captioner(monkeypatch,[' first caption ','second caption'])
    original=captioner.config
    rgb=np.zeros((12,18,3),np.uint8)
    result=generate_batch(captioner,[rgb,rgb],profile(),dict(min_words=40,max_words=80))
    assert calls['processor']['text']==['Describe the image.']*2
    assert len(calls['processor']['images'])==2
    assert calls['generate']['min_length']==40 and calls['generate']['max_length']==80
    assert [r['raw_caption'] for r in result]==['first caption','second caption']
    assert all(r['native_batch_size']==2 and r['image_only_captioning'] for r in result)
    assert all(r['answers'][0]['result']['peak_allocated_bytes']==10 for r in result)
    assert all('shared batch' in r['answers'][0]['result']['latency_scope'] for r in result)
    assert captioner.config is original


@pytest.mark.parametrize('outputs',[['only one'],['first','']])
def test_empty_or_missing_batch_outputs_fail_and_restore_config(monkeypatch,outputs):
    captioner,_=fake_captioner(monkeypatch,outputs);original=captioner.config
    with pytest.raises(RuntimeError,match='Empty/incorrect-cardinality'):
        generate_batch(captioner,[np.zeros((2,3,3),np.uint8)]*2,profile(),dict(min_words=40,max_words=80))
    assert captioner.config is original


def test_non_rgb_input_fails_before_neural_execution(monkeypatch):
    captioner,_=fake_captioner(monkeypatch,['unused'])
    with pytest.raises(ValueError,match='RGB batch'):
        generate_batch(captioner,[np.zeros((4,5),np.uint8)],profile(),dict(min_words=40,max_words=80))


def test_oom_propagates_without_dummy_caption_and_restores_config(monkeypatch):
    import torch
    captioner,_=fake_captioner(monkeypatch,['must not be returned']);original=captioner.config
    def oom(**kwargs):raise torch.cuda.OutOfMemoryError('qualified OOM propagation sentinel')
    captioner._model.generate=oom
    with pytest.raises(torch.cuda.OutOfMemoryError,match='OOM propagation'):
        generate_batch(captioner,[np.zeros((2,3,3),np.uint8)],profile(),dict(min_words=40,max_words=80))
    assert captioner.config is original
