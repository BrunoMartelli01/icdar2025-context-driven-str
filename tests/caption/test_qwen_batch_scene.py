from types import SimpleNamespace

import numpy as np
import pytest

from caption.qwen_batch_scene import LocalQwenBatchScene


def setup_model(monkeypatch,texts):
    import torch
    for name in ('init','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:20)
    captured={}
    class Inputs(dict):
        def to(self,*args):return self
    class Processor:
        tokenizer=SimpleNamespace(padding_side='right')
        def apply_chat_template(self,*args,**kwargs):return 'native chat'
        def __call__(self,**kwargs):return Inputs(input_ids=torch.tensor([[0,0,1,2],[0,3,4,5]]),image_grid_thw=torch.tensor([[1,28,28],[1,14,56]]))
        def batch_decode(self,values,**kwargs):captured['suffixes']=values.tolist();return texts
    class Model:
        hf_device_map={'visual':0,'lm_head':'cpu'}
        def generate(self,**kwargs):return torch.tensor([[0,0,1,2,11,12],[0,3,4,5,21,22]])
    model=LocalQwenBatchScene('snapshot','cuda:0',256,1024,'cpu_offload','7GiB','40GiB')
    model.load=lambda:None;model.processor=Processor();model.model=Model()
    return model,captured


def test_left_padded_native_suffixes_and_per_image_grids_preserve_order(monkeypatch):
    model,captured=setup_model(monkeypatch,['first caption','second caption'])
    output=model.generate([np.zeros((2,3,3),np.uint8)]*2,dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))
    assert model.processor.tokenizer.padding_side=='left'
    assert captured['suffixes']==[[11,12],[21,22]]
    assert [r['raw_caption'] for r in output]==['first caption','second caption']
    assert output[0]['answers'][0]['result']['image_grid_thw']==[[1,28,28]]
    assert output[1]['answers'][0]['result']['image_grid_thw']==[[1,14,56]]
    assert all(r['native_batch_size']==2 for r in output)
    assert all('shared native caption batch' in r['answers'][0]['result']['latency_scope'] for r in output)


def test_missing_caption_in_batch_fails_without_replacement(monkeypatch):
    model,_=setup_model(monkeypatch,['first caption',''])
    with pytest.raises(RuntimeError,match='incorrect-cardinality'):
        model.generate([np.zeros((2,3,3),np.uint8)]*2,dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))


def test_more_than_three_images_is_unqualified():
    model=LocalQwenBatchScene('snapshot','cuda:0',256,1024)
    with pytest.raises(ValueError,match='one to three'):
        model.generate([np.zeros((2,3,3),np.uint8)]*4,dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))
