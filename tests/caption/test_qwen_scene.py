import numpy as np
import pytest

from caption.qwen_scene import LocalQwenScene


def loaded(monkeypatch,text='A blue oval sign.'):
    import torch
    for name in ('init','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:20)
    captured={}
    class Inputs(dict):
        def to(self,*args):return self
    class Processor:
        def apply_chat_template(self,messages,**kwargs):captured['messages']=messages;return 'native image chat'
        def __call__(self,**kwargs):return Inputs(input_ids=torch.tensor([[1,2,3]]),image_grid_thw=torch.tensor([[1,28,28]]))
        def batch_decode(self,tokens,**kwargs):captured['decoded_tokens']=tokens.tolist();return [text]
    class Model:
        def generate(self,**kwargs):return torch.tensor([[1,2,3,8,9]])
    model=LocalQwenScene('snapshot','cuda:0',256*28*28,1024*28*28)
    model.load=lambda:None;model.processor=Processor();model.model=Model()
    return model,captured


def test_native_generated_suffix_excludes_input_chat_and_records_grid(monkeypatch):
    model,captured=loaded(monkeypatch)
    p=dict(prompt='Describe visible details.',generation=dict(max_new_tokens=256,do_sample=False))
    result=model.generate([np.zeros((12,18,3),np.uint8)],p,dict(min_words=60,max_words=160))[0]
    assert captured['decoded_tokens']==[[8,9]]
    assert captured['messages'][0]['content']==[dict(type='image'),dict(type='text',text=p['prompt'])]
    assert result['raw_caption']=='A blue oval sign.'
    assert result['answers'][0]['result']['image_grid_thw']==[[1,28,28]]
    assert result['caption_family']=='Qwen2-VL-7B caption ablation'
    assert not result['ground_truth_in_caption'] and not result['parseq_in_prompt']


def test_empty_caption_fails_without_replacement(monkeypatch):
    model,_=loaded(monkeypatch,' ')
    with pytest.raises(RuntimeError,match='Empty native'):
        model.generate([np.zeros((2,3,3),np.uint8)],dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))


def test_unqualified_multiple_image_batch_fails_before_load():
    model=LocalQwenScene('snapshot','cuda:0',256,1024)
    with pytest.raises(ValueError,match='one RGB'):
        model.generate([np.zeros((2,3,3),np.uint8)]*2,dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))


def test_cpu_weight_offload_device_map_is_recorded_explicitly(monkeypatch):
    model,_=loaded(monkeypatch);model.placement='cpu_offload'
    model.model.hf_device_map={'visual':0,'lm_head':'cpu'}
    result=model.generate([np.zeros((2,3,3),np.uint8)],dict(prompt='Describe.',generation={}),dict(min_words=60,max_words=160))[0]
    native=result['answers'][0]['result']
    assert native['placement']=='cpu_offload'
    assert native['hf_device_map']=={'visual':'0','lm_head':'cpu'}
