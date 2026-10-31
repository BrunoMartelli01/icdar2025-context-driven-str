import numpy as np

from caption.florence_scene import LocalFlorenceScene


def test_native_special_tokens_are_excluded_from_caption_but_raw_sequence_is_preserved(monkeypatch):
    import torch
    for name in ('init','reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda *args:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *args:10)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *args:20)
    class Processor:
        def __call__(self,**kwargs):return {}
        def batch_decode(self,tokens,skip_special_tokens):
            return ['A red sign.'] if skip_special_tokens else ['<s>A red sign.</s><pad>']
        def post_process_generation(self,text,task,image_size):return {task:text}
    class Model:
        def generate(self,**kwargs):return torch.tensor([[1,2,3]])
    captioner=LocalFlorenceScene('snapshot','cuda:0');captioner.load=lambda:None
    captioner.model=Model();captioner.processor=Processor()
    result=captioner.generate([np.zeros((2,3,3),np.uint8)],
        dict(task='<MORE_DETAILED_CAPTION>',generation={}),dict(min_words=60,max_words=160))[0]
    assert result['raw_caption']=='A red sign.'
    assert result['answers'][0]['result']['raw_native_sequence']=='<s>A red sign.</s><pad>'
    assert result['answers'][0]['result']['generated_token_ids']==[1,2,3]
