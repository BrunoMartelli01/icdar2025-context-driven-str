from types import SimpleNamespace

import pytest

from caption.blip2_captioner import BLIP2ApproxConfig
from caption.paper_style_blip2 import InstructionCaptionConfig
from caption.detailed_scene import detailed_profile


def config():return InstructionCaptionConfig(model_id='Salesforce/instructblip-flan-t5-xl',revision='bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d',device='cuda:0',dtype='float16')


def fake_batch(captioner,rgbs,profile,target):
    return [dict(answers=[dict(prompt=profile['prompts'][0],result=dict(raw_caption=profile['prompts'][0]))]) for _ in rgbs]


def test_instruction_model_has_own_family_and_raw_text(monkeypatch):
    monkeypatch.setattr('caption.detailed_scene.generate_batch',fake_batch)
    profile=dict(caption_model='instructblip',assembly='single',prompts=['Native  text!'],generation={})
    result=detailed_profile(SimpleNamespace(config=config()),['RGB'],profile,dict(min_words=60,max_words=160))[0]
    assert result['caption_family']=='InstructBLIP ablation'
    assert result['raw_caption']=='Native  text!'
    assert result['image_only_captioning'] and not result['parseq_in_prompt'] and not result['ground_truth_in_caption']


def test_multi_aspect_preserves_every_native_answer_and_only_adds_punctuation(monkeypatch):
    monkeypatch.setattr('caption.detailed_scene.generate_batch',fake_batch)
    profile=dict(caption_model='instructblip',assembly='paragraph_join',prompts=['Red sign','A chair!'],generation={})
    result=detailed_profile(SimpleNamespace(config=config()),['RGB','RGB'],profile,dict(min_words=60,max_words=160))
    assert len(result)==2
    assert result[0]['raw_caption']=='Red sign. A chair!'
    assert [a['result']['raw_caption'] for a in result[0]['answers']]==['Red sign','A chair!']


def test_mismatched_family_fails_before_inference(monkeypatch):
    def forbidden(*args):raise AssertionError('must not infer')
    monkeypatch.setattr('caption.detailed_scene.generate_batch',forbidden)
    with pytest.raises(ValueError,match='matching model family'):
        detailed_profile(SimpleNamespace(config=BLIP2ApproxConfig()),['RGB'],
            dict(caption_model='instructblip',assembly='single',prompts=['Describe.']),dict(min_words=60,max_words=160))


def test_wrong_native_batch_cardinality_propagates(monkeypatch):
    monkeypatch.setattr('caption.detailed_scene.generate_batch',lambda *args:[])
    with pytest.raises(RuntimeError,match='cardinality'):
        detailed_profile(SimpleNamespace(config=config()),['RGB'],
            dict(caption_model='instructblip',assembly='single',prompts=['Describe.']),dict(min_words=60,max_words=160))
