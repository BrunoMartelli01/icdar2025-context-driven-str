from types import SimpleNamespace

import pytest

from recognition.paper_caption_comparison import (accepted_images, crop_transcript,
    evaluate_routing, score_transcript, whole_image_counts)
from recognition.common_budget_pipeline import counts
from scoring.paper_image_router import PaperRouterConfig


def test_all_outputs_preserved_in_spatial_order():
    rows=[dict(raw_text='word!',bbox=[5,5,9,9]),dict(raw_text='WORD',bbox=[0,0,4,4]),
          dict(raw_text='',bbox=[10,5,14,9])]
    result=crop_transcript(rows)
    assert [r['raw_text'] for r in result['words']]==['WORD','word!','']
    assert not result['eligible']
    assert rows[0]['raw_text']=='word!'


def test_punctuation_casefold_repeated_words_and_empty_predictions():
    rows=[dict(raw_text=x) for x in ['WORD!','word','---','wrong']]
    c=counts(rows,[dict(text='word'),dict(text='WORD.'),dict(text='missing')],[False]*4)
    assert (c['tp'],c['fp'],c['fn'])==(2,2,1)


def test_t1_is_one_prediction_and_does_not_invent_multiple_words():
    c=whole_image_counts('two words!', [dict(text='two'),dict(text='words')])
    assert (c['tp'],c['fp'],c['fn'])==(0,1,2)


def test_inclusive_threshold_zero_and_invalid_scores():
    scores=dict(a=.8,b=.79,c=None)
    assert accepted_images(scores,dict.fromkeys(scores,''),threshold=.8)==['a']
    assert accepted_images(scores,dict.fromkeys(scores,''),fallback_images=3)==[]
    with pytest.raises(ValueError,match='Nonfinite'):
        accepted_images({'a':float('nan')},{'a':''},threshold=.8)


def test_quota_ties_use_rgb_key_without_labels():
    assert accepted_images(dict(a=.5,b=.5,c=.1),dict(a='z',b='a',c='b'),fallback_images=2)==['b']
    with pytest.raises(ValueError,match='Eligible'):
        accepted_images(dict(a=None,b=.5),dict(a='z',b='a'),fallback_images=0)


class Semantic:
    config=SimpleNamespace(max_sequence_length=384)
    def compare_candidates(self,*args):return .9,.1


def tokenizer(text,**kwargs):return dict(input_ids=text.split())


def test_raw_router_uses_paper_formula_semantic_winner_and_no_metric_normalization():
    c=PaperRouterConfig(.6,.4,.8,'paper_selected','raw')
    f=score_transcript('caption',[], 'caption',Semantic(),tokenizer,c)
    assert f['decision']['semantic_winner']=='T1'
    assert f['decision']['C']==pytest.approx(.94)
    assert f['scores']['paper_selected']==pytest.approx(.94)
    assert f['scores']['joined_t3'] is None


def test_overflow_falls_back_without_truncation():
    c=PaperRouterConfig(.6,.4,.8,'paper_selected','raw')
    class FailSemantic(Semantic):
        def compare_candidates(self,*args):raise AssertionError('must not encode overflow')
    f=score_transcript('word '*385,[],'caption',FailSemantic(),tokenizer,c)
    assert f['context_overflow'] and f['decision'] is None
    assert f['scores']==dict(paper_selected=None,joined_t3=None)


def fixture():
    light=counts([dict(raw_text='right')],[dict(text='right')],[False])
    t1=whole_image_counts('wrong',[dict(text='right')])
    f={s:dict(scores=dict(paper_selected=.9,joined_t3=.2),decision=dict(semantic_winner='T1')) for s in ('a','b')}
    return f,dict.fromkeys(f,light),dict.fromkeys(f,t1),dict.fromkeys(f,light)


def test_selected_t1_and_fallback_denominators_and_counts():
    f,l,t,d=fixture();r=evaluate_routing(f,l,t,d,['a'],'paper_selected')
    assert (r['combined']['tp'],r['combined']['fp'],r['combined']['fn'])==(1,1,1)
    assert r['combined']['images']==2
    assert (r['deepsolo_images'],r['total_images'],r['deepsolo_percentage'])==(1,2,50)
    assert r['source_counts']==dict(T1=1,T3=0,DeepSolo=1)


def test_t3_ablation_keeps_same_light_output():
    f,l,t,d=fixture();r=evaluate_routing(f,l,t,d,['a'],'joined_t3')
    assert r['combined']['f1']==1


def test_missing_fallback_never_reduces_denominator():
    f,l,t,d=fixture();del d['b']
    with pytest.raises(ValueError,match='Missing branch'):
        evaluate_routing(f,l,t,d,[],'paper_selected')


def test_ineligible_acceptance_fails():
    f,l,t,d=fixture();f['a']['scores']['paper_selected']=None
    with pytest.raises(ValueError,match='Ineligible'):
        evaluate_routing(f,l,t,d,['a'],'paper_selected')
