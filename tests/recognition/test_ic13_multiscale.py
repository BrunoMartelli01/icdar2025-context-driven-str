import numpy as np
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import features
from scripts.prepare_ic13_multiscale import fused


def row(name,text,scale,box):
    return dict(candidate_id=name,raw_text=text,confidence=.9,detector_score=.9,scale=scale,bbox=box)


def test_scale_support_counts_distinct_views_and_preserves_repeated_words():
    lex=DeepSoloLexicon(['HELLO'],{'HELLO':'Hello'},distance_exclusive=1.5)
    raw=[row('a','HELLO',640,[0,0,10,10]),row('b','HEL1O',960,[0,0,10,10]),
         row('c','HELLO',640,[0,0,10,10]),row('d','HELLO',1280,[20,0,30,10])]
    result=fused(raw,lex,.5,2,.5,.7)
    assert len(result)==1 and result[0]['scale_support']==2
    result=fused(raw,lex,.5,1,.5,.7)
    assert len(result)==2


def test_empty_consensus_features_are_defined():
    f=features([],[],[],[100,100],use_consensus=True)
    assert np.array_equal(f['consensus'],np.zeros(8))
