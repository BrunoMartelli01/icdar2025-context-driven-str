import numpy as np
import pytest

from recognition.uniform_word_union import candidates,training_labels,select


def light(text,box,row=0):
    return dict(raw_text=text,bbox=box,source_row=row,word_probability=.9,detector_score=.8,sequence_confidence_native=.95)


def heavy(text,box,index=0):
    return dict(raw_text=text,text=text,bbox=box,instance_id=index,score=.8,ignored=False)


def test_agreeing_overlap_merges_but_distant_repeated_words_are_preserved():
    rows,x=candidates([light('CAT',[0,0,10,10]),light('CAT',[20,0,30,10],1)],
        [heavy('cat',[0,0,10,10])],[40,20],.5)
    assert len(rows)==2 and x.shape==(2,24)
    assert rows[0]['union_source']=='deepsolo' and rows[1]['fixed_light_complement']


def test_prediction_features_ignore_ground_truth_and_evaluation_ignore_flags():
    a=heavy('cat',[0,0,10,10]);_,x=candidates([], [a],[40,20],.5)
    a.update(ignored=True,ground_truth='secret',recognition_confidence=.99);_,y=candidates([], [a],[40,20],.5)
    np.testing.assert_array_equal(x,y)


def test_labels_preserve_multiset_and_ignore_denominators():
    rows,_=candidates([light('CAT',[20,0,30,10])],[heavy('CAT',[0,0,10,10])],[40,20],.5)
    truth=[dict(text='CAT',bbox=[20,0,30,10])]
    assert training_labels(rows,truth,[False,False]).tolist()==[0,1]
    assert training_labels(rows,truth,[False,True]).tolist()==[1,0]


def test_conflict_selection_uses_probability_with_inclusive_boundary():
    rows,_=candidates([light('CAT',[0,0,10,10])],[heavy('DOG',[0,0,10,10])],[40,20],.5)
    selected=select(rows,[.7,.8],.8,.7)
    assert len(selected)==1 and selected[0]['raw_text']=='CAT'
    with pytest.raises(ValueError):select(rows,[1.1,.8],.8,.7)
