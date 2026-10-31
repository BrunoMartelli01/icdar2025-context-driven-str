import pytest

from recognition.uniform_word_admission import admit,select_threshold


def test_original_indices_and_inclusive_probability_cut_preserve_word_multiplicity():
    rows=[dict(source_row=2,raw_text='same'),dict(source_row=0,raw_text='same')]
    out=admit(rows,[.5,.1,.9],.5)
    assert [r['source_row'] for r in out]==[2,0]
    assert 'word_probability' not in rows[0]
    assert len(admit(rows,[.5,.1,.9],.6))==1
    with pytest.raises(ValueError):admit(rows,[.5],.5)
    with pytest.raises(ValueError):admit(rows,[.5,.1,float('nan')],.5)


def test_validation_tie_prefers_retaining_more_words_with_lower_cut():
    trials=[dict(threshold=.5,counts=dict(f1=.90)),dict(threshold=.3,counts=dict(f1=.8995)),dict(threshold=0.,counts=dict(f1=.88))]
    assert select_threshold(trials,.001)['threshold']==.3
    assert select_threshold(trials,0.)['threshold']==.5
    with pytest.raises(ValueError):select_threshold([],.001)
