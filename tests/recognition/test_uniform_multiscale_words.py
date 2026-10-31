import pytest

from recognition.uniform_multiscale_words import augment


def row(text,box,p):return dict(raw_text=text,bbox=box,sequence_confidence_native=p)


def test_native_prefix_and_distant_repetitions_survive_but_overlap_and_low_confidence_do_not():
    native=[row('word',[0,0,10,10],.5)]
    larger=[row('word',[0,0,10,10],.99),row('word',[20,0,30,10],.95),row('noise',[40,0,50,10],.8)]
    got=augment(native,larger,minimum_sequence_probability=.9,maximum_overlap=.3,maximum_candidates=50)
    assert got[:1]==native and len(got)==2 and got[1]['bbox']==[20,0,30,10]
    assert got[1]['source_row']==1 and got[1]['scale_source_row']==1


def test_extra_candidates_are_ranked_before_cap_and_invalid_policy_fails():
    larger=[row('weak',[20,0,30,10],.9),row('strong',[40,0,50,10],.99)]
    got=augment([],larger,minimum_sequence_probability=.9,maximum_overlap=.3,maximum_candidates=1)
    assert got[0]['raw_text']=='strong'
    with pytest.raises(ValueError):augment(larger,[],minimum_sequence_probability=.9,maximum_overlap=.3,maximum_candidates=1)
