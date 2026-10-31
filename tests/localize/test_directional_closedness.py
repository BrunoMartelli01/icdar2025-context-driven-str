import numpy as np
import pytest

from localize.directional_closedness import closedness, neighbor


def test_complete_word_scores_above_a_fragment_with_same_internal_affinity():
    labels=np.zeros((8,8),np.int32);labels[2:6,2:6]=1
    canvas=np.ones((8,8),bool);edges=[]
    for dy,dx in ((0,1),(0,-1),(1,0),(-1,0)):
        other=neighbor(labels,dy,dx)
        probability=np.where((labels==1)&(other==1),.95,.05)
        border=neighbor(canvas,dy,dx)
        edges.append((probability,border,dy,dx))
    whole=closedness(labels,edges,canvas,[1])[1]
    fragment=labels.copy();fragment[4:6]=0
    part=closedness(fragment,edges,canvas,[1])[1]
    assert whole['available'] and part['available']
    assert whole['score'] > part['score']
    assert whole['internal_mean']==pytest.approx(part['internal_mean'])


def test_no_wrap_and_missing_support_is_explicit():
    x=np.arange(12).reshape(3,4)
    assert np.array_equal(neighbor(x,0,1)[:,-1],np.zeros(3))
    edges=[(np.full((4,4),.5),np.ones((4,4),bool),0,1)]
    result=closedness(np.zeros((8,8),np.int32),edges,np.ones((4,4),bool),[1])[1]
    assert result['available'] is False and result['score'] is None


def test_bad_affinities_cannot_become_rejected_components():
    with pytest.raises(ValueError,match='Nonfinite'):
        closedness(np.ones((4,4),np.int32),[(np.full((4,4),np.nan),np.ones((4,4),bool),0,1)],np.ones((4,4),bool),[1])
