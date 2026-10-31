import numpy as np
import pytest

from localize.directional_affinity_split import affinity_split
from localize.directional_closedness import neighbor


def example():
    owners=np.zeros((12,16),np.int32);owners[2:10,2:8]=1;owners[2:10,8:14]=2
    merged=(owners>0).astype(np.int32);canvas=np.ones(owners.shape,bool);edges=[]
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1),(-4,0),(4,0),(0,-4),(0,4)):
        other=neighbor(owners,dy,dx)
        p=np.where((owners>0)&(owners==other),.95,.05)
        edges.append((p,neighbor(canvas,dy,dx),dy,dx))
    return owners,merged,edges,canvas


def split(labels,edges,canvas,threshold=.7):
    return affinity_split(labels,edges,canvas,threshold=threshold,minimum_cells=4,minimum_fraction=.15,minimum_coverage=.7)


def test_neural_cut_separates_touching_words_without_losing_pixels():
    owners,merged,edges,canvas=example()
    out,audit=split(merged,edges,canvas)
    assert np.array_equal(out>0,merged>0)
    assert len(np.unique(out[owners==1]))==1 and len(np.unique(out[owners==2]))==1
    assert out[3,3]!=out[3,10] and audit['added_instances']==1


def test_existing_instances_never_merge_and_no_cut_control_is_exact():
    owners,merged,edges,canvas=example()
    out,audit=split(owners,edges,canvas)
    assert np.array_equal(out,owners) and audit['added_instances']==0
    out,_=split(merged,edges,canvas,threshold=None)
    assert np.array_equal(out,merged)


def test_overfragmentation_with_no_large_core_reverts_to_original():
    owners,merged,edges,canvas=example()
    weak=[(np.full_like(p,.1),v,dy,dx) for p,v,dy,dx in edges]
    out,audit=split(merged,weak,canvas)
    assert np.array_equal(out,merged) and audit['added_instances']==0


def test_weak_one_way_connection_does_not_override_opposite_boundary():
    owners,merged,edges,canvas=example()
    asymmetric=[(np.full_like(p,.99) if (dy,dx)==(0,1) else p,v,dy,dx) for p,v,dy,dx in edges]
    out,audit=split(merged,asymmetric,canvas)
    assert audit['added_instances']==1 and out[3,3]!=out[3,10]


def test_full_resolution_labels_and_invalid_affinity_guard():
    owners,merged,edges,canvas=example()
    fine=np.repeat(np.repeat(merged,2,axis=0),2,axis=1)
    out,_=split(fine,edges,canvas)
    assert np.array_equal(out>0,fine>0)
    with pytest.raises(ValueError,match='affinity'):
        split(merged,[(np.full_like(p,np.nan),v,dy,dx) for p,v,dy,dx in edges],canvas)
