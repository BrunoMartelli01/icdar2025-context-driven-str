from itertools import product

import numpy as np
import pytest

from localize.word_graph_cut import minimum_cut, partition, seed_pairs, select, make_proposal


ENERGY={'edge_floor':.02,'probability_power':2.,'color_sigma':.15,
    'unary_weight':1.,'pairwise_weight':2.,'integer_scale':256}


def test_integer_cut_matches_exhaustive_energy():
    d=np.array([[0.,1.],[.2,.3],[.5,.1],[1.,0.]])
    u=np.array([0,1,2,0]);v=np.array([1,2,3,3]);w=np.array([.2,.7,.3,.1])
    seeds=np.array([1,0,0,2]);labels,a=minimum_cut(d,u,v,w,seeds,integer_scale=256)
    costs=[]
    for middle in product([1,2],repeat=2):
        x=np.array([1,*middle,2]);costs.append(int(np.rint(d*256)[np.arange(4),x-1].sum()+np.rint(w*256)[x[u]!=x[v]].sum()))
    assert a['integer_energy']==a['max_flow']==min(costs)
    assert labels[0]==1 and labels[-1]==2


def test_cut_respects_diagonal_support_and_hard_markers():
    support=np.eye(7,dtype=bool);p=np.full((7,7),.8);rgb=np.zeros((7,7,3),np.uint8)
    seeds=np.zeros((7,7),np.int32);seeds[0,0]=1;seeds[-1,-1]=2
    result,a=partition(support,p,rgb,seeds,ENERGY,method='graphcut')
    assert np.array_equal(result>0,support)
    assert result[0,0]==1 and result[-1,-1]==2
    assert a['cut_edges']>=1


@pytest.mark.parametrize('method',['graphcut','geodesic'])
def test_bridge_partition_preserves_all_foreground_and_holes(method):
    support=np.zeros((18,35),bool);support[3:15,2:14]=1;support[3:15,21:33]=1;support[8:10,14:21]=1
    support[5:7,4:6]=0
    p=np.where(support,.95,.1);p[8:10,14:21]=.51
    seeds=np.zeros_like(support,dtype=np.int32);seeds[7:11,6:10]=1;seeds[7:11,25:29]=2
    labels,a=partition(support,p,np.zeros((*p.shape,3),np.uint8),seeds,ENERGY,method=method)
    assert np.array_equal(labels>0,support)
    assert (labels[3:15,2:14][support[3:15,2:14]]==1).all()
    assert (labels[3:15,21:33]==2).all()
    assert a['cut_probability']<.6


def test_deterministic_quantized_cut():
    p=np.full((8,20),.9);support=np.ones_like(p,bool);rgb=np.zeros((*p.shape,3),np.uint8)
    seeds=np.zeros(p.shape,np.int32);seeds[:,0]=1;seeds[:,-1]=2
    a,info=partition(support,p,rgb,seeds,ENERGY,method='graphcut')
    b,other=partition(support,p,rgb,seeds,ENERGY,method='graphcut')
    assert np.array_equal(a,b) and info==other


@pytest.mark.parametrize('value',[np.nan,-.1])
def test_invalid_energy_fails(value):
    with pytest.raises(ValueError):minimum_cut([[value,1],[0,1]],[0],[1],[1],[1,2],integer_scale=256)


def test_missing_seed_fails_instead_of_substituting():
    with pytest.raises(ValueError):minimum_cut([[0,1],[0,1]],[0],[1],[1],[1,0],integer_scale=256)


def test_capacity_overflow_is_explicit():
    with pytest.raises(OverflowError):minimum_cut([[0,1],[0,1]],[0],[1],[1e12],[1,2],integer_scale=256)


def test_two_seed_sets_are_shared_and_duplicate_free():
    support=np.zeros((15,30),bool);support[2:13,1:12]=1;support[2:13,18:29]=1;support[7,12:18]=1
    p=np.where(support,.95,.1);p[7,12:18]=.51
    c={'seed_thresholds':[.7,.85],'erosion_radii':[1],
       'minimum_seed_area':4,'minimum_seed_fraction':.03,'maximum_seed_regions':4}
    pairs=seed_pairs(support,p,c)
    assert pairs and len({s.tobytes() for s,_ in pairs})==len(pairs)
    assert all(set(np.unique(s))=={0,1,2} for s,_ in pairs)


def test_selection_preserves_baseline_when_off_and_respects_global_cap():
    base=[make_proposal([0,0,20,10],(384,640),1),make_proposal([30,0,40,10],(384,640),1)]
    children=[make_proposal([0,0,8,10],(384,640),1),make_proposal([10,0,20,10],(384,640),1)]
    groups=[{'id':'g','parent_index':0,'method':'graphcut','energy_index':0,
       'children':children,'child_area_fraction':.4,'audit':{'normalized_cut_affinity':.1,'cut_probability':.55}}]
    assert select(base,groups,{'method':'baseline'},2)==(base,[])
    p={'method':'graphcut','energy_index':0,'max_cut_probability':.8,'max_cut_affinity':.3}
    chosen,ids=select(base,groups,p,2)
    assert len(chosen)==2 and ids==['g']
    assert base[0] not in chosen


def test_fractional_native_mapping_uses_original_floor_and_ceil():
    p=make_proposal([1.5,2.5,4.5,5.5],(384,640),1)
    assert p['bbox']==[7,7,23,17]
    assert p['padded_bbox_original']==[2,4,28,20]
