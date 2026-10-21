import pytest

from scripts.evaluate_native128_crop_limit import reconstruct, select


def test_reconstruction_restores_rejected_contours_and_opencv_order():
    shard={'variants':{'0':{'raw_contours':3,'proposals':[
        {'contour_index':2,'bbox_native':[50,50,70,60],'bbox_area_native':200}],
        'rejected':[{'contour_index':1,'bbox':[30,30,40,40],'bbox_area':100},
                    {'contour_index':0,'bbox':[1,1,3,3],'bbox_area':4}]}}}
    candidates=reconstruct(shard)
    assert [p['contour_index'] for p in candidates]==[0,1,2]
    assert [p['contour_index'] for p in select(candidates,(256,256),area=0,padding=1,limit=None)]==[2,1,0]
    assert [p['contour_index'] for p in select(candidates,(256,256),area=150,padding=1,limit=None)]==[2]


def test_caps_are_nested_and_unlimited_really_keeps_more_than20():
    candidates=[{'contour_index':i,'bbox_native':[1,1,3+i,3],'area':2*(2+i)} for i in range(25)]
    all_rows=select(candidates,(384,640),area=0,padding=2,limit=None)
    assert len(all_rows)==25
    for cap in [5,10,20]:
        assert select(candidates,(384,640),area=0,padding=2,limit=cap)==all_rows[:cap]


def test_equal_padded_area_ties_preserve_original_contour_order():
    candidates=[{'contour_index':i,'bbox_native':[10+i*20,10,20+i*20,20],'area':100} for i in range(3)]
    assert [p['contour_index'] for p in select(candidates,(128,128),area=0,padding=1,limit=None)]==[0,1,2]


def test_reconstruction_rejects_missing_or_duplicate_contours():
    with pytest.raises(ValueError,match='cover every'):
        reconstruct({'variants':{'0':{'raw_contours':1,'proposals':[],'rejected':[]}}})


@pytest.mark.parametrize('limit',[0,-1,False,1.5])
def test_invalid_cap_does_not_silently_mean_unlimited(limit):
    with pytest.raises(ValueError,match='positive integer'):
        select([],(128,128),area=0,padding=0,limit=limit)
