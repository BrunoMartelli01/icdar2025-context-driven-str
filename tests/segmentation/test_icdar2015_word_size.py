import copy

import pytest

from segmentation.icdar2015_word_size import projected_component_area, size_admitted


def proposal(area=4, bbox=(0, 0, 10, 5)):
    return {'component_area':area,'bbox_native':[0,0,2,2],'bbox':list(bbox)}


def test_true_pixel_area_not_bounding_box_and_input_immutable():
    p=proposal(1,bbox=(0,0,100,100));before=copy.deepcopy(p)
    assert projected_component_area(p,(1280,720),(736,1280))==4
    assert size_admitted(p,(1280,720),(736,1280),minimum_area=4,minimum_height=100)
    assert not size_admitted(p,(1280,720),(736,1280),minimum_area=4.01,minimum_height=0)
    assert p==before


def test_projected_area_across_scales_and_original_height_boundary():
    p=proposal(4)
    assert projected_component_area(p,(1280,720),(736,1280))==16
    assert projected_component_area(proposal(16),(1280,720),(1440,2560))==16
    assert size_admitted(p,(1280,720),(736,1280),minimum_area=16,minimum_height=5)
    assert not size_admitted(p,(1280,720),(736,1280),minimum_area=0,minimum_height=6)


@pytest.mark.parametrize('area,height',[(-1,0),(0,-1),(float('nan'),0)])
def test_invalid_thresholds_fail(area,height):
    with pytest.raises(ValueError):size_admitted(proposal(),(1280,720),(736,1280),minimum_area=area,minimum_height=height)
