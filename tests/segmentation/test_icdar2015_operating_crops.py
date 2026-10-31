"""GT-free crop generation preserves components, original boxes and tiny words."""

import copy
import numpy as np
import pytest

from segmentation.icdar2015_improvements import binary_proposals
from segmentation.icdar2015_operating_policy import component_crop


def test_axis_crop_is_identical_to_baseline_and_padding_does_not_mutate_box():
    rgb=np.arange(64*96*3,dtype=np.uint16).reshape(64,96,3).astype(np.uint8)
    mask=np.zeros((32,48),bool);mask[10:14,15:30]=1
    proposals,labels=binary_proposals(mask,(96,64),(64,96));p=proposals[0];before=copy.deepcopy(p)
    x1,y1,x2,y2=p["bbox"]
    np.testing.assert_array_equal(component_crop(rgb,labels,p,(64,96),"axis0"),rgb[y1:y2,x1:x2])
    padded=component_crop(rgb,labels,p,(64,96),"axis2")
    assert padded.shape[:2]==(y2-y1+4,x2-x1+4)
    assert p==before


@pytest.mark.parametrize("method",["axis0","axis2","oriented0","oriented2"])
def test_singleton_components_and_edge_crops_are_retained(method):
    rgb=np.full((64,96,3),127,np.uint8);mask=np.zeros((32,48),bool);mask[0,0]=1
    proposals,labels=binary_proposals(mask,(96,64),(64,96))
    crop=component_crop(rgb,labels,proposals[0],(64,96),method)
    assert crop.size>0 and crop.ndim==3 and np.all(crop==127)


def test_rotated_component_is_rectified_with_long_axis_horizontal():
    rgb=np.zeros((64,96,3),np.uint8);mask=np.zeros((32,48),bool)
    for i in range(8):mask[10+i:12+i,15+i:18+i]=1
    proposals,labels=binary_proposals(mask,(96,64),(64,96))
    crop=component_crop(rgb,labels,proposals[0],(64,96),"oriented0")
    assert crop.shape[1]>crop.shape[0]
