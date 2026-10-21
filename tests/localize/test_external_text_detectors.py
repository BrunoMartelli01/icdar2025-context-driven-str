"""Geometry/RGB boundary checks independent of optional pretrained runtimes."""
import cv2
import numpy as np
import pytest

from localize.external_text_detectors import crop_region, doctr_regions, region


def test_doc_tr_denormalizes_box_axes_using_original_image_dimensions():
    rows = doctr_regions({'words':np.array([[.1,.2,.4,.6,.8]])}, (100,50))
    assert rows[0]['bbox'] == [10,10,40,30]
    assert rows[0]['score'] == .8
    quad = np.array([[[.1,.2],[.4,.2],[.4,.6],[.1,.6],[0,.8]]])
    result = doctr_regions({'words':quad}, (100,50))
    assert result[0]['bbox'] == rows[0]['bbox']
    assert result[0]['score'] == rows[0]['score']


def test_clipping_logs_outside_region_and_malformed_outputs_raise():
    assert region([[-5,0],[-1,0],[-1,10],[-5,10]], .8,(20,20),0)['rejection'] == 'no in-image area'
    assert region([[-5,0],[10,0],[10,10],[-5,10]], .8,(20,20),0)['bbox'] == [0,0,10,10]
    for score in (float('nan'),1.2,-.1):
        with pytest.raises(ValueError): region([[0,0],[3,0],[3,3],[0,3]],score,(20,20),0)
    assert region([[1,1]] * 4,.9,(20,20),0)['rejection'] == 'degenerate detector polygon'
    with pytest.raises(ValueError): doctr_regions({'unknown':np.empty((0,5))},(20,20))
    with pytest.raises(ValueError): doctr_regions({'words':np.empty((1,4))},(20,20))
    with pytest.raises(ValueError): doctr_regions({'words':np.array([[[0,0],[1,0],[1,1],[0,1],[.8,.9]]])},(20,20))


def test_empty_detections_stay_empty_without_dummy_regions():
    assert doctr_regions({'words':np.empty((0,5))},(20,20)) == []
    assert doctr_regions({'words':np.empty((0,5,2))},(20,20)) == []


def test_direct_crop_preserves_rgb_even_when_all_three_dimensions_are_three():
    image = np.arange(8*9*3,dtype=np.uint8).reshape(8,9,3)
    row = region([[2,1],[5,1],[5,4],[2,4]],.9,(9,8),0)
    crop = crop_region(image,row,'axis_aligned')
    assert crop.shape == (3,3,3) and crop.flags.c_contiguous
    np.testing.assert_array_equal(crop,image[1:4,2:5])
    with pytest.raises(ValueError): crop_region(image,row,'unknown')


def test_rectification_aligns_rotated_long_axis_and_keeps_rgb():
    image = np.full((60,80,3),(220,40,15),np.uint8)
    polygon = cv2.boxPoints(((40,30),(30,8),35))
    row = region(polygon,.9,(80,60),0)
    crop = crop_region(image,row,'rectified')
    assert crop.shape[1] >= crop.shape[0] and crop.shape[2] == 3
    np.testing.assert_array_equal(crop[crop.shape[0]//2,crop.shape[1]//2],[220,40,15])


def test_rejected_boxes_and_wrong_channels_are_never_sent_to_ocr():
    image = np.zeros((20,20,3),np.uint8)
    row = region([[0,0],[5,0],[5,5],[0,5]],.9,(20,20),0)
    with pytest.raises(ValueError): crop_region(image[:,:,:1],row,'axis_aligned')
    with pytest.raises(ValueError): crop_region(image.astype(np.float32),row,'axis_aligned')
    with pytest.raises(ValueError): crop_region(image,{**row,'valid':False},'axis_aligned')


@pytest.mark.parametrize('architecture',['db','fast'])
def test_real_pinned_doctr_postprocessor_quad_contract(architecture):
    # Exercise the installed implementation; its score encoding is independent
    # evidence for the adapter, rather than a manually invented model response.
    from doctr.models.detection.differentiable_binarization.base import DBPostProcessor
    from doctr.models.detection.fast.base import FASTPostProcessor
    processor=(DBPostProcessor if architecture=='db' else FASTPostProcessor)(assume_straight_pages=False)
    probability=np.full((64,96),.8,np.float32)
    bitmap=np.zeros((64,96),np.uint8); bitmap[20:40,20:70]=1
    values=processor.bitmap_to_boxes(probability,bitmap)
    assert values.shape == (1,5,2)
    assert values[0,4,0] == 0 and values[0,4,1] == pytest.approx(.8)
    rows=doctr_regions({'words':values},(960,640))
    assert len(rows) == 1 and rows[0]['valid'] and rows[0]['score'] == pytest.approx(.8)
    assert len(rows[0]['polygon']) == 4
