import numpy as np
import pytest
from diagnostics.parseq_mask_region_audit import pixel_metrics,failure_bin
from recognition.segmentation_word_topology import _exclusive_gt_box,_native_word_box


def test_true_pixel_recall_is_intersection_not_predicted_area_ratio():
    target=np.array([[1,1],[0,0]])
    pred=np.array([[1,0],[1,1]])
    m=pixel_metrics(target,pred)
    assert m['pixel_recall']==.5
    assert m['pixel_precision']==1/3
    assert m['pixel_iou']==.25
    assert m['pixel_f1']==.4


def test_zero_target_and_absent_prediction():
    z=np.zeros((2,2));o=np.ones((2,2))
    assert failure_bin(pixel_metrics(z,z),0,False)=='NO_NATIVE_GT_FOREGROUND'
    assert failure_bin(pixel_metrics(o,z),0,False)=='LOW_MASK_RECALL_LT_50'


def test_good_mask_bad_crop_is_not_called_bad_segmentation():
    m=pixel_metrics(np.ones((2,2)),np.ones((2,2)))
    assert failure_bin(m,.1,False)=='HIGH_MASK_RECALL_PRECISION_BUT_POOR_CROP'
    assert failure_bin(m,.7,False)=='HIGH_MASK_RECALL_PRECISION_AND_GOOD_BOX_OCR_UNRESOLVED'
    assert failure_bin(m,.7,True)=='OCR_CORRECT'


def test_inclusive_gt_border_and_non_square_mapping():
    box=_exclusive_gt_box((0,0,639,319),(640,320))
    assert box==(0,0,640,320)
    assert _native_word_box(box,(640,320))==(0,0,128,128)
    assert _native_word_box((320,160,640,320),(640,320))==(64,64,128,128)


def test_reject_unaligned_masks():
    with pytest.raises(ValueError):
        pixel_metrics(np.ones((2,2)),np.ones((3,2)))
