import numpy as np
import pytest

from diagnostics.train_v1_ocr_failure_audit import native_box, pixel_metrics, mask_bin, crop_failure, scalar_summary


CFG={'mask_recall_bins':[.5,.8],'high_mask_precision':.8}


def test_mapping_floor_ceil_exclusive_borders():
    assert native_box([0,0,1000,500],(500,1000))==(0,0,128,128)
    assert native_box([1,1,2,2],(500,1000))==(0,0,1,1)
    assert native_box([500,250,1000,500],(500,1000))==(64,64,128,128)


def test_metrics_use_pixel_intersection_not_pred_count_over_gt_count():
    m=pixel_metrics(np.array([[1,0],[1,0]]),np.array([[0,1],[0,1]]))
    assert m['recall']==0 and m['precision']==0 and m['iou']==0
    assert m['gt_pixels']==m['pred_pixels']==2
    assert mask_bin(m,CFG)=='LOW_RECALL_LT_50'


def test_empty_foreground_has_explicit_undefined_precision_and_recall():
    m=pixel_metrics(np.zeros((2,2)),np.zeros((2,2)))
    assert m['precision'] is None and m['recall'] is None
    assert mask_bin(m,CFG)=='NO_NATIVE_TARGET_FOREGROUND'


@pytest.mark.parametrize('recall,precision,expected',[(.49,1,'LOW_RECALL_LT_50'),(.5,1,'PARTIAL_RECALL_50_80'),(.8,.79,'HIGH_RECALL_LOW_PRECISION'),(.8,.8,'HIGH_RECALL_HIGH_PRECISION')])
def test_mask_bins_boundaries(recall,precision,expected):
    assert mask_bin({'gt_pixels':10,'recall':recall,'precision':precision},CFG)==expected


@pytest.mark.parametrize('exact,available,multi,iou,expected',[(True,True,True,.2,'OCR_CORRECT'),(False,False,False,0,'NO_OVERLAPPING_CROP'),(False,True,True,.7,'MULTIWORD_CROP'),(False,True,False,.5,'GOOD_BOX_OCR_ERROR'),(False,True,False,.49,'LOW_IOU_CROP')])
def test_observed_crop_categories_not_inferred_causes(exact,available,multi,iou,expected):
    assert crop_failure(exact,available,multi,iou,.5)==expected


def test_null_fuzzy_missing_crop_remains_in_denominator():
    rows=[{'exact':False,'available':False,'fuzzy':None,'crop_failure':'NO_OVERLAPPING_CROP',
           'mask_bin':'LOW_RECALL_LT_50','geometry_category':'NO_FOREGROUND',
           'native_metrics':{'pred_pixels':0},'best_iou':0,'any_overlap_exact':False,'any_good_iou_exact':False}]
    s=scalar_summary(rows)
    assert s['words']==s['failures']==1 and s['available']==s['fuzzy_ge_0_8']==0


def test_invalid_metric_shapes_fail():
    with pytest.raises(ValueError):pixel_metrics(np.zeros((2,2)),np.zeros((3,3)))
