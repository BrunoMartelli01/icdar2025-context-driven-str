import numpy as np
import pytest

from recognition.residual_text_vqa import residual_views
from scripts.run_yolo_residual_vqa import raw_signals, score_features


def test_mask_union_preserves_outside_pixels_and_overlap_order():
    rgb=np.random.default_rng(0).integers(0,256,(20,30,3),dtype=np.uint8)
    boxes=[[2,3,12,13],[8,5,18,15]]
    a,meta=residual_views(rgb,boxes,mask_value=128,blur_fraction=.05)
    b,_=residual_views(rgb,list(reversed(boxes)),mask_value=128,blur_fraction=.05)
    mask=np.zeros(rgb.shape[:2],bool)
    for x1,y1,x2,y2 in boxes:mask[y1:y2,x1:x2]=True
    assert np.array_equal(a['original'],rgb)
    for key in ('hard_mask','blur_mask'):
        assert np.array_equal(a[key][~mask],rgb[~mask])
        assert np.array_equal(a[key],b[key])
    assert np.all(a['hard_mask'][mask]==128)
    assert meta['masked_pixels']==mask.sum()
    assert meta['masking_may_cover_unrecognized_text_inside_boxes'] is True
    with pytest.raises(ValueError,match='Out-of-image'):
        residual_views(rgb,[[-1,0,4,4]],mask_value=128,blur_fraction=.05)


def test_zero_boxes_leave_both_residuals_identical_to_original():
    rgb=np.zeros((5,6,3),np.uint8)
    views,meta=residual_views(rgb,[],mask_value=128,blur_fraction=.05)
    assert all(np.array_equal(rgb,x)for x in views.values())
    assert meta['mask_fraction']==0


def test_risk_uses_strongest_mask_and_same_context_control():
    def q(plain,ctx):return {'plain':{'yes_probability':plain},'context':{'yes_probability':ctx}}
    scores=raw_signals({'original':q(.9,.8),'hard_mask':q(.1,.4),'blur_mask':q(.2,.3),'blank':q(.01,.2)})
    assert scores['vqa_residual_plain']==-.2
    assert scores['vqa_visual_drop_plain']==pytest.approx(.7)
    assert scores['vqa_visual_drop_context']==pytest.approx(.4)
    assert scores['vqa_context_specific']==pytest.approx(-.3)
