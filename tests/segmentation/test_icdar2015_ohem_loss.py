"""Hard negatives must affect only the declared pixels and preserve empty images."""

import pytest
import torch

from segmentation.icdar2015_target70_ohem import ohem_loss,word_selection_key


def batch(target,valid=None):
    target=torch.tensor(target,dtype=torch.float32).reshape(1,1,1,-1)
    return {"mask":target,"validity":torch.ones_like(target) if valid is None else torch.tensor(valid).reshape_as(target).float(),
        "separation":torch.zeros_like(target),"ignored_negative":torch.zeros_like(target)}


def test_all_positives_and_only_three_hardest_negatives_have_gradient():
    logits=torch.tensor([0.,-4.,-3.,-2.,-1.,0.,2.,100.]).reshape(1,1,1,-1).requires_grad_()
    data=batch([1,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0])
    loss,counts=ohem_loss(logits,data,negative_ratio=3,separation_weight=1,ignore_negative_weight=1)
    loss.backward();gradient=logits.grad.flatten()
    assert gradient[0]==pytest.approx(-.125)
    assert gradient[1:4].eq(0).all() and gradient[4:7].gt(0).all() and gradient[7]==0
    assert counts=={"positive_pixels":1,"available_negative_pixels":6,"selected_negative_pixels":3}


def test_negative_only_images_keep_all_valid_background_gradients():
    logits=torch.zeros(1,1,1,4,requires_grad=True);data=batch([0,0,0,0],[1,1,1,0])
    loss,counts=ohem_loss(logits,data,negative_ratio=3,separation_weight=3,ignore_negative_weight=3)
    loss.backward();assert logits.grad.flatten()[:3].gt(0).all() and logits.grad.flatten()[3]==0
    assert counts["selected_negative_pixels"]==3
    with pytest.raises(ValueError,match="no valid"):
        ohem_loss(logits,batch([0,0,0,0],[0,0,0,0]),negative_ratio=3,separation_weight=3,ignore_negative_weight=3)


def test_invalid_positive_or_nonfinite_predictions_fail():
    data=batch([1,0]);data["ignored_negative"][0,0,0,0]=1
    with pytest.raises(ValueError,match="Invalid OHEM"):
        ohem_loss(torch.zeros_like(data["mask"]),data,negative_ratio=3,separation_weight=3,ignore_negative_weight=3)
    with pytest.raises(FloatingPointError,match="Nonfinite"):
        ohem_loss(torch.full_like(data["mask"],float('nan')),batch([1,0]),negative_ratio=3,separation_weight=3,ignore_negative_weight=3)


def test_raw_word_quality_is_the_primary_checkpoint_metric():
    def scores(raw,emitted):return ({"alphanumeric_casefold":{"f1":raw}},{"alphanumeric_casefold":{"f1":emitted}})
    geometry={"proposal_f1_iou_50":.7}
    assert word_selection_key(*scores(.51,.60),geometry,10)>word_selection_key(*scores(.50,.70),geometry,0)
