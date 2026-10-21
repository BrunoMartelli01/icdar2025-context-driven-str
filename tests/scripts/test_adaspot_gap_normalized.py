import numpy as np
import pytest
import torch

from localize.adaspot_gap_preserve import decode_gap, preserve_support
from models.adaspot_gap_context import sampled_patches
from models.adaspot_gap_normalized import NormalizedGapHead
from scripts.train_adaspot_gap_normalized import candidate_key, split_fit_calibration


@pytest.mark.parametrize("kind",["linear","nonlinear"])
def test_normalization_native_and_patch_match_including_padding(kind):
    torch.manual_seed(2)
    mean=torch.randn(576);scale=torch.rand(576)+.2
    head=NormalizedGapHead(kind,mean,scale,hidden=4,initial_bias=-2.)
    with torch.no_grad():
        layer=head.last if kind=="nonlinear" else head.first
        layer.weight.normal_(std=.02)
    feature=torch.randn(64,5,6);positions=torch.tensor([0,5,6,17,29])
    actual=head(sampled_patches(feature,positions));native=head.native_logits(feature).flatten()[positions]
    assert torch.allclose(actual,native,atol=2e-5)
    actual.sum().backward()
    assert head.mean.grad is None and head.scale.grad is None and feature.grad is None


def test_initial_heads_preserve_off_baseline_and_bad_normalization_fails():
    for kind in ["linear","nonlinear"]:
        head=NormalizedGapHead(kind,torch.zeros(576),torch.ones(576),hidden=4,initial_bias=-2.)
        assert torch.equal(head(torch.randn(3,576)),torch.full((3,),-2.))
    with pytest.raises(ValueError):NormalizedGapHead("linear",torch.zeros(576),torch.zeros(576),hidden=4,initial_bias=-2.)


def test_restoration_preserves_all_foreground_ids_and_unseeded_components():
    fg=np.zeros((8,12),bool);fg[1:4,1:10]=True;fg[6,2:4]=True
    seeds=np.zeros_like(fg,dtype=np.int32);seeds[2,2]=1;seeds[2,8]=2
    a=preserve_support(seeds,fg);b=preserve_support(seeds,fg)
    assert np.array_equal(a,b) and np.array_equal(a>0,fg)
    assert a[2,2]==1 and a[2,8]==2 and a[6,2]>2
    assert len(np.unique(a[a>0]))==3
    with pytest.raises(ValueError):preserve_support(seeds,~fg)


def test_full_gap_removal_has_explicit_fallback_and_no_gt_in_decoder():
    words=np.ones((2,6,9),np.float32);canvas=np.ones((6,9),bool)
    pp={"method":"kernel_bfs","text_threshold":.65,"kernel_threshold":.5,"center_threshold":.5,"gap_threshold":.5}
    gap=np.ones(canvas.shape)
    cut=decode_gap(words,gap,canvas,pp,.5,"cut")
    restored=decode_gap(words,gap,canvas,pp,.5,"preserve")
    assert not cut.any() and np.array_equal(restored>0,canvas)
    assert np.array_equal(decode_gap(words,None,canvas,pp,None,"cut")>0,canvas)


def test_fit_calibration_split_is_disjoint_repeatable_and_independent_of_row_order():
    rows=[{"sample_id":str(i)} for i in range(20)]
    a,b=split_fit_calibration(rows,42,4);x,y=split_fit_calibration(list(reversed(rows)),42,4)
    assert not set(a)&set(b) and len(a)==16 and len(b)==4
    assert set(a)==set(x) and set(b)==set(y)
    with pytest.raises(ValueError):split_fit_calibration(rows+rows,42,4)


def test_selection_calibration_criterion_keeps_off_on_ties_and_rewards_real_gain():
    def candidate(epoch,f1,fused,t,decoder):
        return {"epoch":epoch,"threshold":t,"decoder":decoder,"aggregate":{"word_f1":f1,"gt_words_touched_by_fused_components":fused}}
    off=candidate(0,.6,5,None,"cut")
    assert candidate_key(off)>candidate_key(candidate(5,.6,5,.99,"preserve"))
    assert candidate_key(candidate(5,.61,5,.75,"preserve"))>candidate_key(off)
    assert candidate_key(candidate(5,.61,4,.95,"cut"))>candidate_key(candidate(5,.61,5,.99,"cut"))
