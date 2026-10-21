import numpy as np
import pytest
import torch

from segmentation.unet_auxiliary_ocr import (word_supervision, vocabulary,
    ctc_word_losses, segmentation_objective, selection_key)


def options(**kw):
    return dict(characters=list('abc'),max_length=10,time_steps=12,max_words=8,epoch=0,**kw)


def test_owner_order_survives_missing_native_word():
    owner=np.array([[0,1,1,0,3],[0,1,1,0,3]],dtype=np.int32)
    rows,a=word_supervision(owner,np.ones_like(owner),['A','b','c'],**options())
    assert [r['text'] for r in rows]==['a','c']
    assert [r['owner_id'] for r in rows]==[1,3]
    assert rows[0]['bbox']==[1,0,3,2]
    assert a['native_missing']==1 and a['expected_words']==3


def test_ignore_unknown_empty_and_ctc_repeats_are_explicit():
    owner=np.array([[1,2,3,4]],dtype=np.int32)
    settings=options();settings.update(time_steps=2)
    rows,a=word_supervision(owner,np.array([[1,1,1,0]]),['aa','z','','b'],**settings)
    assert not rows
    assert a['ctc_length_infeasible']==a['unsupported_character']==a['empty_transcript']==a['ignored_owner']==1


def test_dense_words_cycle_without_permanent_exclusion():
    owner=np.arange(1,7,dtype=np.int32)[None]
    seen=[]
    for epoch in range(3):
        settings=options();settings.update(epoch=epoch,max_words=2)
        rows,_=word_supervision(owner,np.ones_like(owner),['a']*6,**settings)
        seen.extend(r['owner_id'] for r in rows)
    assert sorted(seen)==list(range(1,7))


def test_owner_without_transcript_is_error():
    with pytest.raises(ValueError,match='Owner ID'):
        word_supervision(np.array([[2]],dtype=np.int32),np.ones((1,1)),['a'],**options())


def test_fit_vocabulary_casefold_is_unicode_and_does_not_strip_punctuation():
    assert vocabulary(['Straße','A!'])==sorted(set('strassea!'))


def test_ctc_blank_and_repeated_label_feasibility_are_not_silently_zeroed():
    logits=torch.randn(2,1,4,requires_grad=True).log_softmax(-1)
    for target in [[[1,1]],[[0]],[[4]], [[]]]:
        with pytest.raises(ValueError):ctc_word_losses(logits,target)


def test_ctc_character_normalization_and_gradient():
    raw=torch.randn(6,2,4,requires_grad=True);probs=raw.log_softmax(-1)
    targets=[[1,2],[1,1,2]]
    actual=ctc_word_losses(probs,targets)
    expected=torch.nn.functional.ctc_loss(probs,torch.tensor([1,2,1,1,2]),
        torch.tensor([6,6]),torch.tensor([2,3]),reduction='none')/torch.tensor([2,3])
    torch.testing.assert_close(actual,expected)
    actual.mean().backward();assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum()>0


def test_segmentation_ignore_gradient_and_existing_loss_equivalence():
    from segmentation.word_multitask import masked_bce_dice
    z=torch.randn(1,2,3,4,requires_grad=True)
    target=torch.zeros(1,1,3,4);target[:,:,1,1]=1
    valid=torch.ones_like(target);valid[:,:,:,0]=0
    batch=dict(mask=target,kernel=target,validity=valid,separation=torch.zeros_like(target))
    c=dict(bce_weight=1.,dice_weight=1.,smooth=1.,lambda_text=1.,lambda_kernel=1.,separation_weight=3.)
    actual=segmentation_objective(z,batch,c)
    expected=sum(masked_bce_dice(z[:,i:i+1],target,valid,bce_weight=1.,dice_weight=1.,smooth=1.) for i in range(2))
    torch.testing.assert_close(actual,expected)
    actual.backward();assert torch.count_nonzero(z.grad[:,:,:,0])==0


def test_epoch0_wins_ties_and_exact_count_precedes_hmean():
    def candidate(e,n,h):return dict(epoch=e,aggregate=dict(exact_casefold_iou_ge_50=n,hmean=h))
    assert selection_key(candidate(0,5,.5))>selection_key(candidate(20,5,.5))
    assert selection_key(candidate(20,6,.1))>selection_key(candidate(0,5,.9))
