import cv2
import numpy as np
import pytest

from segmentation.icdar2015_seed_partition import separated_mask


def run(p,low=.3,high=.8,canvas=None):
    return separated_mask(p,np.ones(p.shape,bool) if canvas is None else canvas,
                          support_threshold=low,seed_threshold=high)[0]


def test_fused_support_separates_without_losing_high_confidence_words():
    p=np.zeros((12,22),np.float32);p[3:9,2:8]=.9;p[3:9,14:20]=.9;p[5:7,8:14]=.4
    assert cv2.connectedComponents((p>=.3).astype('uint8'),connectivity=8)[0]-1==1
    result=run(p)
    assert cv2.connectedComponents(result.astype('uint8'),connectivity=8)[0]-1==2
    assert result[p>=.8].all()


def test_equal_threshold_exactly_replays_binary_mask():
    p=np.random.default_rng(71).random((18,29)).astype('float32');canvas=np.ones(p.shape,bool);canvas[:2]=False
    assert np.array_equal(run(p,.5,.5,canvas),(p>=.5)&canvas)


def test_unseeded_singleton_and_diagonal_component_survive():
    p=np.zeros((12,12),np.float32);p[1,1]=.4;p[8,8]=p[9,9]=.9
    result=run(p)
    assert result[1,1] and result[8,8] and result[9,9]
    assert cv2.connectedComponents(result.astype('uint8'),connectivity=8)[0]-1==2
    assert not run(np.zeros_like(p)).any()


def test_canvas_clips_only_explicit_image_support():
    p=np.ones((8,12),np.float32)*.9;canvas=np.zeros(p.shape,bool);canvas[2:6,2:9]=True
    assert np.array_equal(run(p,canvas=canvas),canvas)


@pytest.mark.parametrize('low,high',[(0.,.8),(.8,.3),(.3,1.)])
def test_invalid_thresholds_fail(low,high):
    with pytest.raises(ValueError):run(np.zeros((4,4),np.float32),low,high)
