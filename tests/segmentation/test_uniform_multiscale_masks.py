import numpy as np
import pytest

from segmentation.uniform_multiscale_masks import fuse


def test_identity_and_constant_probability_average_preserve_grid_and_channels():
    a=np.random.default_rng(3).random((2,32,32),dtype=np.float32)
    assert np.array_equal(fuse(a,a,.5),a)
    b=np.full((2,64,64),.8,np.float32)
    np.testing.assert_allclose(fuse(a,b,.5),.5*a+.4,rtol=1e-6)
    assert fuse(a,b,.5).shape==a.shape


def test_half_pixel_alignment_of_a_ramp_is_explicit_and_no_channel_mix():
    x=np.arange(64,dtype=np.float32)/64
    b=np.stack((np.broadcast_to(x,(64,64)),np.ones((64,64),np.float32)))
    a=np.zeros((2,32,32),np.float32)
    out=fuse(a,b,1.)
    np.testing.assert_allclose(out[0,0],(2*np.arange(32)+.5)/64)
    assert np.all(out[1]==1.)


def test_invalid_probabilities_and_weight_fail():
    a=np.zeros((2,32,32),np.float32)
    for b in [np.full_like(a,np.nan),a.astype(np.float64),np.full_like(a,1.1),a[:1]]:
        with pytest.raises(ValueError):fuse(a,b,.5)
    with pytest.raises(ValueError):fuse(a,a,float('nan'))
