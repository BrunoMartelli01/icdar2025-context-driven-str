"""Perfect-localization diagnostics must retain RGB corner geometry."""

import numpy as np
import pytest

from scripts.diagnose_icdar2015_target70_oracle import quad_crop


def test_axis_aligned_quad_matches_exact_inclusive_rgb_crop():
    image=np.arange(20*30*3,dtype=np.uint16).reshape(20,30,3).astype(np.uint8)
    crop=quad_crop(image,[(3,4),(12,4),(12,10),(3,10)])
    np.testing.assert_array_equal(crop,image[4:11,3:13])


def test_self_crossing_quad_fails_instead_of_silently_changing_geometry():
    with pytest.raises(ValueError,match="convex"):
        quad_crop(np.zeros((20,30,3),dtype=np.uint8),[(3,4),(12,10),(12,4),(3,10)])
