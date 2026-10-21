import numpy as np
import pytest

from diagnostics.train_v1_crop_glyph_audit import packed_rgb,color_counts,crop_glyph_retention


def test_exact_glyph_pixels_not_rectangular_area():
    rgb=np.full((10,20,3),255,dtype=np.uint8)
    rgb[2:8,1:4]=[1,2,3];rgb[2:8,10:13]=[4,5,6]
    packed=packed_rgb(rgb);totals=color_counts(packed)
    rows,_=crop_glyph_retention(packed,[[1,2,3],[4,5,6]],[5,0,20,10],totals)
    assert [r['retained_fraction'] for r in rows]==[0,1]
    rows,_=crop_glyph_retention(packed,[[1,2,3]],[0,0,20,5],totals)
    assert rows[0]['retained_fraction']==.5
    rows,_=crop_glyph_retention(packed,[[1,2,3]],None,totals)
    assert rows[0]['retained_fraction']==0


def test_missing_annotation_color_fails():
    packed=packed_rgb(np.full((2,2,3),255,dtype=np.uint8))
    with pytest.raises(ValueError):crop_glyph_retention(packed,[[1,2,3]],None,color_counts(packed))
