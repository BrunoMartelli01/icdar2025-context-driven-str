import pytest

from pipeline.uniform_scaled_frontend import scaled_policy


def test_native_coordinate_grid_changes_with_canvas_without_mutating_rgb_padding_or_limits():
    policy=dict(native_size=384,padding=4,maximum_ocr_candidates=50,text_threshold=.5,kernel_threshold=.5,minimum_transcript_probability=.9)
    for canvas,native in [(512,256),(768,384),(1024,512),(1280,640)]:
        actual=scaled_policy(policy,canvas)
        assert actual['native_size']==native and actual['padding']==4 and actual['maximum_ocr_candidates']==50
        assert actual['text_threshold']==actual['kernel_threshold']==.5
    assert policy['native_size']==384 and policy['minimum_transcript_probability']==.9
    for canvas in [True,0,1000,1024.]:
        with pytest.raises(ValueError):scaled_policy(policy,canvas)
