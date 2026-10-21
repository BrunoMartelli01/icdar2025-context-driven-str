"""Scientific reporting: paired populations, micro metrics and matching bounds."""
from pathlib import Path
import runpy

import pytest

MODULE=runpy.run_path(str(Path(__file__).resolve().parents[2]/
    'artifacts/diagnostics/research-synthesis-20261021-v1/analyze.py'))


def test_identical_policies_have_zero_paired_uncertainty():
    values={'a':[3,5,6],'b':[0,1,2]}
    report=MODULE['bootstrap'](values,values,200,42)
    assert report['correct_delta']==0
    assert report['f1_delta_ci95_percentage_points']==[0.,0.]


def test_aggregate_is_micro_f1_and_not_mean_image_f1():
    old={'large':[9,10,10],'small':[0,1,1]}
    new={'large':[9,10,10],'small':[1,1,1]}
    report=MODULE['bootstrap'](old,new,200,42)
    assert report['f1_delta_percentage_points']==pytest.approx(100/11)
    assert report['images_more_correct']==1 and report['images_less_correct']==0


def test_different_image_population_fails():
    with pytest.raises(ValueError,match='population'):
        MODULE['bootstrap']({'a':[1,1,1]},{'b':[1,1,1]},100,42)


def test_changed_per_image_gt_denominator_fails():
    with pytest.raises(ValueError,match='denominators'):
        MODULE['bootstrap']({'a':[1,1,1]},{'a':[1,1,2]},100,42)


@pytest.mark.parametrize('right,count',[(20,1),(21,0)])
def test_exact_matching_respects_inclusive_iou_threshold(right,count):
    word=[{'bbox':[0,0,10,10],'text':'WORD!'}]
    prediction=[{'bbox':[0,0,right,10],'text':'word!'}]
    assert MODULE['matching'](prediction,word,.5)==count


def test_empty_or_duplicate_predictions_cannot_inflate_oracle():
    word=[{'bbox':[0,0,10,10],'text':'A'}]
    prediction=[{'bbox':[0,0,10,10],'text':'a'}]*3
    assert MODULE['matching']([],word,.5)==0
    assert MODULE['matching'](prediction,word,.5)==1
    assert MODULE['matching'](prediction,[],.5)==0


def test_identical_rgb_with_different_detection_geometry_is_not_collapsed():
    a={'bbox':[0,0,10,10],'text':'A','crop_array_sha256':'pixels'}
    b={**a,'bbox':[0,0,9,10]}
    assert len(MODULE['deduplicate']([a,a,b]))==2
