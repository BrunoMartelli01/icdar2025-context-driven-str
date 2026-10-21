import numpy as np
import pytest

from recognition.text_coverage_router import coverage_features, projected_support, scores, fit_selector


def test_two_text_regions_one_unread_and_exclusive_coordinates():
    p = np.zeros((2, 8, 8), np.float32)
    p[:, 1:3, 1:3] = .8
    p[:, 5:7, 5:7] = .8
    f = coverage_features(p, (8, 8), [[1, 1, 3, 3]], probability_threshold=.5, minimum_component_area=2)
    assert f['text_coverage'] == pytest.approx(.5)
    assert f['kernel_uncovered_pixels'] == 4
    assert f['text_uncovered_components'] == 1
    assert not f['ground_truth_used']
    full = coverage_features(p, (8, 8), [[0, 0, 8, 8]], probability_threshold=.5, minimum_component_area=2)
    assert full['text_coverage'] == 1


def test_center_letterbox_padding_is_not_mistaken_for_uncovered_text():
    p = np.ones((2, 8, 8), np.float32)
    f = coverage_features(p, (8, 4), [[0, 0, 8, 4]], probability_threshold=.5, minimum_component_area=1)
    assert f['text_foreground_pixels'] == 32
    assert f['text_uncovered_pixels'] == 0
    assert f['text_coverage'] == 1


def test_empty_prediction_mask_is_explicit_and_neutral():
    f = coverage_features(np.zeros((2, 8, 8)), (8, 8), [], probability_threshold=.5, minimum_component_area=1)
    assert f['text_mask_empty'] and f['text_coverage'] == 1
    assert scores({'a': {'confidence_score': None, 'coverage': f}}, [1], .5, 'text_coverage')['a'] is None


def test_projection_matches_frozen_native_frame_with_odd_padding():
    from recognition.clip_router_evaluation import native_mask
    from PIL import Image, ImageDraw
    im = Image.new('L', (13, 8), 0)
    ImageDraw.Draw(im).rectangle([2, 1, 9, 6], fill=255)
    np.testing.assert_array_equal(projected_support((13, 8), [[2, 1, 10, 7]], 16), native_mask(im, 16))


@pytest.mark.parametrize('box', [[0,0,0,1], [-1,0,2,2], [0,0,9,8], [0,.1,2,2]])
def test_invalid_boxes_are_not_silently_dropped(box):
    with pytest.raises(ValueError): projected_support((8, 8), [box], 8)


def test_invalid_probabilities_are_errors():
    p = np.zeros((2, 8, 8)); p[0, 0, 0] = np.nan
    with pytest.raises(ValueError): coverage_features(p, (8,8), [], probability_threshold=.5, minimum_component_area=1)


def counts(tp, fp, fn):
    return dict(tp=tp,fp=fp,fn=fn,detected_words=tp+fp,gt_words=tp+fn,all_outputs=tp+fp,ignored_outputs=0)


def test_fit_coverage_recovers_wrong_acceptance_and_rejects_population_leakage():
    f={'a':{'confidence_score':.9,'coverage':{'text_coverage':.1}},
       'b':{'confidence_score':.8,'coverage':{'text_coverage':1.}}}
    b={'a':{'light':counts(0,1,1),'deepsolo':counts(1,0,0)},
       'b':{'light':counts(1,0,0),'deepsolo':counts(1,0,0)}}
    fitted=fit_selector(f,b,['a','b'],weights=[0,.5],coverage_fields=['text_coverage'],coverage_targets=[.5],primary_coverage=.5)
    assert fitted['winner']['weight']==.5
    assert fitted['winner']['accepted_ids']==['b']
    assert not fitted['test_labels_used']
    with pytest.raises(ValueError):fit_selector(f,b,['a'],weights=[0],coverage_fields=['text_coverage'],coverage_targets=[.5],primary_coverage=.5)
    assert scores(f,[.1,1.],0,'text_coverage') == {'a':.9,'b':.8}


def test_ties_keep_confidence_control_and_deterministic_budget():
    from recognition.clip_router_prompt_ablation import budget_accept
    f = {s:{'confidence_score':.5,'coverage':{'text_coverage':1.}} for s in ['b','a']}
    b = {s:{'light':counts(1,0,0),'deepsolo':counts(1,0,0)} for s in f}
    fit = fit_selector(f,b,list(f),weights=[.5,0],coverage_fields=['text_coverage'],
        coverage_targets=[.5],primary_coverage=.5)
    assert fit['winner']['weight'] == 0
    assert fit['winner']['accepted_ids'] == ['a']
    assert budget_accept({'b':.5,'a':.5},['b','a'],1) == {'a'}
    # A scalar threshold includes all ties; an exact rank budget need not.
    assert fit['scalar_thresholds']['0.5'] == .5


def test_missing_readings_cannot_be_accepted_to_fill_a_budget():
    f = {s:{'confidence_score':None,'coverage':{'text_coverage':1.}} for s in ['a','b']}
    b = {s:{'light':counts(0,0,1),'deepsolo':counts(1,0,0)} for s in f}
    with pytest.raises(ValueError,match='Insufficient nonempty'):
        fit_selector(f,b,list(f),weights=[0,.5],coverage_fields=['text_coverage'],
            coverage_targets=[.5],primary_coverage=.5)


def test_empty_readings_still_count_all_false_negatives():
    from recognition.paper_image_router_evaluation import word_metrics
    from recognition.yolo_router_comparison import aggregate
    gt = [{'text':'one','bbox':[0,0,2,2]}, {'text':'two','bbox':[2,2,4,4]}]
    result = aggregate([word_metrics([],gt)])
    assert (result['tp'],result['fp'],result['fn']) == (0,0,2)
    assert result['accuracy_tn0'] == 0


def test_no_foreground_padding_false_alarm_and_both_channels_remain_distinct():
    p = np.ones((2,8,8),np.float32)
    p[1,:,4:] = 0
    f = coverage_features(p,(8,4),[[0,0,4,4]],probability_threshold=.5,minimum_component_area=1)
    assert f['text_coverage'] == .5
    assert f['kernel_coverage'] == 1
