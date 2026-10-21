from itertools import combinations

import pytest

from recognition.yolo_router_comparison import aggregate, oracle, route_point


def m(tp, pred, gt=3):
    return dict(tp=tp, fp=pred-tp, fn=gt-tp, detected_words=pred, gt_words=gt,
                all_outputs=pred, ignored_outputs=0)


def test_empty_predictions_keep_false_negatives():
    result = aggregate([m(0,0)])
    assert result['precision'] is None
    assert result['f1'] == result['accuracy_tn0'] == 0
    assert result['fn'] == 3


def test_whole_image_replacement_and_t1_is_one_prediction():
    branches = {'a': dict(yolo=m(2,3), deepsolo=m(3,4), t1=m(1,1))}
    result = route_point(branches, {'a'}, {'a':'t1'})
    assert result['combined']['detected_words'] == 1
    assert result['combined']['fn'] == 2
    assert route_point(branches, set())['combined']['detected_words'] == 4
    with pytest.raises(ValueError):
        route_point(branches, {'unknown'})


@pytest.mark.parametrize('budget', [None,0,1,2,3,4])
def test_oracle_matches_exhaustive_micro_objective(budget):
    branches = {str(i):dict(yolo=a,deepsolo=b) for i,(a,b) in enumerate([
        (m(3,8),m(2,2)), (m(2,2),m(3,5)), (m(0,0),m(0,3)), (m(1,1),m(2,6))])}
    choices = [set(c) for k in range(5) if budget is None or k == budget
               for c in combinations(branches,k)]
    expected = max(route_point(branches,c)['combined']['accuracy_tn0'] for c in choices)
    actual = oracle(branches,budget)
    assert actual['combined']['accuracy_tn0'] == pytest.approx(expected)
    if budget is not None:
        assert actual['accepted_images'] == budget


def test_oracle_rejects_different_gt_denominators():
    with pytest.raises(ValueError):
        oracle({'a':dict(yolo=m(1,1,2),deepsolo=m(1,1,3))})
