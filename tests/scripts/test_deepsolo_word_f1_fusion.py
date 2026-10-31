import pytest

from scripts.evaluate_deepsolo_word_f1_fusion import counts, paired_bootstrap, select_policies


def policy(resolution):
    return dict(resolution=resolution, decoder='greedy', score_threshold=.4,
                recognition_threshold=0., same_text_nms_iou=None)


def prediction(text, bbox):
    return dict(text=text, bbox=bbox, ignored=False, score=.8, recognition_confidence=.9, instance_id=0)


def test_word_count_and_spatial_protocols_remain_distinct():
    p = [prediction('word!', [20, 0, 30, 10])]
    gt = [dict(text='WORD', bbox=[0, 0, 10, 10])]
    assert counts(p, gt, 'word', .5)['tp'] == 1
    assert counts(p, gt, 'localized', .5)['tp'] == 0


def test_selection_uses_only_explicit_train_population():
    train = [dict(ground_truth=[dict(text='word', bbox=[0, 0, 10, 10])],
                  native=[prediction('wrong', [0, 0, 10, 10])],
                  large=[prediction('WORD!', [0, 0, 10, 10])])]
    selected, grid = select_policies(train, [policy('native'), policy('large')], .5)
    assert all(p['resolution'] == 'large' for p in selected.values())
    assert len(grid) == 2
    # Ties prefer the first declared candidate, without a TEST tiebreaker.
    train[0]['native'] = train[0]['large']
    selected, _ = select_policies(train, [policy('native'), policy('large')], .5)
    assert all(p['resolution'] == 'native' for p in selected.values())


def test_bootstrap_is_paired_reproducible_and_keeps_empty_images():
    rows = [dict(tp=2, fp=1, fn=1), dict(tp=0, fp=0, fn=1)]
    a = paired_bootstrap(rows, rows, 50, 123)
    assert a == paired_bootstrap(rows, rows, 50, 123)
    assert a['delta_f1_95_percentile_interval'] == [0., 0.]
    with pytest.raises(ValueError):
        paired_bootstrap(rows, rows[:1], 50, 123)
