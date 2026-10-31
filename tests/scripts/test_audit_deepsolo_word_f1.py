from scripts.audit_deepsolo_word_f1 import independent_counts


def test_hungarian_recount_keeps_duplicates_invalid_empty_and_symbol_denominators():
    g = [dict(text='word', bbox=[0, 0, 10, 10]), dict(text='!', bbox=[20, 0, 30, 10])]
    p = [dict(text='WORD!', ignored=False, bbox=[0, 0, 10, 10]),
         dict(text='word', ignored=False, bbox=[20, 0, 30, 10]),
         dict(text='', ignored=False, bbox=None, invalid_envelope={}),
         dict(text='word', ignored=True, bbox=[0, 0, 10, 10])]
    assert independent_counts(p, g, False, .5) == dict(tp=1, fp=2, fn=1)
    assert independent_counts(p, g, True, .5) == dict(tp=1, fp=2, fn=1)
    p[0]['bbox'] = [100, 100, 110, 110]
    assert independent_counts(p, g, True, .5) == dict(tp=0, fp=3, fn=2)


def test_empty_image_is_an_evaluated_failure_when_gt_is_present():
    assert independent_counts([], [dict(text='word')], False, .5) == dict(tp=0, fp=0, fn=1)
