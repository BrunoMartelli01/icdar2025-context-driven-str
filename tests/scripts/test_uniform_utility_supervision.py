from scripts.supervise_uniform_utility_oof import freeze_operating_points


def test_common_fraction_requires_all_validation_targets_before_test():
    config = dict(alternate_fallback_fractions=[.5, .75, 1.])
    plans = {d: dict(partitions=dict(test=list(range(n))), target=.8, budget=n//2)
        for d, n in [('small', 4), ('large', 8)]}
    gates = dict(small=dict(validation_recommended_fallback_images=2, policies={
        '2': dict(counts=dict(f1=.81)), '3': dict(counts=dict(f1=.82)), '4': dict(counts=dict(f1=.85))}),
        large=dict(validation_recommended_fallback_images=6, policies={
        '4': dict(counts=dict(f1=.79)), '6': dict(counts=dict(f1=.8)), '8': dict(counts=dict(f1=.84))}))
    points = freeze_operating_points(config, plans, gates)
    assert points['validation_common_fraction'] == 1.
    assert points['validation_common_quotas'] == dict(small=4, large=8)
    assert not points['test_used_for_selection']


def test_no_common_validation_success_is_explicit():
    config = dict(alternate_fallback_fractions=[.5, 1.])
    plans = dict(d=dict(partitions=dict(test=[1, 2]), target=.8, budget=1))
    gates = dict(d=dict(validation_recommended_fallback_images=None,
        policies={'1': dict(counts=dict(f1=.7)), '2': dict(counts=dict(f1=.8))}))
    points = freeze_operating_points(config, plans, gates)
    assert points['validation_common_fraction'] is None and points['validation_common_quotas'] is None
