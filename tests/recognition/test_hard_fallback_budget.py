import copy

import pytest

from recognition.hard_fallback_budget import validate_policy, batch_quota, verify_route


CONTRACT = dict(reference_images=233, maximum_fallback_images=61, batch_rounding='floor')


def test_policy_refuses_excess_before_loading_inference_assets(tmp_path):
    import json
    from pipeline.uniform_word_pipeline import run
    cfg = tmp_path/'approximate/config.json'; cfg.parent.mkdir()
    cfg.write_text(json.dumps(dict(mode='approximate', hard_budgets=dict(icdar13=CONTRACT))))
    with pytest.raises(ValueError, match='exceeds the hard'):
        run(cfg, tmp_path/'missing.json', 'icdar13', 62, tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('n,k', [(233,61), (500,228), (300,150)])
def test_fraction_never_exceeds_hard_limit_on_other_batch_sizes(n,k):
    contract = dict(reference_images=n,maximum_fallback_images=k,batch_rounding='floor')
    assert batch_quota(contract,k,n,n)==k
    for images in (1,12,46,100,125,n,2*n+1):
        used = batch_quota(contract,k,n,images)
        assert used*n <= k*images


def test_mismatched_population_invalid_types_and_missing_budget_fail():
    for value in (62,True,-1):
        with pytest.raises(ValueError):validate_policy(dict(d=CONTRACT),'d',value)
    with pytest.raises(ValueError):validate_policy({},'d',61)
    with pytest.raises(ValueError):batch_quota(CONTRACT,61,234,233)
    with pytest.raises(ValueError):batch_quota(CONTRACT,61,233,0)
    with pytest.raises(ValueError):validate_policy(dict(d=dict(CONTRACT,batch_rounding='ceil')),'d',61)


def test_route_cannot_duplicate_omit_or_exceed_its_selected_image_count():
    # floor(61/233 * 12) = 3; the historical ceil rule would allow4.
    decision=dict(accepted_ids=list(range(9)),fallback_ids=[9,10,11],
        deepsolo_images=3,total_images=12,deepsolo_percentage=25.)
    verify_route(decision,list(range(12)),CONTRACT,61)
    for changed in (dict(fallback_ids=[8,9,10,11]),dict(accepted_ids=list(range(8))),
                    dict(fallback_ids=[9,9,11]),dict(deepsolo_images=4),dict(deepsolo_percentage=26.)):
        broken=copy.deepcopy(decision);broken.update(changed)
        with pytest.raises(ValueError):verify_route(broken,list(range(12)),CONTRACT,61)
