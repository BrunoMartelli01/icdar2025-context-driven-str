import copy

import pytest

from scripts.audit_clip_specialized_routers import exact_budget_curve, validate_completed_branches


def original_and_completed():
    metric = {'tp':1,'fp':0,'fn':0,'tn':0,'detected_words':1,'gt_words':1,
              'all_outputs':1,'ignored_outputs':0,'invalid_envelopes':0}
    row = {'sample_id':'train:1','parseq':metric,'deepsolo':None,
           'ground_truth':[{'text':'TRAIN'}]}
    test = {**copy.deepcopy(row),'sample_id':'test:1','deepsolo':copy.deepcopy(metric)}
    original = {'splits':{'train':[row],'test':[test]}}
    completed = copy.deepcopy(original)
    completed['splits']['train'][0]['deepsolo'] = copy.deepcopy(metric)
    return original, completed


def test_completion_can_only_fill_missing_fallbacks():
    original,completed = original_and_completed()
    frozen = copy.deepcopy(original)
    validate_completed_branches(original,completed)
    assert original == frozen


@pytest.mark.parametrize('split',['train','test'])
def test_completion_rejects_changed_gt_or_recognition(split):
    original,completed = original_and_completed()
    completed['splits'][split][0]['ground_truth'][0]['text'] = 'CHANGED'
    with pytest.raises(ValueError,match='non-DeepSolo'):
        validate_completed_branches(original,completed)


def test_completion_rejects_replacing_existing_fallback():
    original,completed = original_and_completed()
    completed['splits']['test'][0]['deepsolo'].update(tp=0,fp=1,fn=1)
    with pytest.raises(ValueError,match='existing DeepSolo'):
        validate_completed_branches(original,completed)


def test_completion_rejects_unavailable_or_duplicate_images():
    original,completed = original_and_completed()
    completed['splits']['train'][0]['deepsolo'] = None
    with pytest.raises(ValueError,match='missing DeepSolo'):
        validate_completed_branches(original,completed)
    completed['splits']['train'] *= 2
    with pytest.raises(ValueError,match='duplicate'):
        validate_completed_branches(original,completed)


def test_descriptive_budget_ties_are_not_reported_as_scalar_thresholds():
    _,completed = original_and_completed()
    template = completed['splits']['train'][0]
    rows = [{**copy.deepcopy(template),'sample_id':sid} for sid in ['b','a','empty']]
    curve = exact_budget_curve(rows,{'b':.5,'a':.5,'empty':None},['b','a','empty'],3)
    assert [p['parseq_images'] for p in curve] == [0,1,2]
    assert curve[1]['accepted_sample_ids'] == ['a']
    assert curve[1]['rank_cut_score'] == .5
    assert curve[1]['inclusive_scalar_accepts'] == 2
    assert all(p['threshold_is_binary_budget_encoding'] for p in curve)
    assert all(p['combined_model']['gt_words'] == 3 for p in curve)
