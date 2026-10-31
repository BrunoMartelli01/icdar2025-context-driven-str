from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from recognition.selective_word_repair import (BoundaryHead, annotation_text_label,
    append_compatible, assess, boundary_layout, boundary_pairs, boundary_targets,
    conditional_features, conflicts, exact_matching, local_variants, recovery_label,
    render_boundaries, routed_candidates, triage)
from scripts.run_selective_word_repair_tests import checked_config, fit_binary, probability


def candidate(identity, text='word', box=(0,0,10,10), **kwargs):
    return dict(id=identity,text=text,bbox=list(box),parents=[],method='baseline',**kwargs)


def word(text='word', box=(0,0,10,10)):
    return dict(text=text,bbox=list(box))


def test_locked_additions_preserve_erroneous_baseline_and_allow_repeated_words_elsewhere():
    base=[candidate('a','wrong')];old=deepcopy(base)
    pool=base+[candidate('duplicate','word'),candidate('elsewhere','word',(20,0,30,10))]
    result=append_compatible(base,pool,{'a':0.,'duplicate':1.,'elsewhere':.8},.5,.6)
    assert base==old and result[:len(base)]==base
    assert [p['id'] for p in result]==['a','elsewhere']


def test_family_conflicts_preserve_sibling_splits():
    a=candidate('a');b=candidate('b',box=(20,0,30,10))
    a.update(method='split',parents=['parent']);b.update(method='split',parents=['parent'])
    assert not conflicts(a,b,.6)
    assert conflicts(a,candidate('parent'),.6)


def test_maximum_matching_reassigns_instead_of_greedy_losing_a_word():
    gt=[word(box=(0,0,10,10)),word(box=(5,0,15,10))]
    pp=[candidate('wide',box=(0,0,15,10)),candidate('left',box=(0,0,10,10))]
    assert exact_matching(pp,gt,.5)['correct']==2


def test_empty_and_duplicate_predictions_stay_in_denominator():
    pp=[candidate('one'),candidate('two'),candidate('empty','')]
    result=assess(pp,[word()],.5)
    assert result['spatial']['correct']==1
    assert result['spatial']['fp']==2
    assert result['lexical']['predictions']==3


def test_iou_boundary_and_spatial_vs_lexical_distinction():
    pp=[candidate('half',box=(0,0,5,10))]
    assert exact_matching(pp,[word()],.5)['correct']==1
    pp[0]['bbox']=[0,0,4,10]
    result=assess(pp,[word()],.5)
    assert result['spatial']['correct']==0 and result['lexical']['correct']==1


def test_utility_labels_separate_recovery_from_cleanup_only():
    base=[candidate('good'),candidate('bad','nonsense',(30,0,40,10))]
    assert recovery_label(base,base[:1],[word()],.5)==dict(recovered=0,damaged=0,beneficial=0,cleanup_only=1)
    result=recovery_label([],base[:1],[word()],.5)
    assert result['beneficial']==1 and result['recovered']==1
    assert recovery_label(base,[],[word()],.5)['damaged']==1


def test_annotation_proxy_marks_partial_text_even_when_not_a_complete_word():
    p=candidate('partial',box=(0,0,2,10))
    assert annotation_text_label(p,[word()],.3)==1
    assert exact_matching([p],[word()],.5)['correct']==0
    assert annotation_text_label(candidate('outside',box=(30,0,40,10)),[word()],.3)==0


@pytest.mark.parametrize('text_score,correct_score,expected',[(.1,.9,'keep'),(.9,.1,'reread'),(.1,.1,'reject')])
def test_triage_defers_real_but_uncertain_text(text_score,correct_score,expected):
    assert triage(candidate('x'),text_score,correct_score,dict(correct_threshold=.8,text_threshold=.2))==expected


def test_reread_budget_uses_scores_and_stable_ids():
    pp=[candidate('b'),candidate('a'),candidate('confident')]
    scores={'a':(.9,.1),'b':(.9,.1),'confident':(.8,.95)}
    result=routed_candidates(pp,scores,dict(text_threshold=.2,correct_threshold=.8),1)
    assert [p['id'] for p in result]==['a']


def test_unicode_byte_indices_and_copying_are_not_character_indices():
    text='é A猫'
    chars,flags,indices=boundary_layout(text)
    assert chars=='éA猫' and flags==[1,0] and indices==[1,2]
    assert render_boundaries(text,[.1,.9],.5,3)=='éA 猫'


def test_gap_targets_reject_missing_letters_and_accept_only_boundaries():
    assert boundary_targets('KrispyKreme','Krispy Kreme')==[0,0,0,0,0,1,0,0,0,0]
    assert boundary_targets('restaur','restaurant') is None
    assert boundary_targets('A HEAD','AHEAD')==[0,0,0,0]


def test_boundary_limits_keep_original_and_invalid_scores_fail():
    assert render_boundaries('abcd',[1,1,1],.5,2)=='abcd'
    with pytest.raises(ValueError):render_boundaries('éa',[float('nan')],.5,3)
    with pytest.raises(ValueError):render_boundaries('abc',[.5],.5,3)


def test_training_pairs_only_join_spatial_neighbors_and_keep_clean_character_alignment():
    options=dict(max_bytes=128,maximum_pairs=100,maximum_neighbors=1,gt_coverage=.85,
        neighbors=dict(minimum_vertical_overlap=.6,maximum_gap_heights=1.,maximum_height_ratio=2.))
    words=[word('ONE',(0,0,20,10)),word('TWO',(22,0,42,10)),word('FAR',(0,50,20,60))]
    sample=dict(sample_id='fit',text_blocks=[dict(text='O NETWO',bbox=[0,0,42,10])])
    pairs=boundary_pairs([sample],{'fit':words},options,random.Random(1))
    joins=[p for p in pairs if p['kind']=='local_missing_space']
    assert len(joins)==1 and joins[0]['target']=='ONE TWO'
    assert all(boundary_targets(p['source'],p['target'])==p['labels'] for p in pairs)
    assert any(p['kind']=='real_ocr' for p in pairs)


def test_conditional_features_do_not_confuse_same_word_at_another_location_with_duplicate():
    p=candidate('a');base=[candidate('b',box=(30,0,40,10))]
    x=conditional_features(p,base,[.5])
    assert x[3]==0. and x[4]==1.


def test_local_rgb_variants_are_clipped_and_split_by_the_predicted_map_only():
    p=candidate('p',box=(2,2,38,12))
    probability=np.ones((40,40));probability[:,20]=0
    options=dict(expansion_fraction=.5,neighbors=dict(minimum_vertical_overlap=.6,maximum_gap_heights=1.,maximum_height_ratio=2.),
        minimum_split_aspect=2.5,minimum_split_width=16,split_window=[.3,.7],maximum_variants=3)
    variants=local_variants(p,[p],{'p'},[40,40],probability,options)
    assert variants[0]['boxes']==[[0,0,40,17]]
    split=next(v for v in variants if v['kind']=='split')
    assert split['boxes']==[[2,2,20,12],[20,2,38,12]]
    assert p['bbox']==[2,2,38,12]


def test_boundary_head_strict_roundtrip_and_finite_gradients():
    torch.manual_seed(1);model=BoundaryHead(7,4);x=torch.randn(5,7)
    old=model(x);old.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    other=BoundaryHead(7,4);other.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(old,other(x),rtol=0,atol=0)


def test_single_class_fit_is_explicit_empirical_constant_not_fake_predictions():
    c=dict(seed=1,classifier=dict(max_iter=2,min_samples_leaf=2))
    model,meta=fit_binary([[0],[1],[2]],[0,0,0],c,'test')
    assert meta['status']=='SINGLE_CLASS_EMPIRICAL_CONSTANT'
    assert probability(model,[[8],[9]])==[0.,0.]


def test_strict_namespace_is_rejected_before_reading_artifacts(tmp_path):
    with pytest.raises(ValueError,match='APPROXIMATE'):
        checked_config(tmp_path/'strict.json','strict')


def test_oom_is_recorded_and_propagated_without_model_substitution(tmp_path,monkeypatch):
    import json
    import scripts.run_selective_word_repair_tests as runner
    output=tmp_path/'failed'
    def fail(*args):raise torch.cuda.OutOfMemoryError('controlled qualification OOM')
    monkeypatch.setattr(runner,'checked_config',fail)
    monkeypatch.setattr('sys.argv',['runner','--mode','approximate','--config',str(tmp_path/'config.json'),
                                  '--output-root',str(output),'--stage','prepare'])
    with pytest.raises(torch.cuda.OutOfMemoryError,match='controlled'):
        runner.main()
    receipt=json.loads((output/'FAILED.json').read_text())
    assert receipt['oom'] is True and receipt['status']=='FAILED'
