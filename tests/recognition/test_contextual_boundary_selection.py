"""Focused scientific checks for boundary selection and leakage boundaries."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from recognition.contextual_boundary_selection import (candidates, class_utility,
    copied_segmentation, edit_label, image_fold, language_features, neighbours,
    projection_profile, select, visual_features)
from recognition.text_reconstruction import TextPrior


CFG=dict(minimum_characters=4,maximum_characters=32,maximum_words=3,minimum_piece_characters=2,maximum_lexical=6)
CTX=dict(neighbours_per_side=2,minimum_vertical_overlap=.5,maximum_gap_heights=8.,maximum_height_ratio=2.5)


def block(text,identity='a',box=None):
    return dict(id=identity,text=text,bbox=box or [0,0,80,20],confidence=.9,minimum_token_probability=.8)


@pytest.mark.parametrize('old,new,expected',[
    ('NEWYORK','new york','NEW YORK'),('caféBAR','café BAR','café BAR'),
    ('AHEAD','A HEAD','A HEAD'),('NEWYORK','NEW WORK',None),
    ('NEWYORK','NEWYORK',None),('Straße','Stra sse',None),('ABC','A B C D',None)])
def test_unicode_copy_boundaries(old,new,expected):
    assert copied_segmentation(old,new,3)==expected


def test_candidate_pool_gt_blind_and_deduplicated():
    prior=TextPrior(['new','york'],['new','york'])
    result=candidates('NEWYORK',['NEW YORK','new york','NEW WORK'],prior,CFG)
    assert [r['text'] for r in result]==['NEWYORK','NEW YORK']
    assert result[1]['origins']==['byt5','lexical']


@pytest.mark.parametrize('text',['','12.3','A','A-B'])
def test_protected_original_still_present(text):
    assert candidates(text,[],TextPrior([],[]),CFG)==[dict(text=text,origins=['keep'])]


def test_context_uses_geometry_and_original_only():
    blocks=[block('THE','left',[-30,0,-2,20]),block('AHEAD'),
            block('ROAD','below',[0,90,50,110]),block('KEEP','right',[85,0,130,20])]
    assert neighbours(blocks,1,CTX)==([0],[3])
    decorated=copy.deepcopy(blocks);decorated[0]['gt_text']='FORBIDDEN'
    assert neighbours(decorated,1,CTX)==([0],[3])


def test_selection_keeps_ties_empty_and_observation_count():
    s=dict(blocks=[block('NEWYORK'),block('','empty')],
           candidates=[[dict(text='NEWYORK'),dict(text='NEW YORK')],[dict(text='')]])
    assert select(s,[[.4],[]],.5)[0]==['NEWYORK','']
    assert select(s,[[.6],[]],.5)[0]==['NEW YORK','']
    with pytest.raises(ValueError):select(s,[[float('nan')],[]],.5)
    with pytest.raises(ValueError):select(s,[[],[]],.5)


def test_no_letter_edit_can_enter_selection():
    s=dict(blocks=[block('NEWYORK')],candidates=[[dict(text='NEWYORK'),dict(text='NEW WORK')]])
    with pytest.raises(ValueError):select(s,[[.9]],.5)


def test_recovery_damage_neutral_are_different_labels():
    b=[block('AHEAD')]
    assert edit_label(b,0,'A HEAD',[dict(text='AHEAD')])['label']==2
    assert edit_label(b,0,'A HEAD',[dict(text='A'),dict(text='HEAD')])['label']==1
    assert edit_label(b,0,'A HEAD',[dict(text='OTHER')])['label']==0


def test_duplicate_occurrence_is_not_a_new_recovery():
    b=[block('NEW'),block('NEWYORK','b')]
    label=edit_label(b,1,'NEW YORK',[dict(text='NEW')])
    assert label['label']==0 and label['recovered']==0


def test_class_utility_not_confidence_of_non_neutral():
    p=np.array([[.1,.2,.7],[.1,.8,.1]])
    assert np.allclose(class_utility(p,[0,1,2]),[-.5,.7])
    with pytest.raises(ValueError):class_utility(p[:,[0,2]],[0,2])


def test_visual_profile_constant_and_narrow_images_are_finite():
    for shape in [(1,1,3),(10,40,3)]:
        p=projection_profile(np.zeros(shape,np.uint8))
        v=visual_features(p,dict(text='NEW YORK'),.04)
        assert len(v)==12 and np.isfinite(v).all()


def test_language_score_failures_propagate():
    row=dict(status='scored',mean_log_probability=-1.,sum_log_probability=-2.,token_count=2)
    assert language_features(row,row)[2]==0
    with pytest.raises(ValueError):language_features(dict(status='input_token_limit'),row)


def test_folds_are_stable_and_image_bound():
    assert image_fold('sample',3,7)==image_fold('sample',3,7)
    assert len({image_fold(str(i),3,7) for i in range(100)})==3
    with pytest.raises(ValueError):image_fold('sample',1,7)


def test_namespace_rejects_strict():
    from scripts.run_contextual_boundary_selection import checked_config
    with pytest.raises(ValueError):checked_config(Path('configs/strict/missing.json'),'approximate')
    with pytest.raises(ValueError):checked_config(Path('configs/approximate/missing.json'),'strict')


def test_oom_receipt_and_reraise():
    import torch
    from scripts.run_contextual_boundary_selection import main
    with tempfile.TemporaryDirectory() as temp:
        output=Path(temp)/'run'
        args=['run','--mode','approximate','--config','configs/approximate/contextual_boundary_selection_v1.json','--output',str(output)]
        with patch('sys.argv',args),patch('scripts.run_contextual_boundary_selection.checked_config',side_effect=torch.cuda.OutOfMemoryError('injected')):
            with pytest.raises(torch.cuda.OutOfMemoryError):main()
        assert json.loads((output/'failure.json').read_text())['oom']
        assert not (output/'results.json').exists()


def test_selector_never_reads_evaluation_labels_before_freezing_choices():
    from scripts.run_contextual_boundary_selection import selectors, ARMS
    def sample(identity,text):
        return dict(sample_id=identity,blocks=[block(text)],
                    candidates=[[dict(text=text),dict(text='A HEAD')]])
    data=dict(fit=[sample('fit-benefit','AHEAD'),sample('fit-harm','AHEAD'),sample('fit-neutral','AHEAD')],
              calibration=[sample('cal','AHEAD')],evaluation=[sample('eval-forbidden','AHEAD')],
              test=[sample('test-forbidden','AHEAD')])
    features={phase:{s['sample_id']:[[{arm:[float(i)] for arm in ARMS}]] for i,s in enumerate(ss)}
              for phase,ss in data.items()}
    config=dict(selector=dict(seeds=[7],iterations=1,max_leaf_nodes=2,min_samples_leaf=1,
                learning_rate=.1,l2_regularization=1.,early_stopping=False,damage_limits=[0],thresholds=[.5,1.01]))
    labels={'fit-benefit':[dict(text='A'),dict(text='HEAD')], 'fit-harm':[dict(text='AHEAD')],
            'fit-neutral':[dict(text='OTHER')], 'cal':[dict(text='AHEAD')]}
    def permitted(s):
        assert s['sample_id'] in labels, 'EVAL/TEST annotation access during selection'
        return labels[s['sample_id']]
    with tempfile.TemporaryDirectory() as temp:
        with patch('scripts.run_contextual_boundary_selection.annotations',side_effect=permitted):
            scores,choices=selectors(Path(temp),config,data,features)
        assert (Path(temp)/'selection_frozen.json').is_file()
        assert not json.loads((Path(temp)/'selection_frozen.json').read_text())['evaluation_test_used_for_selection']


def test_calibration_objectives_are_distinct_and_damage_cap_is_enforced():
    from scripts.calibrate_contextual_boundary_sensitivity import choose
    curve=[dict(correct=10,f1=.8,damaged=0,threshold=1.01),
           dict(correct=12,f1=.7,damaged=0,threshold=.2),
           dict(correct=20,f1=.9,damaged=3,threshold=.1)]
    assert choose(curve,2,'correct_then_f1')['threshold']==.2
    assert choose(curve,2,'f1_then_correct')['threshold']==1.01


def test_exhaustive_thresholds_use_calibration_only():
    from scripts.calibrate_contextual_boundary_sensitivity import calibrate
    s=dict(sample_id='cal',blocks=[block('AHEAD')],candidates=[[dict(text='AHEAD'),dict(text='A HEAD')]])
    data=dict(calibration=[s],evaluation=object(),test=object())
    scores=dict(arm=dict(calibration=dict(cal=[[.025]])))
    def scoring(samples,predictions):
        assert samples is data['calibration']
        split=predictions['cal'][0]=='A HEAD'
        return dict(aggregate=dict(correct=2 if split else 0,f1=.8 if split else 0.,damaged=0,predictions=2 if split else 1,gt_words=2,recovered=2 if split else 0))
    with tempfile.TemporaryDirectory() as temp:
        with patch('scripts.calibrate_contextual_boundary_sensitivity.evaluate',side_effect=scoring):
            choices=calibrate(data,scores,['arm'],[0],Path(temp))
        assert choices['arm_cap0_correct_then_f1']['choice']['threshold']==.025
        assert (Path(temp)/'selection_frozen.json').is_file()
