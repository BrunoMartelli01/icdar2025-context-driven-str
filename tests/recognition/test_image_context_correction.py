"""Scientific boundaries for global OCR context, correction and rejection."""
import copy
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from recognition.image_context_correction import (accept_generated, adjacent_spans, aligned_target,
    apply_choices, confusions, context_windows, corrupt_text, filter_readings, listwise_loss,
    operation_features, pooled_language, select_subset, unambiguous_pairs)
from recognition.text_reconstruction import TextPrior


def block(text,box=None):
    return dict(text=text,bbox=box or [0,0,40,20],confidence=.9,minimum_token_probability=.8)


class Tokenizer:
    def encode(self,text,add_special_tokens=False):return list(map(ord,text))
    def decode(self,ids):return ''.join(map(chr,ids))


def test_all_other_ocr_tokens_are_covered_without_target_or_truncation():
    blocks=[block('abcdefghijk'),block('TARGET'),block('abcdefghijk'),block('xyz')]
    for mode in ('ordered','inventory'):
        windows=context_windows(blocks,[1],Tokenizer(),5,mode)
        assert [v for w in windows for v in w['token_ids']]==list(map(ord,'abcdefghijkabcdefghijkxyz'))
        assert all(1 not in w['source_indices'] for w in windows)
        assert all(len(w['token_ids'])<=5 for w in windows)
        assert sum(w['source_indices'].count(0) for w in windows)==3


def test_context_is_gt_blind_and_uses_original_readings_only():
    blocks=[block('THE'),block('TARGET'),block('ROAD')]
    original=context_windows(blocks,[1],Tokenizer(),30,'ordered')
    decorated=copy.deepcopy(blocks)
    for b in decorated:b.update(gt_text='FORBIDDEN',corrected_text='FORBIDDEN')
    assert context_windows(decorated,[1],Tokenizer(),30,'ordered')==original
    assert original[0]['left']==['THE'] and original[0]['right']==['ROAD']
    inventory=context_windows(blocks,[1],Tokenizer(),30,'inventory')
    assert inventory[0]['caption']=='THE ROAD' and inventory[0]['left']==[]


def test_empty_and_multi_parent_context():
    assert context_windows([block('A')],[0],Tokenizer(),5,'inventory')==[]
    result=context_windows([block('A'),block('B'),block('C')],[0,1],Tokenizer(),5,'inventory')
    assert result[0]['caption']=='C'
    with pytest.raises(ValueError):context_windows([],[],Tokenizer(),0,'inventory')


def test_pooled_language_counterfactual_and_weighting():
    result=pooled_language([-3.,-3.],[-1.,-5.],[.9,.1])
    assert result[2]==0 and result[3]==-2 and result[4]==2
    assert result[6]==pytest.approx(1.6) and result[7]==.5 and result[8]==2
    with pytest.raises(ValueError):pooled_language([-1.],[float('nan')],[1.])
    with pytest.raises(ValueError):pooled_language([],[],[])


def test_adjacency_excludes_vertical_words_and_non_neighbours():
    cfg=dict(minimum_vertical_overlap=.5,maximum_gap_heights=1.)
    b=[block('SAN',[0,0,30,20]),block('FRANCISCO',[34,0,90,20]),block('FAR',[0,50,30,70])]
    assert adjacent_spans(b,cfg)==[(0,),(1,),(2,),(0,1)]


ALIGN=dict(minimum_gt_coverage=.8,partial_overlap_limit=.1,maximum_words=3,maximum_center_heights=.75)


def test_clean_alignment_rejects_partial_neighbour_and_multiline_targets():
    words=[dict(text='NEW',bbox=[0,0,30,20]),dict(text='YORK',bbox=[35,0,70,20])]
    assert aligned_target([0,0,70,20],words,ALIGN)=='NEW YORK'
    assert aligned_target([0,0,50,20],words,ALIGN) is None
    words[1]['bbox']=[35,40,70,60]
    assert aligned_target([0,0,70,60],words,ALIGN) is None


def test_conflicting_sources_are_removed_without_choosing_a_target():
    rows=[dict(source='X',target='A'),dict(source='x',target='B'),dict(source='Y',target='Y')]
    kept,count=unambiguous_pairs(rows)
    assert kept==rows[2:] and count==1


def test_empirical_confusions_follow_clean_to_observed_direction():
    result=confusions([dict(source='RAKERY',target='BAKERY')])
    assert result==[dict(operation='replace',clean='b',noisy='r',count=1)]
    cfg=dict(identity_probability=0.,boundary_probability=0.,random_alphabet='abc')
    assert corrupt_text('BAKERY',random.Random(1),result,'empirical',cfg)==('RAKERY','character')


@pytest.mark.parametrize('text,expected',[('NEW YORK','NEW YORK'),('NEW WORK','NEW WORK'),('','NO'),('NEW 123','NO')])
def test_neural_validation_allows_bounded_letter_edits_and_protects_invalid(text,expected):
    cfg=dict(maximum_words=3,maximum_relative_edit=.25)
    result=accept_generated(['NEWYORK'],text,cfg)
    assert result==(None if expected=='NO' else expected)


def test_conflicting_operations_consume_every_original_at_most_once():
    b=[block('SAN'),block('FRANCISCO'),block('')]
    ops=[dict(parents=[0,1],text='SANFRANCISCO'),dict(parents=[1],text='FRAN CISCO')]
    output,chosen=apply_choices(b,ops,[.9,.8],.5)
    assert output==['SANFRANCISCO',''] and chosen==[0]
    assert apply_choices(b,ops,[.5,.5],.5)==(['SAN','FRANCISCO',''],[])
    with pytest.raises(ValueError):apply_choices(b,ops,[float('nan'),.1],.5)
    with pytest.raises(ValueError):apply_choices(b,[dict(parents=[0],text='')],[1.],0.)
    with pytest.raises(ValueError):apply_choices(b,[dict(parents=[0,0],text='X')],[1.],0.)


def test_candidate_kind_ablations_share_expanded_pool():
    ops=[dict(kind=s) for s in ('boundary','character','fragment')]
    assert select_subset(ops,'boundary')==[0]
    assert select_subset(ops,'character')==[0,1]
    assert select_subset(ops,'fragment')==[0,2]
    assert select_subset(ops,'expanded')==[0,1,2]


def test_listwise_keep_and_multiple_valid_alternatives_have_finite_gradients():
    for labels in ([0,2],[1,0],[1,1]):
        values=torch.tensor([.1,.2],requires_grad=True)
        loss=listwise_loss(values,labels);loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(values.grad).all()
    assert listwise_loss(torch.tensor([2.,-2.]),[1,0])<listwise_loss(torch.tensor([-2.,2.]),[1,0])


def test_filter_presence_preserves_uncertain_actual_annotation_overlap():
    b=[block('WRONG'),block('CORRECT'),block('')]
    assert filter_readings(b,[.1,.9,.1],.5,'exact')[0]==['CORRECT']
    assert filter_readings(b,[.1,.9,.1],.5,'presence',[.9,.9,.1],.5)[0]==['WRONG','CORRECT']
    assert filter_readings(b,[.1,.9,.1],0.,'exact')[0]==['WRONG','CORRECT','']
    with pytest.raises(ValueError):filter_readings(b,[.1,.9,.1],.5,'presence')


def test_duplicate_filter_only_removes_overlapping_lower_scored_duplicate():
    b=[block('WORD'),block('word'),block('WORD',[100,0,140,20])]
    assert filter_readings(b,[.9,.1,.1],.5,'duplicate')[0]==['WORD','WORD']


def test_features_are_finite_and_independent_of_annotations():
    b=[block('NEWYORK')];op=dict(parents=[0],text='NEW YORK',kind='boundary')
    f=operation_features(b,op,TextPrior(['new','york'],[]),{'new','york'})
    assert len(f)==24 and np.isfinite(f).all()


def test_namespace_and_gpu_error_are_actionable(tmp_path):
    from scripts.run_image_context_correction import checked_config,main
    with pytest.raises(ValueError):checked_config(Path('configs/strict/missing.json'),'approximate')
    args=['run','--mode','approximate','--config','configs/approximate/missing.json','--output',str(tmp_path/'failed')]
    with patch('sys.argv',args),patch('scripts.run_image_context_correction.checked_config',side_effect=torch.cuda.OutOfMemoryError('injected')):
        with pytest.raises(torch.cuda.OutOfMemoryError):main()
    assert json.loads((tmp_path/'failed/failure.json').read_text())['oom']


def test_calibration_reads_only_calibration_labels_and_keep_is_admissible():
    from scripts.run_image_context_correction import calibrate
    samples=[dict(sample_id='cal',blocks=[block('AHEAD')],pool=[dict(parents=[0],text='A HEAD',kind='boundary')])]
    c=dict(selector=dict(thresholds=[0.,1e6],damage_limits=[0,2]))
    with patch('scripts.run_contextual_boundary_selection.annotations',side_effect=lambda s:[dict(text='AHEAD')]):
        result=calibrate(samples,{'cal':[.5]},'boundary',c)
    assert result['choices']['correct_cap0']['threshold']==1e6


def test_generator_rejects_missing_train_targets_without_fallback():
    from scripts.run_image_context_correction import pairs_for
    cfg=dict(seed=1,paths=dict(text_run='/unused'),generator=dict(training_pairs=5,max_input_bytes=100,max_target_bytes=100,maximum_real_pairs=3),geometry={},alignment={})
    with patch('scripts.run_image_context_correction.read',return_value={'seed':0}),patch('scripts.run_image_context_correction.annotations',return_value=[]):
        with pytest.raises(ValueError,match='No clean TRAIN targets'):pairs_for([],cfg,'empirical',1,[],False)


def test_gpu_preflight_queue_does_not_retry_model_or_oom_errors(tmp_path):
    from scripts.run_image_context_correction import wait_gpu
    c=dict(runtime=dict(minimum_free_bytes=10,maximum_queue_seconds=1.,queue_log_seconds=1.,queue_poll_seconds=.1))
    with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.mem_get_info',return_value=(20,30)),\
         patch('scripts.run_image_context_correction.gpu_start',side_effect=torch.cuda.OutOfMemoryError('injected')):
        with pytest.raises(torch.cuda.OutOfMemoryError):wait_gpu(tmp_path,c,'test')


def test_gpu_queue_timeout_is_explicit(tmp_path):
    from scripts.run_image_context_correction import wait_gpu
    c=dict(runtime=dict(minimum_free_bytes=10,maximum_queue_seconds=0.,queue_log_seconds=1.,queue_poll_seconds=.1))
    with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.mem_get_info',return_value=(0,30)):
        with pytest.raises(RuntimeError,match='queue timeout'):wait_gpu(tmp_path,c,'test')


def test_full_selector_stage_freezes_before_any_eval_or_test_label_read(tmp_path):
    from scripts.run_image_context_correction import selectors
    def sample(sid):
        b=block('AHEAD');b['id']='original'
        return dict(sample_id=sid,blocks=[b],pool=[dict(parents=[0],text='A HEAD',kind='boundary')])
    data=dict(fit=[sample('fit-good'),sample('fit-bad'),sample('fit-neutral')],
        calibration=[sample('cal')],evaluation=[sample('FORBIDDEN-eval')],test=[sample('FORBIDDEN-test')])
    keys=('word','local','ordered','inventory','global_context','wrong_context')
    features={p:{s['sample_id']:dict(operations=[{k:[float(i)] for k in keys}],blocks=[[float(i)]]) for i,s in enumerate(ss)} for p,ss in data.items()}
    for phase,value in features.items():(tmp_path/f'features_{phase}.json').write_text(json.dumps(value))
    permitted={'fit-good':['A','HEAD'],'fit-bad':['AHEAD'],'fit-neutral':['OTHER'],'cal':['AHEAD']}
    calls=[]
    def labels(s):
        sid=s['sample_id'];assert sid in permitted,'Evaluation/test annotations used during selection';calls.append(sid)
        return [dict(text=t,bbox=[0,0,40,20]) for t in permitted[sid]]
    c=dict(seed=1,alignment=ALIGN,generator=dict(recipes=[]),selector=dict(iterations=1,max_leaf_nodes=2,min_samples_leaf=1,
        learning_rate=.1,l2_regularization=1.,listwise_hidden=4,listwise_learning_rate=.001,listwise_epochs=1,listwise_batch_groups=2,
        thresholds=[0.,1e6],damage_limits=[0,2]))
    with patch('scripts.run_image_context_correction.annotations',side_effect=labels),\
         patch('scripts.run_contextual_boundary_selection.annotations',side_effect=labels):
        selectors(tmp_path,c,data)
    frozen=json.loads((tmp_path/'selection_frozen.json').read_text())
    assert len(frozen['models'])==15 and not frozen['evaluation_test_used_for_selection']
    assert 'cal' in calls and all('FORBIDDEN' not in sid for sid in calls)


def test_independent_audit_preserves_empty_duplicate_and_punctuation_denominators():
    from scripts.audit_image_context_correction import independent_count
    count,matched=independent_count(['WORD','','Word!','WORD'],[dict(text='word'),dict(text='WORD!')])
    assert count==dict(correct=2,predictions=4,gt_words=2)
    assert dict(matched)=={'word':1,'word!':1}


def test_joint_order_distinguishes_recoverable_from_discarded_reading():
    from recognition.image_context_joint_policy import compose
    b=[block('NEWYORK'),block('WORD')];ops=[dict(parents=[0],text='NEW YORK')]
    args=(b,ops,[.8],.5,[.1,.9],[.1,.9],.5,.5,'exact')
    assert compose(*args,'filter_first')[0]==['WORD']
    assert compose(*args,'correct_first')[0]==['NEW YORK','WORD']


def test_joint_presence_filter_retains_uncertain_text_and_empty_keep():
    from recognition.image_context_joint_policy import compose
    b=[block('WRONG'),block('')]
    assert compose(b,[],[],.5,[.1,.1],[.9,.9],.5,.5,'presence','filter_first')[0]==['WRONG','']


def test_joint_merging_consumes_parents_once_and_preserves_unaffected_word():
    from recognition.image_context_joint_policy import compose
    b=[block('FRAN'),block('CISCO'),block('WORD')];ops=[dict(parents=[0,1],text='FRANCISCO')]
    out,trace=compose(b,ops,[.9],.5,[.1,.1,.9],[.1,.1,.9],.5,.5,'exact','correct_first')
    assert out==['FRANCISCO','WORD'] and trace['removed']==[]


def test_joint_unknown_order_is_an_explicit_error():
    from recognition.image_context_joint_policy import compose
    with pytest.raises(ValueError):compose([],[],[],0.,[],[],0.,0.,'exact','unknown')


@pytest.mark.parametrize('recipe',['random','empirical'])
def test_training_budget_replenishes_empty_synthetic_corruptions(recipe):
    from scripts.run_image_context_correction import pairs_for
    sample=dict(sample_id='FIT',blocks=[])
    c=dict(paths=dict(text_run='/unused'),generator=dict(training_pairs=3,max_input_bytes=100,max_target_bytes=100,maximum_real_pairs=0),geometry={},alignment={})
    with patch('scripts.run_image_context_correction.read',return_value=dict(seed=0)),\
         patch('scripts.run_image_context_correction.annotations',return_value=[dict(text='A')]),\
         patch('scripts.run_image_context_correction.corrupt_text',side_effect=[('', 'character'),('A','identity'),('A','identity'),('A','identity')]):
        pairs,info=pairs_for([sample],c,recipe,1,[],False)
    assert len(pairs)==3 and all(p['source'] for p in pairs) and info['pairs']==3


def test_training_budget_shortfall_is_an_explicit_failure():
    from scripts.run_image_context_correction import pairs_for
    sample=dict(sample_id='FIT',blocks=[])
    c=dict(paths=dict(text_run='/unused'),generator=dict(training_pairs=3,max_input_bytes=100,max_target_bytes=100,maximum_real_pairs=0),geometry={},alignment={})
    with patch('scripts.run_image_context_correction.read',return_value=dict(seed=0)),\
         patch('scripts.run_image_context_correction.annotations',return_value=[dict(text='A')]),\
         patch('scripts.run_image_context_correction.corrupt_text',return_value=('', 'character')):
        with pytest.raises(ValueError,match='exact declared'):pairs_for([sample],c,'random',1,[],False)
