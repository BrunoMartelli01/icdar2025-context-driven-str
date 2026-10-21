import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from caption.blip2_captioner import BLIP2ApproxConfig
from caption.paper_style_blip2 import generate_profile,InstructionCaptionConfig,LocalPaperBLIP2
from scoring.paper_image_router import (PaperRouterConfig,assemble_t3,paper_decision,
    localized_word_scores,caption_quality)
from scripts import run_paper_image_router_baseline as runner
from scripts.crop_clip_runtime import write_json


def config(**changes):
    return PaperRouterConfig(**{'alpha':.6,'beta':.4,'tau':.8,
        'primary_gate':'paper_selected','lexical_mode':'raw',**changes})


def row(identity,text,x=0,y=0):
    return {'instance_id':identity,'raw_text':text,'bbox':[x,y,x+4,y+3],
            'confidence':0,'ignored':True}


def test_all_word_assembly_same_as_clip_preserves_repeats_and_low_confidence():
    rows=[row(3,'BAR',10,10),row(2,'BAR',0,10),row(1,'TOP',0,0)]
    assembled=assemble_t3(rows)
    assert assembled['text']=='TOP BAR BAR'
    assert [w['instance_id'] for w in assembled['words']]==[1,2,3]
    assert assembled['eligible'] and assembled['all_predictions_included']
    assert assembled['deduplication'] is False
    assert assemble_t3(list(reversed(rows)))==assembled


def test_empty_crops_preserved_and_force_full_image_fallback():
    t3=assemble_t3([row(1,'TEXT'),row(2,'',10)])
    assert len(t3['words'])==2 and not t3['eligible']
    decision=paper_decision('WORD',t3['text'],'TEXT',.5,1,config())
    assert all(v is None for v in localized_word_scores(decision,t3).values())
    assert assemble_t3([])['eligible'] is False


@pytest.mark.parametrize('fault',['duplicate','bbox','nan','not_text'])
def test_invalid_crop_population_never_silently_dropped(fault):
    rows=[row(1,'A')]
    if fault=='duplicate':rows.append(row(1,'B'))
    if fault=='bbox':rows[0]['bbox']=[0,0,0,2]
    if fault=='nan':rows[0]['bbox'][0]=float('nan')
    if fault=='not_text':rows[0]['raw_text']=None
    with pytest.raises(ValueError):assemble_t3(rows)


def test_paper_candidate_selection_uses_semantic_score_then_paired_confidence():
    d=paper_decision('xxx','caption','caption',.9,.8,config())
    assert d['semantic_winner']=='T1' and d['selected_output']=='xxx'
    assert d['C']==d['C1'] and d['C3']>d['C1']
    assert d['needs_fallback'] and d['ground_truth_in_routing'] is False


def test_algorithm_strict_semantic_tie_prefers_t3_and_threshold_is_inclusive():
    d=paper_decision('different','caption','caption',2/3,2/3,config())
    assert d['semantic_winner']=='T3' and d['C']==pytest.approx(.8)
    assert d['needs_fallback'] is False
    below=paper_decision('different','caption','caption',.5,(2/3)-1e-6,config())
    assert below['needs_fallback']


def test_t1_winner_uses_its_own_confidence_instead_of_a_forced_fallback():
    d=paper_decision('caption','WORD','caption',.9,.8,config())
    assert d['needs_fallback'] is False and d['semantic_winner']=='T1'
    scores=localized_word_scores(d,assemble_t3([row(1,'WORD')]))
    assert scores['paper_selected']==d['C1'] and scores['joined_t3']==d['C3']


def test_t1_winner_must_pass_the_threshold_and_can_win_without_crop_outputs():
    from recognition.paper_image_router_evaluation import word_metrics
    cfg={'router':{'primary_gate':'paper_selected','tau':.8},'primary_caption_profile':'p',
        'report_thresholds':[.7,.8]}
    old={'calibration_images':80,'execution':{'seed':1},'coverage':.5,
        'comparison_budget':{'train':62}}
    high=paper_decision('caption','','caption',.9,.2,config())
    low=paper_decision('caption','','caption',.5,.2,config())
    assert localized_word_scores(high,assemble_t3([]))['paper_selected']==pytest.approx(.94)
    assert localized_word_scores(low,assemble_t3([]))['paper_selected']==pytest.approx(.7)
    assert not high['needs_fallback'] and low['needs_fallback']
    gt=[{'text':'caption'},{'text':'another'}]
    branches={'train':[{'sample_id':sid,'parseq':metric(0,0,2),
        't1':word_metrics([{'text':'caption','ignored':False}],gt),'deepsolo':metric(2,0,0)}
        for sid in ('high','low')]}
    samples={'train':[{'sample_id':'high'},{'sample_id':'low'}]}
    features={'p':{sid:{'selected_source':'T1','scores':localized_word_scores(d,assemble_t3([])),
        'caption_quality':{'words':60}} for sid,d in [('high',high),('low',low)]}}
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as directory:
        report=runner.evaluate(cfg,old,branches,samples,features,Path(directory),{},True)
    point=report['profiles']['p']['paper_selected']['splits']['train']['points']['paper_tau']
    assert point['accepted_sample_ids']==['high']
    assert point['accepted_sample_sources']=={'high':'T1'}
    assert point['t1_images']==1 and point['t3_images']==0 and point['deepsolo_images']==1
    assert point['words_retained_parseq']==1 and point['precision_retained_parseq']==1
    assert point['combined_model']['tp']==3 and point['combined_model']['fn']==1
    assert point['combined_model']['accuracy_tn0']==pytest.approx(.75)


@pytest.mark.parametrize('s1,s3',[(float('nan'),.5),(.5,float('inf')),(2,.5)])
def test_nonfinite_or_out_of_range_scores_fail(s1,s3):
    with pytest.raises(ValueError):paper_decision('a','b','c',s1,s3,config())


def test_raw_lexical_behavior_is_declared_and_no_implicit_casefold():
    d=paper_decision('HELLO','hello','hello',1,1,config())
    assert d['L1']==0 and d['L3']==1
    assert d['lexical_mode']=='raw'
    assert d['lexical_function']=='rapidfuzz.fuzz.ratio/100'


@pytest.mark.parametrize('change',[{'alpha':.5},{'tau':float('nan')},
    {'lexical_mode':'lowercase'},{'primary_gate':'max_c'}])
def test_scientific_configuration_rejects_unrecorded_interpretations(change):
    with pytest.raises(ValueError):config(**change)


def test_caption_quality_reports_repetition_length_not_visual_correctness():
    text='a sign on a sign on a sign on a sign'
    result=caption_quality(text,40,80)
    assert not result['within_target_words'] and result['unique_trigram_fraction']<.5
    assert not result['visual_correctness_automatically_verified']


def test_image_only_profile_preserves_all_raw_answers_and_config():
    captured=[]
    class Captioner:
        config=BLIP2ApproxConfig()
        def caption(self,rgb):
            assert rgb.shape==(4,4,3)
            captured.append(self.config.prompt)
            text='a sign' if len(captured)==1 else 'a street.'
            return SimpleNamespace(to_dict=lambda:{'raw_caption':text,'success':True})
    captioner=Captioner();old=captioner.config
    profile={'prompts':['Describe objects','Describe context'],'assembly':'paragraph_join','generation':{'max_new_tokens':5}}
    result=generate_profile(captioner,np.zeros((4,4,3),np.uint8),profile,{'min_words':40,'max_words':80})
    assert result['raw_caption']=='a sign. a street.' and len(result['answers'])==2
    assert captured==profile['prompts'] and captioner.config is old
    assert result['parseq_in_prompt'] is False and result['ground_truth_in_caption'] is False
    assert not result['quality']['within_target_words']


def test_failed_caption_is_propagated_without_dummy_and_config_restored():
    class Captioner:
        config=BLIP2ApproxConfig()
        def caption(self,rgb):
            return SimpleNamespace(to_dict=lambda:{'raw_caption':'','success':True})
    c=Captioner();old=c.config
    with pytest.raises(RuntimeError,match='empty/failed'):
        generate_profile(c,np.zeros((2,2,3),np.uint8),{'prompts':['describe'],
            'assembly':'single','generation':{}},{'min_words':40,'max_words':80})
    assert c.config is old


def test_baseline_cache_seals_prompt_device_source_and_payload(tmp_path):
    plan={'config_sha256':'cfg','code_sha256':{'file':'code'},'software':{'torch':'v'},
        'assets':{'blip2':{'revision':'a'*40}},'device':'cpu','caption_placement':'single_device',
        'config':{'caption_profiles':{'figure4_single':{'caption_model':'blip2'}}}}
    s={'sample_id':'image','image_sha256':'b'*64}
    contract=runner.cache_contract(plan,s,'captions','figure4_single')
    v={'contract':contract,'result':{'raw_caption':'a real scene'}}
    v['payload_sha256']=runner.payload_digest(v)
    path=runner.cache_path(tmp_path,contract);write_json(path,v)
    assert runner.read_cache(tmp_path,contract)==v
    assert runner.cache_path(tmp_path,{**contract,'device':'cuda:0'})!=path
    assert runner.cache_path(tmp_path,{**contract,'profile':'figure4_structured'})!=path
    v['result']['raw_caption']='edited';write_json(path,v)
    with pytest.raises(ValueError,match='payload changed'):runner.read_cache(tmp_path,contract)


def metric(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'tn':0,'detected_words':tp+fp,'gt_words':tp+fn,
        'all_outputs':tp+fp,'ignored_outputs':0,'invalid_envelopes':0,'matched_gt_indices':[]}


def test_whole_image_gate_evaluation_keeps_denominators_and_no_missing_fallback_substitution(tmp_path):
    cfg={'router':{'primary_gate':'paper_selected','tau':.8},'primary_caption_profile':'p',
         'report_thresholds':[0,.5,.8,1]}
    old={'calibration_images':80,'execution':{'seed':1},'coverage':.5,'comparison_budget':{'train':62,'test':149}}
    branches={'train':[{'sample_id':'a','parseq':metric(1,0,1),'deepsolo':None}],
        'test':[{'sample_id':'b','parseq':metric(1,1,1),'deepsolo':metric(2,0,0)}]}
    samples={'train':[{'sample_id':'a'}],'test':[{'sample_id':'b'}]}
    feature=lambda:{'selected_source':'T3','scores':{'paper_selected':.7,'joined_t3':.7},'caption_quality':{'words':60}}
    result=runner.evaluate(cfg,old,branches,samples,{'p':{'a':feature(),'b':feature()}},tmp_path,{},True)
    points=result['profiles']['p']['paper_selected']['splits']
    assert points['train']['points']['paper_tau']['combined_model'] is None
    assert points['test']['points']['paper_tau']['combined_model']['accuracy_tn0']==1
    assert points['train']['points']['same_budget_descriptive']['combined_model']['gt_words']==2
    assert points['test']['points']['same_budget_descriptive']['combined_model']['accuracy_tn0']==pytest.approx(1/3)
    assert result['selection_uses_test_labels'] is False
    assert (tmp_path/'comparison.csv').exists()


def frozen_fixture(tmp_path):
    from scripts.crop_clip_runtime import sha
    current={'software':{'torch':'v'},'assets':{'parseq':{'checkpoint_sha256':'weight'}},
        'config':{'execution':{'parseq_device':'cpu'}},'code_sha256':{'recognition/parseq.py':'code'},
        'source_inputs_sha256':{'image':'rgb'}}
    old={**current,'config_sha256':'config'}
    old_path=tmp_path/'old_plan.json';write_json(old_path,old)
    contract={'stage':'t1','sample_id':'a','image_sha256':'rgb','software':old['software'],
        'asset':old['assets']['parseq'],'code_sha256':old['code_sha256'],'config_sha256':'config'}
    value={'contract':contract,'result':{'raw_text':'word','input_role':'T1_full_image',
        'provenance':{'strict_state_dict_load':True,'checkpoint_sha256':'weight'}}}
    value['payload_sha256']=runner.payload_digest(value)
    shard=tmp_path/'shard.json';write_json(shard,value)
    manifest=tmp_path/'manifest.json';write_json(manifest,{'source_plan_path':str(old_path),
        'source_plan_sha256':sha(old_path),'samples':[{'sample_id':'a','path':str(shard),'sha256':sha(shard)}]})
    return {'frozen_t1_manifest_sha256':sha(manifest)},{'frozen_t1_manifest':manifest},current,{'train':[{'sample_id':'a','image_sha256':'rgb'}]},shard


def test_frozen_t1_reuse_preserves_outputs_and_checks_image_recipe_and_shard_hash(tmp_path):
    cfg,paths,current,observed,shard=frozen_fixture(tmp_path)
    assert runner.frozen_t1(cfg,paths,current,observed)['a']['result']['raw_text']=='word'
    changed={**current,'software':{'torch':'different'}}
    with pytest.raises(ValueError,match='recipe changed'):
        runner.frozen_t1(cfg,paths,changed,observed)
    shard.write_text('changed')
    with pytest.raises(ValueError,match='shard changed'):
        runner.frozen_t1(cfg,paths,current,observed)


def test_frozen_t1_reuse_refuses_missing_image_population(tmp_path):
    cfg,paths,current,observed,shard=frozen_fixture(tmp_path)
    observed['test']=[{'sample_id':'b','image_sha256':'other'}]
    with pytest.raises(ValueError,match='population changed'):
        runner.frozen_t1(cfg,paths,current,observed)


def test_caption_backend_is_explicit_and_instruction_ablation_uses_own_metadata(tmp_path,monkeypatch):
    calls=[]
    class Model:
        def to(self,device):assert str(device)=='cpu';return self
        def eval(self):return self
    def load(snapshot,**kwargs):
        assert kwargs['local_files_only'] and kwargs['output_loading_info']
        calls.append('instruct');return Model(),{}
    monkeypatch.setattr('transformers.InstructBlipForConditionalGeneration.from_pretrained',load)
    monkeypatch.setattr('transformers.AutoProcessor.from_pretrained',lambda *a,**kw:object())
    config=InstructionCaptionConfig(model_id='Salesforce/instructblip-flan-t5-xl',revision='b'*40)
    captioner=LocalPaperBLIP2(config,tmp_path,{'gpu':'6GiB','cpu':'48GiB'})
    captioner._load()
    assert calls==['instruct'] and captioner.config.model_id.startswith('Salesforce/instructblip')
    with pytest.raises(ValueError,match='pinned'):
        replace(config,revision='main')
    with pytest.raises(ValueError,match='quantization'):
        replace(config,quantization='4bit')


def test_context_overflow_preserves_all_t3_words_and_forces_fallback(tmp_path,monkeypatch):
    released=[]
    class Model:
        def __init__(self,*args,**kwargs):pass
        def tokenizer(self,text,**kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids':list(range(len(text.split())+2))}
    class Semantic:
        def __init__(self,*args,**kwargs):pass
        def compare_candidates(self,*args):
            pytest.fail('Overflowing transcripts must not be silently truncated/scored')
        def release(self):released.append(True)
    monkeypatch.setattr('sentence_transformers.SentenceTransformer',Model)
    monkeypatch.setattr(runner,'MPNetSemanticSimilarity',Semantic)
    monkeypatch.setattr(runner,'read_cache',lambda *a:{'result':{
        'raw_caption':'a shop with a sign','quality':{'words':6}}})
    cfg={'execution':{'mpnet_device':'cpu','mpnet_precision':'float32'},
        'mpnet':{'max_sequence_length':384},'router':vars(config())}
    plan={'config':{'caption_profiles':{'p':{'caption_model':'blip2'}}},
        'profiles':['p'],'assets':{'mpnet':{'snapshot':'unused'},'blip2':{}},
        'config_sha256':'cfg','code_sha256':{},'software':{},'device':'cpu',
        'caption_placement':'single_device','_frozen_t1':{'a':{'result':{'raw_text':'shop'}}}}
    sample={'sample_id':'a','image_sha256':'rgb','rows':[row(1,' '.join(['shop']*400))]}
    result=runner.score_samples(cfg,{'output_root':tmp_path},plan,{'train':[sample]})['p']['a']
    assert result['T3_assembly']['text']==sample['rows'][0]['raw_text']
    assert result['token_lengths']['T3']==402 and result['context_overflow_forces_fallback']
    assert result['paper_decision'] is None and set(result['scores'].values())=={None}
    assert released==[True] and result['ground_truth_in_routing'] is False


def full_comparison_fixture(tmp_path):
    cfg={'router':{'primary_gate':'paper_selected','tau':.8},'primary_caption_profile':'p',
         'report_thresholds':[0,.5,.8,1]}
    old={'calibration_images':1,'execution':{'seed':1},'coverage':.5,
         'comparison_budget':{'train':1,'test':1},'source_plan_sha256':'source',
         'branch_metrics_sha256':'branches'}
    samples={'train':[{'sample_id':'a'},{'sample_id':'b'}],'test':[{'sample_id':'c'}]}
    ids={s:[r['sample_id'] for r in rows] for s,rows in samples.items()}
    branches={s:[{'sample_id':r['sample_id'],'parseq':metric(1,1,1),
        'deepsolo':metric(2,0,0)} for r in rows] for s,rows in samples.items()}
    features={'p':{sid:{'selected_source':'T3','scores':{'paper_selected':.7,'joined_t3':.7},
        'caption_quality':{'words':60}} for rows in ids.values() for sid in rows}}
    clip_root=tmp_path/'clip';clip_root.mkdir()
    clip_plan=clip_root/'resolved_configuration.json'
    write_json(clip_plan,{'config':{'source_plan_sha256':'source'},'branch_metrics_sha256':'branches'})
    points={s:runner.source.evaluate(rows,{r['sample_id']:.7 for r in rows},.5,
        ids[s],sum(r['parseq']['gt_words'] for r in rows)) for s,rows in branches.items()}
    data={'status':'SUCCESS','claim_label':'approximate reproduction','plan_sha256':runner.sha(clip_plan),
        'split_ids':ids,'winner_train':'train_winner','original_router':points,
        'comparisons':{'train_winner':{s:{'train_threshold':p} for s,p in points.items()}}}
    result_path=clip_root/'results.json';write_json(result_path,data)
    return cfg,old,branches,samples,features,{'clip_results':{'test_model':str(result_path)}},data,result_path


def test_full_comparison_reports_frozen_clip_and_paper_on_the_same_words(tmp_path):
    import csv
    cfg,old,branches,samples,features,paths,data,result_path=full_comparison_fixture(tmp_path)
    report=runner.evaluate(cfg,old,branches,samples,features,tmp_path,paths,False)
    assert report['clip_comparison']['test_model']['family_train']=='train_winner'
    assert report['clip_comparison']['test_model']['results_sha256']==runner.sha(result_path)
    rows=list(csv.DictReader((tmp_path/'comparison.csv').open()))
    paper=next(r for r in rows if r['method']=='paper/p/paper_selected'
        and r['split']=='test' and r['policy']=='same_budget_descriptive')
    clip=next(r for r in rows if r['method']=='clip/test_model/train_winner' and r['split']=='test')
    assert paper['final_words']==clip['final_words']=='2'
    assert float(paper['final_accuracy_tn0'])==float(clip['final_accuracy_tn0'])==pytest.approx(1/3)
    assert report['selection_uses_test_labels'] is False


@pytest.mark.parametrize('fault',['split','branch'])
def test_full_comparison_refuses_clip_results_from_a_different_protocol(tmp_path,fault):
    cfg,old,branches,samples,features,paths,data,result_path=full_comparison_fixture(tmp_path)
    if fault=='split':data['split_ids']['test']=['different_image']
    else:old['branch_metrics_sha256']='different_branches'
    write_json(result_path,data)
    with pytest.raises(ValueError,match='source/split changed'):
        runner.evaluate(cfg,old,branches,samples,features,tmp_path,paths,False)
