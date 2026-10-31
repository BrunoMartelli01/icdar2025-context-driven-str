"""Rescore saved word-slot experts without symbols and refit TRAIN-only gates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from recognition.word_slot_text import ALPHANUMERIC,normalize_word_text,comparison_label
from recognition.word_slot_evaluation import measure,aggregate,select
from recognition.word_slot_routing import FrozenWordSlotRouter
from segmentation.word_slot_data import read,write,sha,sample_key
from scripts.run_word_slot_methods import digest
from scripts.evaluate_word_slot_methods import summarize,raw_path
from scripts.analyze_word_slot_routers import analyze
from scripts.report_word_slot_methods import run as report_run,hungarian,router_result_label

REPO=Path(__file__).resolve().parents[1]
PHASES=('calibration','evaluation','test')


def verify_source(source,expected):
    for relative,value in expected.items():
        if sha(source/relative)!=value:raise ValueError('Historical source changed: '+relative)
    prediction_hashes=read(source/'analysis_source_sha256.json')
    for relative,value in prediction_hashes.items():
        if sha(source/relative)!=value:raise ValueError('Historical OCR shard changed: '+relative)
    return len(prediction_hashes)


def prepare(source,output,config):
    original=read(source/'protocol.json');protocol=deepcopy(original)
    protocol['source_training_config_digest']=original['config_digest']
    protocol['source_protocol_sha256']=sha(source/'protocol.json')
    protocol['config']['evaluation']['text_normalization']=ALPHANUMERIC
    protocol['config']['evaluation']['normalization']=comparison_label(ALPHANUMERIC)
    protocol['config_digest']=digest(protocol['config'])
    protocol['derived_analysis']={'kind':config['experiment_kind'],'source_run':str(source),
        'source_protocol_sha256':sha(source/'protocol.json'),'models_retrained':False,
        'new_model_inference':False,'raw_transcriptions_changed':False,
        'scope':'CPU-only metric reassessment and calibration refit; original training used its historical text policy'}
    write(output/'protocol.json',protocol)
    # These explicitly declared links are read-only inputs to analysis. Never
    # invoke inference/training in this derived directory.
    for name in ('predictions','branches','warmup','mae_preprocessing_v2_complete.json','teacher_cache_audit.json','inference_runtime.json'):
        (output/name).symlink_to(source/name,target_is_directory=(source/name).is_dir())
    if (source/'annotation_visual_review.json').exists():write(output/'annotation_visual_review.json',read(source/'annotation_visual_review.json'))
    inventory={};legacy_checks=0
    original_rows=read(source/'word_per_image.json')
    for phase in PHASES:
        records=protocol['index'][phase]
        inventory[phase]={'gt_words':sum(len(r['ground_truth']) for r in records),
          'gt_strings_with_removed_characters':sum(normalize_word_text(g['text'],ALPHANUMERIC)!=g['text'].casefold() for r in records for g in r['ground_truth']),
          'empty_normalized_gt':sum(not normalize_word_text(g['text'],ALPHANUMERIC) for r in records for g in r['ground_truth'])}
        for record in records:
            path=source/'deepsolo'/phase/(sample_key(record['sample_id'])+'.json');old=read(path)
            legacy=measure(old['predictions'],record['ground_truth'],protocol['config']['evaluation']['iou_threshold'])
            for key in ('tp','fp','fn','matched_gt_indices'):
                if legacy[key]!=old['metric'][key]:raise ValueError('Legacy metric regression: '+str(path))
            legacy_checks+=1
            item=deepcopy(old);item['metric']=measure(item['predictions'],record['ground_truth'],protocol['config']['evaluation']['iou_threshold'],ALPHANUMERIC)
            item['text_normalization']=ALPHANUMERIC;item['source_analysis_shard_sha256']=sha(path)
            write(output/'deepsolo'/phase/path.name,item)
    selections=read(source/'threshold_selection.json')
    for phase in PHASES:
        for arm,choice in selections.items():
            saved={r['sample_id']:r for r in original_rows[phase][arm]}
            for record in protocol['index'][phase]:
                raw=read(raw_path(source,arm,phase,record['sample_id']))['predictions']
                chosen=choice['chosen'];metric=measure(select(raw,chosen['presence'],chosen['confidence']),record['ground_truth'],protocol['config']['evaluation']['iou_threshold'])
                for key in ('tp','fp','fn','matched_gt_indices'):
                    if metric[key]!=saved[record['sample_id']][key]:raise ValueError('Legacy student metric regression')
                legacy_checks+=1
    write(output/'normalization_inventory.json',inventory)
    write(output/'legacy_metric_regression.json',{'status':'PASS','image_method_checks':legacy_checks,'default_policy':'casefold; punctuation retained','historical_outputs_modified':False})
    return protocol


def frozen_policy_replay(source,output,protocol,arms):
    selections=read(source/'threshold_selection.json');fits=read(source/'router_fit_audits.json')
    original_routes=read(source/'router_per_image.json');standalone={};routed={}
    for phase in PHASES:
        records=protocol['index'][phase];iou=protocol['config']['evaluation']['iou_threshold']
        standalone[phase]={'deepsolo':aggregate([read(output/'deepsolo'/phase/(sample_key(r['sample_id'])+'.json'))['metric'] for r in records])}
        for arm in arms:
            chosen=selections[arm]['chosen'];rows=[];by_kind={kind:[] for kind in ('image','image_fixed_half','word')}
            route_lookup={kind:{r['sample_id']:r for r in original_routes[phase][arm][kind]} for kind in by_kind} if phase!='calibration' else {}
            for record in records:
                sid=record['sample_id'];truth=record['ground_truth'];raw=read(raw_path(source,arm,phase,sid))['predictions']
                rows.append({'sample_id':sid,**measure(select(raw,chosen['presence'],chosen['confidence']),truth,iou,ALPHANUMERIC)})
                if phase=='calibration':continue
                deep=read(source/'deepsolo'/phase/(sample_key(sid)+'.json'))['predictions']
                word=read(source/'router_decisions'/arm/phase/(sample_key(sid)+'.json'))['word_routed_predictions']
                image_student=select(raw,*fits[arm]['image']['full_student_thresholds'])
                for kind in by_kind:
                    predictions=word if kind=='word' else image_student if route_lookup[kind][sid]['action']=='unet' else deep
                    by_kind[kind].append({'sample_id':sid,**measure(predictions,truth,iou,ALPHANUMERIC)})
            standalone[phase][arm]=aggregate(rows)
            if phase!='calibration':routed.setdefault(phase,{})[arm]={kind:aggregate(values) for kind,values in by_kind.items()}
    result={'text_normalization':ALPHANUMERIC,'scope':'historical thresholds, model selection and routing decisions frozen; only comparison normalization changes',
       'source_primary_student':read(source/'analysis_selection.json')['primary_best_learned_router_oof'],
       'standalone':standalone,'routed':routed,'selection_or_model_refit':False}
    write(output/'historical_fixed_policy_rescore.json',result)
    return result


def compare_routed_predictions(actual,expected):
    """Exact scientific output; only sigmoid probability roundoff is tolerated."""
    if len(actual)!=len(expected):raise ValueError('Serialized router changes output count')
    maximum=0.
    for a,b in zip(actual,expected):
        left={k:v for k,v in a.items() if k!='router_probability'};right={k:v for k,v in b.items() if k!='router_probability'}
        if left!=right:raise ValueError('Serialized router changes a reading, box, order or expert')
        if ('router_probability' in a)!=('router_probability' in b):raise ValueError('Missing serialized quality probability')
        if 'router_probability' in a:
            difference=abs(a['router_probability']-b['router_probability'])
            if not np.isfinite(difference) or difference>1e-12:raise ValueError('Serialized probability differs beyond float64 sigmoid roundoff tolerance')
            maximum=max(maximum,difference)
    return maximum


def serialized_router_audit(root,arms):
    protocol=read(root/'protocol.json');fits=read(root/'router_fit_audits.json');config=read(root/'router_execution.json')['config'];cases=0;maximum_probability_difference=0.
    for arm in arms:
        router=FrozenWordSlotRouter(fits[arm],config)
        for phase in ('evaluation','test'):
            for record in protocol['index'][phase]:
                sid=record['sample_id'];student=read(raw_path(root,arm,phase,sid))['predictions'];deep=read(root/'deepsolo'/phase/(sample_key(sid)+'.json'))['predictions']
                saved=read(root/'router_decisions'/arm/phase/(sample_key(sid)+'.json'))
                for kind in ('image','word'):
                    events=[]
                    def fallback():events.append('deep');return deep
                    result=router.route(student,record['image_wh'],kind=kind,release_unet=lambda:events.append('release'),run_deepsolo=fallback)
                    expected=saved['word_routed_predictions'] if kind=='word' else select(student,*fits[arm]['image']['full_student_thresholds']) if saved['image_uses_unet'] else deep
                    maximum_probability_difference=max(maximum_probability_difference,compare_routed_predictions(result['predictions'],expected))
                    expected_expert='word_merge' if kind=='word' else 'unet' if saved['image_uses_unet'] else 'deepsolo'
                    if result['expert']!=expected_expert:raise ValueError('Serialized router changes an image-level decision')
                    if kind=='word' and result['decisions']!=saved['word_router_decisions']:raise ValueError('Serialized router changes a word-level decision')
                    if events!=([] if expected_expert=='unet' else ['release','deep']):raise ValueError('Invalid router lifecycle')
                    cases+=1
    report={'status':'PASS','cases':cases,'scope':'serialized coefficients and actual callback ordering on saved RGB-model outputs; no new GPU inference',
            'text_normalization':ALPHANUMERIC,'readings_boxes_experts_order_and_decisions':'exact equality',
            'probability_absolute_tolerance':1e-12,'maximum_probability_absolute_difference':maximum_probability_difference,
            'tolerance_rationale':'sklearn expit versus NumPy logaddexp sigmoid float64 rounding; never tolerates changed routing decisions'}
    write(root/'serialized_router_audit.json',report)
    return report


def comparison_report(source,root,frozen):
    old=read(source/'word_results.json');new=read(root/'word_results.json');routes=read(root/'router_results.json');old_routes=read(source/'router_results.json')
    choices=read(root/'analysis_selection.json');old_primary=frozen['source_primary_student'];primary=choices['primary_best_learned_router_oof']
    arms=['deepsolo',*read(root/'threshold_selection.json')]
    frozen_word={**frozen['routed']['test'][old_primary]['word'],
                 'logical_deepsolo_calls':old_routes['test'][old_primary]['word']['logical_deepsolo_calls']}
    rows=[]
    for arm in arms:
        rows.append({'method':arm,'historical_punctuation_f1':old['test'][arm]['f1'],
          'same_historical_thresholds_symbol_free_f1':frozen['standalone']['test'][arm]['f1'],
          'refit_calibration_symbol_free':new['test'][arm],
          'refit_word_router':routes['test'].get(arm,{}).get('word')})
    write(root/'normalization_comparison.json',{'text_normalization':ALPHANUMERIC,'historical_primary_student':old_primary,'refit_primary_student':primary,'methods':rows,
       'historical_primary_word_router':{'punctuation':old_routes['test'][old_primary]['word'],'frozen_policy_symbol_free':frozen_word},
       'primary_refit_word_router':routes['test'][primary]['word'],'selection_uses_test':False})
    lines=['# Symbol-free word comparison','',
      '**Approximate reproduction / metric reassessment.** Requested by the user. No model retraining or new RGB inference.','',
      'Compare Unicode-casefolded letters and numbers only, deleting punctuation, symbols and whitespace from both sides. '+
      'Examples: `FISHERMAN’S` = `fishermans`, `B&B` = `bb`, `CO-OP` = `coop`. Accents and Unicode letters remain; there is no NFKC or accent folding. '+
      'Raw GT and predictions remain intact. Empty normalized strings cannot match; no care word or output is removed. Bbox IoU>=0.5 and one-to-one matching remain unchanged.','',
      '| Method | Previous F1 % with symbols | Same thresholds, symbols ignored % | CAL-refit, symbols ignored % | CAL-refit word F1 % (D images/total; %) |',
      '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        word='—' if row['refit_word_router'] is None else router_result_label(row['refit_word_router'])
        lines.append(f"| {row['method']} | {row['historical_punctuation_f1']*100:.3f} | {row['same_historical_thresholds_symbol_free_f1']*100:.3f} | {row['refit_calibration_symbol_free']['f1']*100:.3f} | {word} |")
    lines += ['',f"Historical primary student: **{old_primary}**. With its decisions frozen, word-router F1 and DeepSolo usage are {router_result_label(frozen_word)}. "+
      f"The new OOF-calibration-selected student is **{primary}**; its refitted word router F1 and DeepSolo usage are {router_result_label(routes['test'][primary]['word'])} "+
      f"versus DeepSolo {new['test']['deepsolo']['f1']*100:.3f}%. TEST does not select these thresholds or students.",'',
      'All15 student/control methods use the same normalized rule, including quality labels, textual agreement, replacement equivalence and final matching. '+
      'Other router features retain raw text and native confidences. FIT-supervised expert weights and original OCR training are unchanged; this is not retraining with a new charset.','',
      'The historical repeated TEST exposure, prior exposure to calibration, single seed, known duplicate RGB pairs and alternate bbox protocol remain limitations. '+
      'Normalization is user-directed; it does not resolve the paper metric ambiguity or establish official polygon evaluation.','',
      '[Complete normalized report](report.md), [machine-readable comparison](normalization_comparison.json), '+
      '[independent matching audit](independent_audit.json) and [serialized-router audit](serialized_router_audit.json).','']
    (root/'normalization_comparison.md').write_text('\n'.join(lines))


def run(config_path,output):
    config=read(config_path)
    if REPO/'configs/approximate' not in config_path.resolve().parents or config['reproduction_mode']!='approximate' or config['text_normalization']!=ALPHANUMERIC or config.get('experiment_kind')!='word_slot_alphanumeric_reassessment_v1':
        raise ValueError('Explicit APPROXIMATE alphanumeric configuration required')
    if output.exists():raise FileExistsError('Preserve reassessments; use a fresh output directory')
    output.mkdir(parents=True);source=Path(config['source_run']).resolve();start=time.perf_counter()
    write(output/'resolved_reassessment.json',{'config':config,'config_path':str(config_path.resolve()),'config_sha256':sha(config_path),
         'runtime':{'python':platform.python_version(),'device':'cpu','models_executed':[],'numpy':np.__version__,
                    'blas_threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}},
         'source_provenance_sha256':sha(Path(config['source_provenance']))})
    try:
        count=verify_source(source,config['source_artifact_sha256']);print('SOURCE_VERIFIED',count,flush=True)
        protocol=prepare(source,output,config);arms=['pixel_control','pixel_pil',*protocol['config']['arms']]
        frozen=frozen_policy_replay(source,output,protocol,arms);print('FROZEN_POLICY_RESCORED',flush=True)
        summarize(output,protocol['config'],arms,PHASES);print('CALIBRATION_THRESHOLDS_REFIT',flush=True)
        router_config=read(REPO/config['router_config']);router_config['text_normalization']=ALPHANUMERIC
        write(output/'resolved_router_config.json',router_config);analyze(output,router_config,arms,('evaluation','test'))
        report_run(output,read(REPO/config['analysis_config']),read(REPO/config['controls_config']))
        serialized_router_audit(output,arms);comparison_report(source,output,frozen)
        verify_source(source,config['source_artifact_sha256'])
        write(output/'complete.json',{'status':'SUCCESS','claim_label':'approximate reproduction','scope':'symbol-free reassessment; no new model inference or training',
          'text_normalization':ALPHANUMERIC,'source_run':str(source),'source_unchanged':True,'source_prediction_shards_verified':count,
          'seconds':time.perf_counter()-start,'selection_uses_test':False,'config_sha256':sha(config_path),
          'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'recognition/word_slot_text.py',REPO/'recognition/word_slot_evaluation.py',REPO/'recognition/word_slot_routing.py',REPO/'scripts/evaluate_word_slot_methods.py',REPO/'scripts/analyze_word_slot_routers.py',REPO/'scripts/report_word_slot_methods.py')}})
        print('ALPHANUMERIC_REASSESSMENT_COMPLETE',flush=True)
    except Exception as error:
        write(output/'failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.config.resolve(),args.output.resolve())
