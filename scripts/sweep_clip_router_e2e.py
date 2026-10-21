"""Descriptive threshold sweep of the frozen B16 image router, separate splits.

Default: cached outputs only. --execute-missing-train runs the additional
original-image DeepSolo requests needed for all TRAIN thresholds on a GPU host.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback

import torch

from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.candidates import verify_host_source_identity
from recognition.clip_image_router import accepted
from recognition.clip_router_evaluation import deepsolo_predictions, evaluate_predictions, fixed_image_score
from recognition.clip_router_threshold_sweep import COUNT_KEYS, curves
from scripts.crop_clip_runtime import sha, write_json
from scripts.run_clip_high_recall_admission import seal
from scripts.run_clip_router_e2e import (cached, checked, key, read_approximate, resolve,
    save_shard, validate_deepsolo, validity)
from segmentation.data import load_segmentation_manifest

REPO=Path(__file__).resolve().parents[1]


def load(args):
    cfg,paths=read_approximate(args.config),read_approximate(args.paths)
    if (args.mode!='approximate' or cfg['reproduction_mode']!='approximate'
            or cfg['claim_label']!='approximate reproduction' or cfg['table_5_claim'] is not False
            or cfg['tn']!=0 or cfg['matching_iou']!=.5 or cfg['schema_version']!=1
            or not cfg['thresholds'] or cfg['thresholds']!=sorted(set(cfg['thresholds']))
            or any(type(t) not in (int,float) or not 0<=t<=1 for t in cfg['thresholds'])
            or paths['path_base']!='repository_root'):
        raise ValueError('Invalid isolated APPROXIMATE threshold protocol')
    paths={n:resolve(v) for n,v in paths.items() if n not in ('schema_version','path_base')}
    if args.output_root:paths['output_root']=args.output_root.resolve()
    if paths['output_root'].is_relative_to(REPO) or paths['output_root'].is_relative_to(paths['source_run']):
        raise ValueError('Use an isolated external output directory')
    source=paths['source_run'];source_plan=checked(source/'resolved_configuration.json',cfg['source_plan_sha256'])
    if source_plan['reproduction_mode']!='approximate' or source_plan['result_class']=='smoke-test result':
        raise ValueError('Full frozen approximate source required')
    for name,h in source_plan['code_sha256'].items():
        if sha(REPO/name)!=h:raise ValueError('Historical scientific code changed: '+name)
    reports={s:checked(source/s/'results.json',cfg['source_'+s+'_results_sha256']) for s in ('train','test')}
    audit=checked(source/'test_independent_audit.json',cfg['test_independent_audit_sha256'])
    if (audit['status']!='PASS' or audit['images']!=cfg['split_images']['test']
            or audit['plan_sha256']!=cfg['source_plan_sha256']
            or audit['result_sha256']!=cfg['source_test_results_sha256']):
        raise ValueError('Complete independent TEST audit required')
    ds={r['sample_id']:r for r in audit['deepsolo_only_per_image']}
    if len(ds)!=len(audit['deepsolo_only_per_image']) or set(ds)!=set(source_plan['split_ids']['test']):
        raise ValueError('Audited TEST DeepSolo population differs')
    reference_path=Path(source_plan['paths']['reference_run'])/'scores_by_image.json'
    ref=checked(reference_path,source_plan['config']['reference_scores_sha256'])['rank_reference']
    rows={};inputs={};contexts={}
    for split,report in reports.items():
        expected=source_plan['split_ids'][split]
        if (report['status']!='SUCCESS' or report['plan_sha256']!=cfg['source_plan_sha256']
                or [r['sample_id'] for r in report['per_image']]!=expected
                or len(expected)!=cfg['split_images'][split]):
            raise ValueError('Source split incomplete or changed')
        rows[split]=[]
        for r in report['per_image']:
            sid=r['sample_id'];route_path=source/split/'routes'/(key(sid)+'.json')
            route=json.loads(route_path.read_text());observed=route['observed']
            contract={'plan_sha256':cfg['source_plan_sha256'],'sample_id':sid,'image_sha256':observed['image_sha256']}
            route=cached(route_path,contract)
            if sha(Path(observed['image_path']))!=observed['image_sha256']:
                raise ValueError('Frozen original image changed')
            if fixed_image_score(observed['rows'],ref)!=route['score']:
                raise ValueError('Frozen CDF score replay differs')
            inputs[str(route_path)]=sha(route_path)
            preds=[{'bbox':w['bbox'],'text':w['raw_text'],'ignored':w['instance_id'] in route['ignored_ids']}
                   for w in observed['rows']]
            pm=evaluate_predictions(preds,r['ground_truth'],iou_threshold=cfg['matching_iou'])
            dm=None
            if split=='test':
                dm=evaluate_predictions(ds[sid]['predictions'],r['ground_truth'],iou_threshold=cfg['matching_iou'])
                if any(dm[n]!=ds[sid]['metrics'][n] for n in COUNT_KEYS):
                    raise ValueError('Independent TEST per-engine metric differs')
                raw=Path(source_plan['paths']['deepsolo_test_run'])/'detector'/(key(sid)+'.json')
                if sha(raw)!=source_plan['asset_sha256'][str(raw)]:raise ValueError('Audited raw TEST output changed')
                inputs[str(raw)]=sha(raw)
            elif r['action']=='deepsolo':
                raw=source/'train/deepsolo'/(key(sid)+'.json')
                record=cached(raw,contract)
                validate_deepsolo(record['response'],source_plan['fallback_config'],source_plan['fallback_config']['score_threshold'])
                if sha(raw)!=r['raw_deepsolo']['sha256']:raise ValueError('Source TRAIN raw DeepSolo changed')
                inputs[str(raw)]=sha(raw)
                dm=evaluate_predictions(r['predictions_all'],r['ground_truth'],iou_threshold=cfg['matching_iou'])
            chosen=pm if accepted(route['score'],source_plan['config']['threshold']) else dm
            if chosen is None or any(chosen[n]!=r['metrics'][n] for n in COUNT_KEYS):
                raise ValueError('Original full-pipeline point replay differs')
            rows[split].append({'sample_id':sid,'score':route['score'],'parseq':pm,'deepsolo':dm})
            if split=='train':contexts[sid]={'result':r,'observed':observed}
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
          'result_class':'approximate reproduction','table_5_claim':False,'config':cfg,
          'config_sha256':sha(args.config),'paths':{k:str(v) for k,v in paths.items()},
          'source_plan_sha256':cfg['source_plan_sha256'],'input_sha256':inputs,
          'code_sha256':{n:sha(REPO/n) for n in ('scripts/sweep_clip_router_e2e.py',
              'recognition/clip_router_threshold_sweep.py','scripts/clip_router_sweep_visuals.py')},
          'frozen_scientific_code_sha256':source_plan['code_sha256'],
          'software':{'torch':torch.__version__,'python':__import__('sys').version},
          'reference_fit_split':'TRAIN80','test_labels_used_for_descriptive_curves':True,
          'threshold_selected_on_test':False,'main_model_inference':False,'split_ids':source_plan['split_ids']}
    root=paths['output_root']/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,source_plan,rows,contexts,root,plan


def complete_train(rows,contexts,source_plan,root,plan_sha,source_plan_sha,execute):
    fallback=source_plan['fallback_config'];paths=source_plan['fallback_paths']
    adapter=None;samples=None;data_root=None;records={}
    for index,row in enumerate(rows,1):
        if row['deepsolo'] is not None:continue
        sid=row['sample_id'];context=contexts[sid];observed=context['observed']
        path=root/'additional_train_deepsolo'/(key(sid)+'.json')
        contract={'plan_sha256':plan_sha,'source_plan_sha256':source_plan_sha,
                  'sample_id':sid,'image_sha256':observed['image_sha256']}
        if path.exists():record=cached(path,contract)
        elif execute:
            if adapter is None:
                runtime_path=resolve(paths['deepsolo_runtime_manifest'])
                if sha(runtime_path)!=fallback['runtime_sha256']:raise ValueError('DeepSolo runtime changed')
                runtime=verify_host_source_identity(runtime_path,resolve(paths['deepsolo_source_root']))
                c=runtime['candidates'][fallback['candidate']]
                if c['checkpoint_sha256']!=fallback['checkpoint_sha256'] or sha(Path(c['checkpoint_path']))!=fallback['checkpoint_sha256']:
                    raise ValueError('DeepSolo checkpoint differs')
                if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable here: run --execute-missing-train on the GPU host')
                adapter=DeepSoloSubprocessAdapter(Path(paths['deepsolo_python']).absolute(),runtime_path,
                    host_source_root=resolve(paths['deepsolo_source_root']),repository_root=REPO)
            response=adapter.infer(image_path=Path(observed['image_path']),candidate_id=fallback['candidate'],
                inference_options={'device':fallback['device'],'score_threshold':fallback['score_threshold'],'lexicon_type':None},
                reduction_policy='reading_order_space_join').to_dict()
            validate_deepsolo(response,fallback,fallback['score_threshold'])
            record=save_shard(path,contract,{'response':response,'origin':'additional qualified original-image TRAIN inference'})
            print(f'[additional TRAIN DeepSolo] {index}/{len(rows)} {sid}',flush=True)
        else:continue
        response=validate_deepsolo(record['response'],fallback,fallback['score_threshold'])
        if samples is None:
            segroot=Path(source_plan['router_paths']['segmentation_run'])
            c=json.loads((segroot/'run_manifest.json').read_text())['run_contract']
            data_root=Path(c['runtime_inputs']['target_root']);manifest=data_root/c['config']['train_manifest']
            if sha(manifest)!=c['config']['train_manifest_sha256']:raise ValueError('TRAIN manifest changed')
            samples={s.sample_id:s for s in load_segmentation_manifest(manifest,require_checksums=True).samples}
        result=context['result'];valid=validity(samples[sid],data_root,result['image_wh'])
        preds=deepsolo_predictions(response,result['image_wh'],result['exif_orientation'],valid,fallback['score_threshold'])
        row['deepsolo']=evaluate_predictions(preds,result['ground_truth'],iou_threshold=source_plan['config']['matching_iou'])
        records[str(path)]=sha(path)
    return records


def write_csv(path,report):
    fields=['split','threshold','status','parseq_images','deepsolo_images','coverage_images',
        'words_retained_parseq','correct_words_retained_parseq','precision_retained_parseq','words_from_deepsolo',
        'detected_words','all_outputs','ignored_outputs','tp','fp','fn','accuracy_on_detected_words','accuracy_tn0','recall','word_f1','missing_deepsolo_images']
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for split,value in report['splits'].items():
            for p in value['grid']:
                row={k:p.get(k) for k in fields};row.update(split=split,missing_deepsolo_images=len(p['missing_deepsolo_sample_ids']))
                row.update({k:v for k,v in (p['combined_model'] or {}).items() if k in fields})
                writer.writerow(row)


def run(args):
    cfg,source_plan,rows,contexts,root,plan=load(args)
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','images':{n:len(r) for n,r in rows.items()},
            'thresholds':len(cfg['thresholds']),'additional_train_deepsolo_needed':sum(r['deepsolo'] is None for r in rows['train']),
            'output':str(root)}));return root
    seal(root/'resolved_configuration.json',plan);plan_sha=sha(root/'resolved_configuration.json')
    try:
        extra=complete_train(rows['train'],contexts,source_plan,root,plan_sha,cfg['source_plan_sha256'],args.execute_missing_train)
        branch={'plan_sha256':plan_sha,'additional_train_deepsolo_sha256':extra,'splits':rows}
        branch_digest=hashlib.sha256(json.dumps(branch,sort_keys=True,allow_nan=False).encode()).hexdigest()
        output=root/'analyses'/branch_digest[:12];seal(output/'branch_metrics.json',branch)
        result={'status':'SUCCESS_WITH_INCOMPLETE_TRAIN_POINTS' if any(r['deepsolo'] is None for r in rows['train']) else 'SUCCESS',
            'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'result_class':'approximate reproduction','table_5_claim':False,'plan_sha256':plan_sha,
            'branch_metrics_sha256':sha(output/'branch_metrics.json'),'threshold_selected_on_test':False,
            'test_labels_used_for_descriptive_curves':True,'splits_aggregated_together':False,
            'splits':{n:curves(r,cfg['thresholds'],plan['split_ids'][n],cfg['split_care_words'][n]) for n,r in rows.items()}}
        for n,split in result['splits'].items():
            match=next(p for p in split['grid'] if p['threshold']==source_plan['config']['threshold'])
            original=json.loads((Path(plan['paths']['source_run'])/n/'results.json').read_text())['combined_model']
            if match['combined_model']!=original:raise ValueError('Frozen reference aggregate replay differs: '+n)
        seal(output/'results.json',result);write_csv(output/'thresholds.csv',result)
        from scripts.clip_router_sweep_visuals import render
        render(output,result,source_plan['config']['threshold'])
        write_json(root/'latest.json',{'results_path':str(output/'results.json'),'results_sha256':sha(output/'results.json'),
            'branches_sha256':sha(output/'branch_metrics.json'),'status':result['status']})
        for split in ('train','test'):
            print(split.upper(),flush=True)
            for p in result['splits'][split]['grid']:
                c=p['combined_model'];print(json.dumps({'threshold':p['threshold'],'parseq_images':p['parseq_images'],
                    'words_retained_parseq':p['words_retained_parseq'],'final_words':c['detected_words'] if c else None,
                    'precision':c['accuracy_on_detected_words'] if c else None,'accuracy_tn0':c['accuracy_tn0'] if c else None,
                    'missing_deepsolo_images':len(p['missing_deepsolo_sample_ids'])}),flush=True)
        print(json.dumps({'results':str(output/'results.json'),'csv':str(output/'thresholds.csv'),
            'chart':str(output/'threshold_curve.svg'),'interactive':str(output/'explore.html'),'status':result['status']}),flush=True)
        return output
    except Exception as error:
        write_json(root/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
            'worker_response':error.response.to_dict() if getattr(error,'response',None) else None})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=REPO/'configs/approximate/clip_router_threshold_sweep_v1.json')
    p.add_argument('--paths',type=Path,default=REPO/'configs/approximate/clip_router_threshold_sweep_local_paths_v1.json')
    p.add_argument('--output-root',type=Path);p.add_argument('--execute-missing-train',action='store_true');p.add_argument('--check',action='store_true')
    run(p.parse_args())


if __name__=='__main__':main()
