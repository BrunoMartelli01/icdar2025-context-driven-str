"""Staged BLIP2/MPNet whole-image baseline against frozen CLIP/PARSeq outputs.

APPROXIMATE only. Captions and T1 are image-only and cached before scoring.
No segmentation, crop recognition, DeepSolo model or old scientific file changes.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import numpy as np
from PIL import Image
import torch

from caption.blip2_captioner import BLIP2ApproxConfig
from caption.paper_style_blip2 import LocalPaperBLIP2,InstructionCaptionConfig,generate_profile
from recognition.clip_image_router import partition
from recognition.clip_router_prompt_ablation import coverage_threshold,budget_accept
from recognition.clip_router_threshold_sweep import curves
from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer
from recognition.paper_image_router_evaluation import (PROTOCOL,read_word_branches,
    chosen_branches,annotate_candidate_counts)
from scoring.paper_image_router import (PaperRouterConfig,assemble_t3,paper_decision,
    localized_word_scores)
from scoring.semantic import MPNetApproxConfig,MPNetSemanticSimilarity
from scripts.check_table5_assets import check_blip2,check_mpnet,check_parseq
from scripts.crop_clip_runtime import sha,write_json,_cuda_peak
from scripts.run_clip_high_recall_admission import seal,payload_digest
from scripts import run_clip_router_prompt_ablation as source

REPO=Path(__file__).resolve().parents[1]
CONFIG=REPO/'configs/approximate/paper_image_router_baseline_v1.json'
PATHS=REPO/'configs/approximate/paper_image_router_baseline_local_paths_v1.json'
CODE=('scripts/run_paper_image_router_baseline.py','scoring/paper_image_router.py',
      'recognition/paper_image_router_evaluation.py',
      'caption/paper_style_blip2.py','caption/blip2_captioner.py','recognition/parseq.py',
      'scoring/semantic.py','scoring/lexical.py','scripts/check_table5_assets.py',*source.CODE)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_protocol(args):
    for path in (args.config,args.paths):
        if path.resolve().parent!=REPO/'configs/approximate':
            raise ValueError('Baseline configuration must be in configs/approximate')
    cfg=json.loads(args.config.read_text());paths=json.loads(args.paths.read_text())
    if (args.mode!='approximate' or cfg['reproduction_mode']!='approximate'
            or cfg['claim_label']!='approximate reproduction' or cfg['table_5_claim'] is not False
            or cfg['ambiguity_record']!='PAPER-IMAGE-ROUTER-001' or paths['path_base']!='repository_root'):
        raise ValueError('Explicit isolated approximate baseline required')
    PaperRouterConfig(**cfg['router'])
    for name in args.profiles:
        if name not in cfg['caption_profiles']:raise ValueError('Unknown caption profile: '+name)
    for key in ('source_config','source_paths','hf_cache','parseq_source',
                'parseq_source_archive','parseq_checkpoint','output_root','frozen_t1_manifest'):
        paths[key]=(REPO/paths[key]).resolve()
    if args.output_root:paths['output_root']=args.output_root.resolve()
    if paths['output_root'].is_relative_to(REPO):raise ValueError('Experiment outputs must be outside Git')
    for key in ('source_config','source_paths'):
        if sha(paths[key])!=cfg[key+'_sha256']:raise ValueError('Frozen source config/profile changed: '+key)
    source_args=SimpleNamespace(mode='approximate',config=paths['source_config'],
                               paths=paths['source_paths'],output_root=None)
    old_cfg,old_paths,observed,branches,inputs=source.load_protocol(source_args)
    return cfg,paths,observed,branches,inputs,old_cfg


def assets(cfg,paths,profiles):
    hf=SimpleNamespace(hf_cache=paths['hf_cache'])
    blip2=check_blip2(hf);mpnet=check_mpnet(hf)
    parseq=check_parseq(SimpleNamespace(parseq_source=paths['parseq_source'],
        parseq_checkpoint=paths['parseq_checkpoint'],parseq_source_type='github_commit_archive',
        parseq_source_archive=paths['parseq_source_archive']))
    for name,actual in [('blip2',blip2),('mpnet',mpnet)]:
        for k in ('model_id','revision','weight_sha256'):
            if actual[k]!=cfg[name][k]:raise ValueError('Baseline pinned asset identity changed: '+name+'/'+k)
        directory=Path(actual['snapshot'])
        actual['configuration_file_sha256']={str(p.relative_to(directory)):sha(p)
            for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json','.txt','.model')}
    result={'blip2':blip2,'mpnet':mpnet,'parseq':parseq}
    if not any(cfg['caption_profiles'][p]['caption_model']=='instructblip' for p in profiles):return result
    instruction=cfg['instructblip'];directory=paths['hf_cache']/('models--'+instruction['model_id'].replace('/','--'))/'snapshots'/instruction['revision']
    hashes={name:sha(directory/name) for name in instruction['weight_sha256']}
    if hashes!=instruction['weight_sha256']:raise ValueError('Instruction-caption ablation weights changed')
    result['instructblip']={**instruction,'snapshot':str(directory),'configuration_file_sha256':{
            str(p.relative_to(directory)):sha(p) for p in directory.rglob('*')
            if p.is_file() and p.suffix in ('.json','.txt','.model')}}
    return result


def frozen_t1(cfg,paths,plan,observed):
    manifest_path=paths['frozen_t1_manifest']
    if sha(manifest_path)!=cfg['frozen_t1_manifest_sha256']:raise ValueError('Frozen T1 manifest changed')
    manifest=json.loads(manifest_path.read_text());old_path=Path(manifest['source_plan_path'])
    if sha(old_path)!=manifest['source_plan_sha256']:raise ValueError('Frozen T1 source plan changed')
    old=json.loads(old_path.read_text())
    if (old['software']!=plan['software'] or old['assets']['parseq']!=plan['assets']['parseq']
            or old['config']['execution']['parseq_device']!=plan['config']['execution']['parseq_device']
            or old['code_sha256']['recognition/parseq.py']!=plan['code_sha256']['recognition/parseq.py']
            or old['source_inputs_sha256']!=plan['source_inputs_sha256']):
        raise ValueError('Frozen T1 recognition/image/software recipe changed')
    expected={s['sample_id']:s for rows in observed.values() for s in rows}
    entries=manifest['samples']
    if (len(entries)!=len(expected) or len({e['sample_id'] for e in entries})!=len(expected)
            or {e['sample_id'] for e in entries}!=set(expected)):
        raise ValueError('Frozen T1 image population changed')
    result={}
    for entry in entries:
        path=Path(entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError('Frozen T1 shard changed')
        value=json.loads(path.read_text());contract=value['contract'];r=value['result']
        if (value['payload_sha256']!=payload_digest(value) or contract['stage']!='t1'
                or contract['sample_id']!=entry['sample_id']
                or contract['image_sha256']!=expected[entry['sample_id']]['image_sha256']
                or contract['code_sha256']!=old['code_sha256'] or contract['asset']!=old['assets']['parseq']
                or contract['config_sha256']!=old['config_sha256'] or contract['software']!=old['software']
                or r['input_role']!='T1_full_image' or not r['provenance']['strict_state_dict_load']
                or r['provenance']['checkpoint_sha256']!=old['assets']['parseq']['checkpoint_sha256']):
            raise ValueError('Frozen T1 shard identity/recipe changed')
        result[entry['sample_id']]=value
    return result


def software():
    return {p:importlib.metadata.version(p) for p in (
        'torch','transformers','sentence-transformers','rapidfuzz','Pillow','timm','accelerate')}


def cache_contract(plan,sample,stage,profile=None):
    component='parseq' if stage=='t1' else plan['config']['caption_profiles'][profile]['caption_model']
    return {'schema_version':1,'sample_id':sample['sample_id'],
        'image_sha256':sample['image_sha256'],'stage':stage,'profile':profile,
        'config_sha256':plan['config_sha256'],'code_sha256':plan['code_sha256'],
        'software':plan['software'],'asset':plan['assets'][component],
        'device':plan['config']['execution']['parseq_device'] if stage=='t1' else plan['device'],
        'caption_placement':None if stage=='t1' else plan['caption_placement'],
        'ground_truth_used':False,'parseq_predictions_used_for_caption':False}


def cache_path(root,contract):return root/'cache'/contract['stage']/digest(contract)/'result.json'


def read_cache(root,contract):
    path=cache_path(root,contract)
    if not path.exists():return None
    value=json.loads(path.read_text())
    if value['contract']!=contract or value['payload_sha256']!=payload_digest(value):
        raise ValueError('Baseline cache contract/payload changed: '+str(path))
    return value


def samples_for(args,observed):
    return {split:(rows[:args.limit] if args.limit else rows) for split,rows in observed.items()
            if args.split=='both' or split==args.split}


def read_rgb(sample):
    path=Path(sample['image_path'])
    if sha(path)!=sample['image_sha256']:raise ValueError('Source RGB image changed')
    with Image.open(path) as image:return np.ascontiguousarray(np.asarray(image.convert('RGB')))


def generate(args,cfg,paths,plan,samples,stage):
    root=paths['output_root'];model=None
    profiles=[None] if stage=='t1' else args.profiles
    try:
        for profile in profiles:
            if stage=='captions' and model is not None:
                model.release();del model;gc.collect();model=None
                if args.device.startswith('cuda'):torch.cuda.empty_cache()
            rows=[s for part in samples.values() for s in part]
            for i,sample in enumerate(rows,1):
                contract=cache_contract(plan,sample,stage,profile)
                if read_cache(root,contract) is not None:continue
                started=time.monotonic()
                component_device=cfg['execution']['parseq_device'] if stage=='t1' else args.device
                if component_device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(component_device)
                if model is None:
                    if stage=='t1':
                        model=ParseqApproxRecognizer(ParseqApproxConfig(paths['parseq_source'],
                            paths['parseq_checkpoint'],device=component_device,
                            runtime_source_type='github_commit_archive',
                            runtime_source_archive=paths['parseq_source_archive']))
                    else:
                        dtype=cfg['execution']['caption_cuda_dtype'] if args.device.startswith('cuda') else cfg['execution']['caption_cpu_dtype']
                        family=cfg['caption_profiles'][profile]['caption_model']
                        cls=InstructionCaptionConfig if family=='instructblip' else BLIP2ApproxConfig
                        model=LocalPaperBLIP2(cls(model_id=cfg[family]['model_id'],revision=cfg[family]['revision'],
                            device=args.device,dtype=dtype,
                            placement=args.caption_placement,quantization='none'),
                            plan['assets'][family]['snapshot'],cfg['execution']['caption_max_memory'])
                rgb=read_rgb(sample)
                if stage=='t1':
                    result=model.recognize(rgb,input_role='T1_full_image').to_dict()
                else:
                    result=generate_profile(model,rgb,cfg['caption_profiles'][profile],cfg['caption_target'])
                value={'contract':contract,'result':result,'runtime':{'device':component_device,
                    'seconds':time.monotonic()-started,'batch_size':1,
                    'source_image_shape':list(rgb.shape),'cuda_peak':_cuda_peak(component_device)}}
                value['payload_sha256']=payload_digest(value);seal(cache_path(root,contract),value)
                if stage=='captions' or i%25==0 or i==len(rows):print(json.dumps({'stage':stage,'profile':profile,'image':sample['sample_id'],
                    'progress':f'{i}/{len(rows)}','text':result['raw_text'] if stage=='t1' else result['raw_caption'],
                    'quality':result.get('quality')}),flush=True)
    finally:
        if model is not None:model.release()
        del model;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()


def score_samples(cfg,paths,plan,samples):
    # Validate every required shard before loading MPNet or accessing metrics.
    pending=[];cached={}
    for split,rows in samples.items():
        for s in rows:
            cached[s['sample_id'],'t1',None]=plan['_frozen_t1'][s['sample_id']]
            for stage,profile in [('captions',p) for p in plan['profiles']]:
                contract=cache_contract(plan,s,stage,profile);value=read_cache(paths['output_root'],contract)
                if value is None:pending.append((split,s['sample_id'],stage,profile))
                else:cached[s['sample_id'],stage,profile]=value
    if pending:raise FileNotFoundError(f'Missing{len(pending)} baseline shards. Run --stage all on the caption host. First:{pending[0]}')
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(plan['assets']['mpnet']['snapshot'],device=cfg['execution']['mpnet_device'],
                              local_files_only=True,model_kwargs={'torch_dtype':torch.float32})
    semantic=MPNetSemanticSimilarity(MPNetApproxConfig(device=cfg['execution']['mpnet_device'],
        precision=cfg['execution']['mpnet_precision'],max_sequence_length=cfg['mpnet']['max_sequence_length']),model=model)
    config=PaperRouterConfig(**cfg['router']);results={p:{} for p in plan['profiles']}
    try:
        for rows in samples.values():
            for s in rows:
                t3=assemble_t3(s['rows']);t1=cached[s['sample_id'],'t1',None]['result']['raw_text']
                for profile in plan['profiles']:
                    cap=cached[s['sample_id'],'captions',profile]['result'];text=cap['raw_caption']
                    lengths={name:len(model.tokenizer(value,add_special_tokens=True,truncation=False)['input_ids'])
                        for name,value in [('T1',t1),('T3',t3['text']),('T2',text)]}
                    overflow=any(n>cfg['mpnet']['max_sequence_length'] for n in lengths.values())
                    decision=None;scores={'joined_t3':None,'paper_selected':None}
                    if not overflow:
                        s1,s3=semantic.compare_candidates(t1,t3['text'],text)
                        decision=paper_decision(t1,t3['text'],text,s1,s3,config)
                        scores=localized_word_scores(decision,t3)
                    results[profile][s['sample_id']]={'sample_id':s['sample_id'],'T3_assembly':t3,
                        'T1_raw_text':t1,'selected_source':decision['semantic_winner'] if decision else None,
                        'paper_decision':decision,'scores':scores,'token_lengths':lengths,
                        'context_overflow_forces_fallback':overflow,'caption_quality':cap['quality'],
                        'caption_sha256':digest(cap),'ground_truth_in_routing':False,
                        'empty_crop_forces_fallback':not t3['eligible'] and (
                            decision is None or decision['semantic_winner']=='T3')}
    finally:
        semantic.release();del model;gc.collect()
    return results


def evaluate(cfg,old_cfg,branches,samples,features,root,paths,partial,geometric_branches=None):
    report={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'table_5_claim':False,'result_class':'smoke-test result' if partial else 'approximate reproduction',
        'primary_gate':cfg['router']['primary_gate'],'primary_caption_profile':
            cfg['primary_caption_profile'] if cfg['primary_caption_profile'] in features else None,
        'selection_uses_test_labels':False,'profiles':{},'clip_comparison':{},
        'metric_protocol':PROTOCOL,'t1_prediction_count':1,
        'geometry_diagnostics':{'profiles':{},'clip':{},'primary_t1_bbox':'not available; no fabricated localization'}}
    table=[]
    ids={s:[r['sample_id'] for r in rows] for s,rows in samples.items()}
    train_ids=ids.get('train',[])
    cal=partition(train_ids,old_cfg['calibration_images'],old_cfg['execution']['seed'])['calibration'] if not partial else train_ids
    for profile,fs in features.items():
        report['profiles'][profile]={}
        for gate in ('paper_selected','joined_t3'):
            scores={sid:f['scores'][gate] for sid,f in fs.items()}
            cut=coverage_threshold(scores,cal,old_cfg['coverage']) if cal else None
            by_split={}
            for split,split_ids in ids.items():
                rows=chosen_branches([r for r in branches[split] if r['sample_id'] in split_ids],fs,gate)
                gt=sum(r['parseq']['gt_words'] for r in rows)
                points={'paper_tau':source.evaluate(rows,scores,cfg['router']['tau'],split_ids,gt),
                    'train_50_percent_cut':source.evaluate(rows,scores,cut,split_ids,gt)}
                target=min(len(split_ids),old_cfg['comparison_budget'][split])
                keep=budget_accept(scores,split_ids,target)
                points['same_budget_descriptive']=source.evaluate(rows,{sid:float(sid in keep) for sid in split_ids},1.,split_ids,gt)
                for p in points.values():annotate_candidate_counts(p,fs,gate)
                replay=[{**r,'score':scores[r['sample_id']]} for r in rows]
                grid=cfg['report_thresholds']+([] if cut is None else [cut])
                curve=curves(replay,sorted(set(grid)),split_ids,gt)
                for ps in (curve['grid'],curve['full_curve']):
                    for p in ps:annotate_candidate_counts(p,fs,gate)
                by_split[split]={'points':points,'curves':curve,
                    'budget_requested':target,'caption_word_counts':[fs[sid]['caption_quality']['words'] for sid in split_ids]}
                for name,p in points.items():
                    m=p['combined_model']
                    table.append({'method':'paper/'+profile+'/'+gate,'split':split,'policy':name,
                        'threshold':p['threshold'] if name!='same_budget_descriptive' else None,
                        'parseq_images':p['parseq_images'],'parseq_words':p['words_retained_parseq'],
                        't1_images':p['t1_images'],'t3_images':p['t3_images'],
                        'parseq_precision':p['precision_retained_parseq'],'safe_images':p['safe_images'],
                        'complete_images':p['complete_images'],'final_words':m['detected_words'] if m else None,
                        'final_precision':m['accuracy_on_detected_words'] if m else None,
                        'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                        'missing_deepsolo':len(p['missing_deepsolo_sample_ids'])})
            report['profiles'][profile][gate]={'calibration_ids':cal,'threshold_train':cut,'splits':by_split}
        if geometric_branches is not None:
            report['geometry_diagnostics']['profiles'][profile]={}
            for split,split_ids in ids.items():
                rows=[r for r in geometric_branches[split] if r['sample_id'] in split_ids]
                scores={sid:fs[sid]['scores']['joined_t3'] for sid in split_ids}
                report['geometry_diagnostics']['profiles'][profile][split]=curves(
                    [{**r,'score':scores[r['sample_id']]} for r in rows],cfg['report_thresholds'],
                    split_ids,sum(r['parseq']['gt_words'] for r in rows))
    if not partial:
        def replay_clip(split,p):
            keep=p['accepted_sample_ids']
            if len(set(keep))!=len(keep) or not set(keep)<=set(ids[split]):
                raise ValueError('Frozen CLIP acceptance population changed')
            rows=branches[split];gt=sum(r['parseq']['gt_words'] for r in rows)
            q=source.evaluate(rows,{sid:float(sid in keep) for sid in ids[split]},1.,ids[split],gt)
            q['threshold']=p['threshold']
            return annotate_candidate_counts(q,{},'joined_t3')
        for model,path in paths['clip_results'].items():
            result_path=(REPO/path).resolve();data=json.loads(result_path.read_text())
            clip_plan=json.loads((result_path.parent/'resolved_configuration.json').read_text())
            if (sha(result_path.parent/'resolved_configuration.json')!=data['plan_sha256']
                    or clip_plan['config']['source_plan_sha256']!=old_cfg['source_plan_sha256']
                    or clip_plan['branch_metrics_sha256']!=old_cfg['branch_metrics_sha256']
                    or data['split_ids']!=ids):raise ValueError('CLIP comparison source/split changed')
            reference={}
            for split,p in data['original_router'].items():reference[split]=replay_clip(split,p)
            selected={s:{name:replay_clip(s,p) for name,p in ps.items()}
                for s,ps in data['comparisons'][data['winner_train']].items()}
            report['clip_comparison'][model]={'results_path':str(result_path),'results_sha256':sha(result_path),
                'original_router':reference,'family_train':data['winner_train'],
                'selected_family':selected,'metric_protocol':PROTOCOL,'decisions_reused_without_refitting':True}
            report['geometry_diagnostics']['clip'][model]={'original_router':data['original_router'],
                'selected_family':data['comparisons'][data['winner_train']]}
            for split,protocols in selected.items():
                for policy,p in protocols.items():
                    m=p['combined_model']
                    table.append({'method':'clip/'+model+'/'+data['winner_train'],'split':split,
                        'policy':policy,'threshold':p['threshold'] if policy!='same_budget_descriptive' else None,
                        'parseq_images':p['parseq_images'],'parseq_words':p['words_retained_parseq'],
                        't1_images':p['t1_images'],'t3_images':p['t3_images'],
                        'parseq_precision':p['precision_retained_parseq'],'safe_images':p['safe_images'],
                        'complete_images':p['complete_images'],'final_words':m['detected_words'] if m else None,
                        'final_precision':m['accuracy_on_detected_words'] if m else None,
                        'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                        'missing_deepsolo':len(p['missing_deepsolo_sample_ids'])})
        first=next(iter(report['clip_comparison'].values()))
        for split,p in first['original_router'].items():
            m=p['combined_model']
            table.append({'method':'clip/original_B16','split':split,'policy':'original_train_cut',
                'threshold':p['threshold'],'parseq_images':p['parseq_images'],
                't1_images':p['t1_images'],'t3_images':p['t3_images'],
                'parseq_words':p['words_retained_parseq'],'parseq_precision':p['precision_retained_parseq'],
                'safe_images':p['safe_images'],'complete_images':p['complete_images'],
                'final_words':m['detected_words'] if m else None,
                'final_precision':m['accuracy_on_detected_words'] if m else None,
                'final_accuracy_tn0':m['accuracy_tn0'] if m else None,
                'missing_deepsolo':len(p['missing_deepsolo_sample_ids'])})
    report['plan_sha256']=sha(root/'resolved_configuration.json') if (root/'resolved_configuration.json').exists() else None
    write_json(root/'results.json',report)
    with (root/'comparison.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Paper image router — approximate reproduction','',
        'Select T1 or T3 by semantic similarity; threshold that candidate\'s confidence. Accepted T1 is one word; T3 is all crop words.',
        'Main metric: image-level casefold-exact word matching for every engine/CLIP; bbox-IoU diagnostics are separate.',
        'Precision=TP/(TP+FP); TN0 accuracy=TP/(TP+FP+FN). Missing fallback leaves full metrics unavailable.','',
        '| Method | Split | Policy | Images kept | T1 / T3 images | PARSeq words | PARSeq precision % | Final words | Final TN0 % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda value:'—' if value is None else f'{100*value:.2f}'
    for r in table:lines.append(f"|{r['method']}|{r['split']}|{r['policy']}|{r['parseq_images']}|{r['t1_images']} / {r['t3_images']}|{r['parseq_words']}|{fmt(r['parseq_precision'])}|{r['final_words'] if r['final_words'] is not None else '—'}|{fmt(r['final_accuracy_tn0'])}|")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return report


def run(args):
    cfg,paths,observed,branches,inputs,old_cfg=load_protocol(args)
    torch.set_num_threads(cfg['execution']['cpu_threads']);torch.manual_seed(cfg['execution']['seed'])
    np.random.seed(cfg['execution']['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    qualified=assets(cfg,paths,args.profiles);samples=samples_for(args,observed)
    plan={'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':'smoke-test result' if args.limit or args.split!='both' else 'approximate reproduction',
        'config':cfg,'config_sha256':sha(args.config),'paths':{k:str(v) if isinstance(v,Path) else v for k,v in paths.items()},
        'paths_sha256':sha(args.paths),'source_inputs_sha256':inputs,'assets':qualified,'software':software(),
        'code_sha256':{n:sha(REPO/n) for n in CODE},'device':args.device,'caption_placement':args.caption_placement,
        'profiles':args.profiles,'stage':args.stage,'limit':args.limit,'split':args.split,
        'split_ids':{s:[r['sample_id'] for r in rows] for s,rows in samples.items()},
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_build':torch.version.cuda,'deterministic_algorithms':True,'tf32':False,'ground_truth_in_caption':False}
    root=paths['output_root']/digest(plan)[:12];seal(root/'resolved_configuration.json',plan)
    print(json.dumps({'status':'SOURCE_AND_ASSETS_VERIFIED','stage':args.stage,
        'images':{s:len(rows) for s,rows in samples.items()},'run':str(root)}),flush=True)
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','split_images':{s:len(r) for s,r in samples.items()},
            'cuda_available':torch.cuda.is_available(),'assets':qualified,'plan':str(root/'resolved_configuration.json')}),flush=True)
        return
    if args.stage in ('all','captions') and args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use CPU smoke or execute the complete command on the GPU host')
    try:
        plan['_frozen_t1']=frozen_t1(cfg,paths,plan,observed)
        if args.stage in ('all','captions'):generate(args,cfg,paths,plan,samples,'captions')
        if args.stage=='t1':
            print(json.dumps({'status':'FROZEN_T1_VERIFIED','images':len(plan['_frozen_t1'])}),flush=True)
        if args.stage in ('all','score'):
            features=score_samples(cfg,paths,plan,samples)
            write_json(root/'per_image_scores.json',features)
            source_paths=json.loads(paths['source_paths'].read_text())
            word_branches=read_word_branches({**cfg,'source_plan_sha256':old_cfg['source_plan_sha256']},
                Path(source_paths['source_run']),observed,branches,plan['_frozen_t1'])
            write_json(root/'word_branch_metrics.json',word_branches)
            report=evaluate(cfg,old_cfg,word_branches['splits'],samples,features,root,paths,
                bool(args.limit or args.split!='both'),geometric_branches=branches)
            write_json(paths['output_root']/'latest.json',{'results_path':str(root/'results.json'),'results_sha256':sha(root/'results.json')})
            print(json.dumps({'status':report['status'],'report':str(root/'comparison.md')}),flush=True)
    except Exception as error:
        write_json(root/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),
            'out_of_memory':isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)),
            'cuda_peak':_cuda_peak(args.device) if not args.device.startswith('cuda') or torch.cuda.is_available() else None})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=CONFIG);parser.add_argument('--paths',type=Path,default=PATHS)
    parser.add_argument('--profiles',nargs='+',default=['figure4_single','instructblip_scene'])
    parser.add_argument('--stage',choices=['all','captions','t1','score'],default='all')
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cpu')
    parser.add_argument('--caption-placement',choices=['single_device','cpu_offload'],default='single_device')
    parser.add_argument('--split',choices=['both','train','test'],default='both')
    parser.add_argument('--limit',type=int);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    if args.device=='cpu' and args.caption_placement!='single_device':parser.error('CPU requires single_device placement')
    run(args)


if __name__=='__main__':main()
