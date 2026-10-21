"""APPROXIMATE paired YOLO/PARSeq: fixed CLIP versus section3.5 router.

Cache TRAIN detector/OCR, freeze TRAIN-only cuts, THEN score TEST. Reuse
image-identified captions, T1 and complete qualified DeepSolo branches.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch

from recognition.clip_image_router import partition
from recognition.clip_router_evaluation import fixed_image_score, native_mask, evaluate_predictions
from recognition.clip_router_prompt_ablation import coverage_threshold, budget_accept
from recognition.group_word_selection import normalize, reading_confidence
from recognition.paper_image_router_evaluation import word_metrics, PROTOCOL
from recognition.parseq import (ParseqApproxConfig, ParseqApproxRecognizer,
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    verify_parseq_runtime_source)
from recognition.yolo_router_comparison import aggregate, oracle, paired_bootstrap, route_point
from recognition.yolo_word_refinement import joint_score, nms
from scoring.paper_image_router import PaperRouterConfig, assemble_t3, paper_decision, localized_word_scores
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scripts.crop_clip_runtime import sha, array_sha, _cuda_peak, write_json
from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots, validate_checkpoint
from scripts.run_clip_high_recall_admission import payload_digest, seal
from scripts.run_clip_image_router_word_advantage import WordAdvantageScorer
from scripts.run_clip_router_prompt_ablation import model_identity

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO/'configs/approximate/yolo_router_comparison_v1.json'
PATHS = REPO/'configs/approximate/yolo_router_comparison_local_paths_v1.json'
CODE = ('scripts/run_yolo_router_comparison.py','recognition/yolo_router_comparison.py',
    'recognition/clip_router_evaluation.py','recognition/clip_router_prompt_ablation.py',
    'recognition/paper_image_router_evaluation.py','recognition/clip_crop_admission.py',
    'recognition/group_word_selection.py','recognition/yolo_word_refinement.py',
    'recognition/parseq.py','scoring/paper_image_router.py','scoring/semantic.py','scoring/lexical.py',
    'scripts/run_clip_image_router_word_advantage.py','scripts/crop_clip_runtime.py',
    'scripts/evaluate_totaltext_yolo_word_slots.py','recognition/group_word_selection_runtime.py',
    'recognition/clip_word_verifier.py','scripts/run_clip_router_e2e.py')


def read(path):
    return json.loads(Path(path).read_text())


def checked(path, expected):
    if sha(Path(path)) != expected:
        raise ValueError('Frozen source changed: '+str(path))
    return read(path)


def key(sid):
    return sid.replace(':','__')


def load_plan(args):
    for path in (args.config,args.paths):
        if path.resolve().parent != REPO/'configs/approximate':
            raise ValueError('Isolated approximate namespace required')
    cfg, paths = read(args.config), read(args.paths)
    if (cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate'
            or cfg['table_5_claim'] is not False or cfg['primary_router'] != 'clip_native_word_advantage'
            or cfg['metric'] != PROTOCOL or cfg['selection']['joint_threshold'] != .6
            or not 0 < cfg['selection']['nms_iou'] < 1 or args.limit not in (None,1)):
        raise ValueError('Unsupported explicit comparison protocol')
    output = (args.output or Path(paths['output'])).absolute()
    protected = [REPO,*(Path(paths[k]).absolute() for k in
        ('source_root','paper_root','yolo_test_root','yolo_test_selection_root'))]
    if any(output == p or p in output.parents or output in p.parents for p in protected):
        raise ValueError('New outputs must be separate from repository and frozen inputs')
    source = checked(Path(paths['source_root'])/'resolved_configuration.json', cfg['source_plan_sha256'])
    paper = checked(Path(paths['paper_root'])/'resolved_configuration.json', cfg['paper_plan_sha256'])
    checked(paths['completed_branches'], cfg['completed_branches_sha256'])
    dp = checked(Path(paths['yolo_test_root'])/'resolved_configuration.json',cfg['yolo_direct_plan_sha256'])
    sp = checked(Path(paths['yolo_test_selection_root'])/'resolved_configuration.json',cfg['yolo_selection_plan_sha256'])
    detector = checked(Path(dp['detector_run'])/'resolved_configuration.json',dp['detector_plan_sha256'])
    checkpoint = validate_checkpoint(Path(paths['yolo_checkpoint']), detector['config']['checkpoint_source'])
    if checkpoint['sha256'] != detector['checkpoint']['sha256']:
        raise ValueError('TRAIN/TEST detector weights differ')
    ci = model_identity(cfg['clip'],Path(paths['clip_model']))
    mp = paper['assets']['mpnet']
    checked(Path(mp['snapshot'])/'config.json',mp['configuration_file_sha256']['config.json'])
    if sha(Path(mp['snapshot'])/'model.safetensors') != mp['weight_sha256']:
        raise ValueError('MPNet weights differ')
    for filename, digest in mp['configuration_file_sha256'].items():
        if sha(Path(mp['snapshot'])/filename) != digest:
            raise ValueError('MPNet configuration differs')
    if (sha(Path(paths['parseq_checkpoint'])) != PARSEQ_CHECKPOINT_SHA256 or
            sha(Path(paths['parseq_archive'])) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError('PARSeq weights/archive differ')
    parseq_identity = verify_parseq_runtime_source(Path(paths['parseq_source']),
        source_type=PARSEQ_ARCHIVE_SOURCE,source_archive=Path(paths['parseq_archive']))
    if parseq_identity != dp['parseq_runtime']:
        raise ValueError('TRAIN/TEST PARSeq sources differ')
    audit = read(Path(paths['paper_root'])/'completion_audit.json')
    if audit['status'] != 'PASS':
        raise ValueError('Unqualified caption cache')
    captions = {}
    caption_inputs={str(Path(paths['paper_root'])/'completion_audit.json'):sha(Path(paths['paper_root'])/'completion_audit.json')}
    for path, digest in audit['caption_shard_sha256'].items():
        value = checked(path,digest)
        if value['contract']['profile'] == cfg['caption_profile']:
            sid = value['contract']['sample_id']
            if sid in captions or value['payload_sha256'] != payload_digest(value):
                raise ValueError('Duplicate or unsealed caption')
            if value['result']['ground_truth_in_caption'] or value['result']['parseq_in_prompt']:
                raise ValueError('Caption must be image-only')
            captions[sid] = {'path':path,'sha256':digest,'image_sha256':value['contract']['image_sha256'],
                             'text':value['result']['raw_caption']}
            caption_inputs[path]=digest
    tm = checked(paths['t1_manifest'],paper['config']['frozen_t1_manifest_sha256'])
    t1s = {r['sample_id']:r for r in tm['samples']}
    samples = {}
    inputs = {}
    for split, ids in source['split_ids'].items():
        samples[split] = []
        for sid in ids[:args.limit]:
            rp = Path(paths['source_root'])/split/'routes'/(key(sid)+'.json')
            route = checked(rp,paper['source_inputs_sha256'][str(rp)])
            if (route['payload_sha256'] != payload_digest(route) or
                    route['contract']['plan_sha256'] != cfg['source_plan_sha256']):
                raise ValueError('Original image route is unsealed')
            s = {k:route['observed'][k] for k in ('sample_id','image_path','image_sha256')}
            if sha(Path(s['image_path'])) != s['image_sha256']:
                raise ValueError('Original image changed')
            cap, t1 = captions[sid], t1s[sid]
            v = checked(t1['path'],t1['sha256'])
            if (cap['image_sha256'] != s['image_sha256'] or t1['image_sha256'] != s['image_sha256']
                    or v['contract']['image_sha256'] != s['image_sha256']
                    or v['payload_sha256'] != payload_digest(v)):
                raise ValueError('T1/caption source image mismatch')
            s.update(caption=cap, t1={**t1,'text':v['result']['raw_text']})
            samples[split].append(s);inputs[str(rp)] = sha(rp)
    if set(source['split_ids']['train']) & set(source['split_ids']['test']):
        raise ValueError('TRAIN/TEST ID leakage')
    if {s['image_sha256'] for s in samples['train']} & {s['image_sha256'] for s in samples['test']}:
        raise ValueError('TRAIN/TEST image leakage')
    software = {p:importlib.metadata.version(p) for p in
                ('torch','numpy','pillow','transformers','ultralytics','timm','rapidfuzz','sentence-transformers')}
    if software['ultralytics'] != detector['ultralytics']:
        raise ValueError('YOLO runtime differs')
    for p in ('torch','numpy','timm'):
        if software[p] != dp[p]:
            raise ValueError('Cached TEST OCR runtime differs: '+p)
    plan = {'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':'SMOKE_TEST' if args.limit else 'APPROXIMATE_EXPLORATORY_COMPARISON',
        'config':cfg,'paths':paths,'samples':samples,'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),
        'source_input_sha256':{**inputs,**caption_inputs},'code_sha256':{p:sha(REPO/p) for p in CODE},
        'clip_identity':ci,'mpnet_identity':mp,'yolo_identity':checkpoint,
        'yolo_config':detector['config'],'yolo_test_direct_plan':dp,'yolo_test_selection_plan':sp,
        'parseq_identity':parseq_identity,'software':software,'python':sys.version,
        'gpu':torch.cuda.get_device_name(0),'execution':cfg['execution'],
        'hardware_deviation':'RTX3090; sequential models; YOLO CUDA FP32 batch1, PARSeq CPU FP32 batch32, CLIP CUDA FP32 batch16, MPNet CPU FP32',
        'deterministic_algorithms':True,'tf32':False,'test_labels_used_to_fit':False,
        'source_selection_caveat':'YOLO joint0.60 family was explored on TEST previously; this is not untouched validation'}
    seal(output/'resolved_configuration.json',plan)
    return cfg, paths, plan, output


def select_rows(rows, cfg):
    selected = [r for r in rows if normalize(r['raw_text']) and joint_score(r) >= cfg['selection']['joint_threshold']]
    return nms(selected,cfg['selection']['nms_iou'],same_text=True,joint=True)


def infer_train(cfg,paths,plan,output):
    from ultralytics import YOLO
    ph = sha(output/'resolved_configuration.json')
    detector, recognizer = None, None
    detector_shapes=[]
    try:
        for i,s in enumerate(plan['samples']['train'],1):
            path = output/'train/ocr'/(key(s['sample_id'])+'.json')
            if path.exists():
                checked_shard(path,ph);continue
            with Image.open(s['image_path']) as image:
                rgb = np.asarray(image.convert('RGB'))
            if detector is None:
                torch.cuda.reset_peak_memory_stats()
                detector = YOLO(paths['yolo_checkpoint'])
                detector.model.register_forward_pre_hook(
                    lambda module, inputs: detector_shapes.append(list(inputs[0].shape)))
            start = time.monotonic(); inf = plan['yolo_config']['inference']
            detector_shapes.clear()
            pred = detector.predict(np.ascontiguousarray(rgb[:,:,::-1]),imgsz=inf['image_size'],
                conf=inf['confidence_floor'],iou=inf['nms_iou'],max_det=inf['max_detections'],
                device=inf['device'],half=False,verbose=False)[0]
            torch.cuda.synchronize()
            slots, rejected = boxes_to_slots(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),
                pred.boxes.cls.cpu().numpy(),(rgb.shape[1],rgb.shape[0]),inf['confidence_floor'])
            d_seconds = time.monotonic()-start
            if recognizer is None:
                recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),
                    checkpoint=Path(paths['parseq_checkpoint']),device='cpu',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                    runtime_source_archive=Path(paths['parseq_archive'])))
            rows=[];start=time.monotonic()
            batch_size=cfg['execution']['parseq_batch_size']
            for j in range(0,len(slots),batch_size):
                group = slots[j:j+batch_size]
                crops = [np.ascontiguousarray(rgb[r['bbox'][1]:r['bbox'][3],r['bbox'][0]:r['bbox'][2]]) for r in group]
                values = recognizer.recognize_candidate_batch(crops)
                if len(values) != len(group):
                    raise RuntimeError('Incomplete PARSeq batch')
                for slot,crop,value in zip(group,crops,values):
                    rows.append({'candidate_id':f"{s['sample_id']}:yolo:{slot['source_index']}",
                        'bbox':slot['bbox'],'detector_score':slot['score'],'crop_rgb_sha256':array_sha(crop),
                        'crop_shape':list(crop.shape),**value.to_dict()})
            save_shard(path,ph,{'sample_id':s['sample_id'],'image_sha256':s['image_sha256'],'rows':rows,
                'rejected_geometry':rejected,'detector_seconds':d_seconds,'ocr_seconds':time.monotonic()-start,
                'detector_peak':_cuda_peak('cuda:0'),'ground_truth_in_inference':False,
                'parseq_input_shape':[batch_size,3,32,128],'original_shape':list(rgb.shape),
                'yolo_input_shapes':detector_shapes.copy()})
            if i%25 == 0 or i == len(plan['samples']['train']):
                print(f'[YOLO TRAIN OCR] {i}/{len(plan["samples"]["train"])}',flush=True)
    finally:
        if recognizer is not None:recognizer.release()
        del detector,recognizer
        gc.collect();torch.cuda.empty_cache()


def save_shard(path, ph, value):
    result={'plan_sha256':ph,**value};result['payload_sha256']=payload_digest(result)
    seal(path,result)
    return result


def checked_shard(path,ph):
    r=read(path)
    if r['plan_sha256'] != ph or r['payload_sha256'] != payload_digest(r):
        raise ValueError('New shard identity mismatch: '+str(path))
    if 'sample_id' in r and Path(path).stem != key(r['sample_id']):
        raise ValueError('Shard path/sample identity mismatch')
    return r


def observations(split,cfg,paths,plan,output):
    ph=sha(output/'resolved_configuration.json');observed=[]
    for s in plan['samples'][split]:
        if split == 'train':
            original=checked_shard(output/'train/ocr'/(key(s['sample_id'])+'.json'),ph)
        else:
            op=Path(paths['yolo_test_root'])/'ocr'/(key(s['sample_id'])+'.json')
            original=checked(op,plan['yolo_test_selection_plan']['baseline_ocr_sha256'][s['sample_id']])
            if (original['image_sha256'] != s['image_sha256'] or
                    original['contract']['plan_sha256'] != cfg['yolo_direct_plan_sha256']):
                raise ValueError('TEST OCR/image mismatch')
        selected=select_rows(original['rows'],cfg)
        if split == 'test':
            ep=Path(paths['yolo_test_selection_root'])/'evaluation'/(key(s['sample_id'])+'.json')
            legacy=read(ep)['arms']['base640/joint_text_nms_0.6']['selected_candidate_ids']
            if [r['candidate_id'] for r in selected] != legacy:
                raise ValueError('Frozen YOLO TEST selection differs')
        rows=[{**r,'instance_id':r['candidate_id'],'confidence':reading_confidence(r)['confidence_balanced']}
              for r in selected]
        observed.append({**s,'rows':rows,'candidate_pool_size':len(original['rows'])})
    return observed


def score_split(split,cfg,paths,plan,output):
    ph=sha(output/'resolved_configuration.json');samples=observations(split,cfg,paths,plan,output)
    if all((output/split/'features'/(key(s['sample_id'])+'.json')).exists() for s in samples):
        return {s['sample_id']:checked_shard(output/split/'features'/(key(s['sample_id'])+'.json'),ph) for s in samples}
    from transformers import CLIPModel, CLIPProcessor
    torch.cuda.reset_peak_memory_stats()
    model, info=CLIPModel.from_pretrained(paths['clip_model'],local_files_only=True,output_loading_info=True,
                                         use_safetensors=False)
    if any(info.values()):raise ValueError('CLIP key mismatch: '+str(info))
    model=model.eval().float().to('cuda:0')
    clip_cfg={'textness_prompts':{'positive':cfg['positive_prompts'],'negative':cfg['negative_prompts']},
              'word_templates':cfg['word_templates'],'execution':{'batch_size':cfg['execution']['clip_batch_size']}}
    processor=CLIPProcessor.from_pretrained(paths['clip_model'],local_files_only=True)
    scorer=WordAdvantageScorer(model,processor,clip_cfg,'cuda:0')
    contrasts={}
    try:
        for i,s in enumerate(samples,1):
            path=output/split/'clip'/(key(s['sample_id'])+'.json')
            if path.exists():
                contrasts[s['sample_id']]=checked_shard(path,ph);continue
            with Image.open(s['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            crops=[]
            for r in s['rows']:
                a,b,c,d=r['bbox'];crop=np.ascontiguousarray(rgb[b:d,a:c])
                if array_sha(crop) != r['crop_rgb_sha256']:raise ValueError('Frozen RGB crop mismatch')
                if any(len(processor.tokenizer(t.format(text=r['raw_text']),truncation=False)['input_ids']) > model.config.text_config.max_position_embeddings
                       for t in cfg['word_templates']):raise ValueError('Refusing silent CLIP text truncation')
                crops.append(crop)
            values=scorer.score(s['rows'],crops)
            if len(values)!=len(s['rows']):raise RuntimeError('Incomplete CLIP crop scores')
            contrasts[s['sample_id']]=save_shard(path,ph,{**s,'rows':[
                {**r,'clip_word_advantage':v} for r,v in zip(s['rows'],values)],
                'ground_truth_in_scoring':False,'clip_peak':_cuda_peak('cuda:0'),
                'clip_input_shape':[cfg['execution']['clip_batch_size'],3,224,224]})
            if i%50==0 or i==len(samples):print(f'[CLIP {split}] {i}/{len(samples)}',flush=True)
    finally:
        del scorer,model;gc.collect();torch.cuda.empty_cache()
    from sentence_transformers import SentenceTransformer
    mp=SentenceTransformer(plan['mpnet_identity']['snapshot'],device='cpu',local_files_only=True,
                           model_kwargs={'torch_dtype':torch.float32})
    semantic=MPNetSemanticSimilarity(MPNetApproxConfig(device='cpu',max_sequence_length=cfg['mpnet_context_length']),model=mp)
    result={}
    try:
        for s in samples:
            sid=s['sample_id'];path=output/split/'features'/(key(sid)+'.json')
            if path.exists():result[sid]=checked_shard(path,ph);continue
            t3=assemble_t3(s['rows']);t1=s['t1']['text'];caption=s['caption']['text']
            lengths={k:len(mp.tokenizer(v,add_special_tokens=True,truncation=False)['input_ids'])
                     for k,v in [('T1',t1),('T3',t3['text']),('T2',caption)]}
            overflow=any(n>cfg['mpnet_context_length'] for n in lengths.values())
            decision=None;paper_scores={'paper_selected':None,'joined_t3':None}
            if not overflow:
                s1,s3=semantic.compare_candidates(t1,t3['text'],caption)
                decision=paper_decision(t1,t3['text'],caption,s1,s3,PaperRouterConfig(**cfg['paper_router']))
                paper_scores=localized_word_scores(decision,t3)
            result[sid]=save_shard(path,ph,{**contrasts[sid], 'paper_decision':decision,
                'paper_scores':paper_scores,'T3_assembly':t3,'token_lengths':lengths,
                'mpnet_overflow_forces_fallback':overflow,'ground_truth_in_scoring':False})
    finally:
        semantic.release();del semantic,mp;gc.collect()
    return result


def score_images(features,reference):
    return {name:{sid:(fixed_image_score(s['rows'],reference) if name=='clip' else
                (min(r['confidence'] for r in s['rows']) if s['rows'] else None) if name=='native' else
                min((joint_score(r) for r in s['rows']),default=None) if name=='joint' else
                s['paper_scores'][name]) for sid,s in features.items()}
            for name in ('clip','native','joint','paper_selected','joined_t3')}


def calibrate(cfg,plan,output,features):
    ids=[s['sample_id'] for s in plan['samples']['train']]
    split=partition(ids,cfg['calibration_images'],cfg['execution']['seed']) if len(ids)>1 else {'calibration':ids,'validation':[]}
    rows=[r for sid in split['calibration'] for r in features[sid]['rows'] if r['clip_word_advantage'] is not None]
    reference={'native':sorted(r['confidence'] for r in rows),
               'word_advantage':sorted(r['clip_word_advantage'] for r in rows)}
    scores=score_images(features,reference)
    cuts={name:{str(f):coverage_threshold(values,split['calibration'],f) for f in cfg['coverage_targets']}
          for name,values in scores.items()}
    return save_shard(output/'frozen_train_thresholds.json',sha(output/'resolved_configuration.json'),
        {'partition':split,'reference':reference,'cuts':cuts,'fit_split':'TRAIN calibration only',
         'test_features_present_at_freeze':len(list((output/'test/features').glob('*.json'))),
         'train_feature_sha256':{sid:sha(output/'train/features'/(key(sid)+'.json')) for sid in ids},
         'test_labels_used':False,'family_selection':False})


def metric_branches(cfg,paths,plan,output,features):
    from scripts.run_clip_router_e2e import qualify
    qualified=qualify(argparse.Namespace(mode='approximate',config=REPO/paths['qualification_config'],
        paths=REPO/paths['qualification_paths'],device='cpu',limit=None,output_root=None))[-1]
    old=checked(paths['completed_branches'],cfg['completed_branches_sha256'])
    old_by_id={r['sample_id']:r for rows in old['splits'].values() for r in rows}
    result={}
    for split, samples in qualified.items():
        result[split]={}
        for sample in samples:
            sid=sample['observed']['sample_id']
            if sid not in features[split]:continue
            f=features[split][sid];gt=sample['ground_truth'];predictions=[]
            if old_by_id[sid]['ground_truth'] != gt or old_by_id[sid]['deepsolo'] is None:
                raise ValueError('Missing fallback or changed canonical GT')
            for r in f['rows']:
                support=Image.new('L',tuple(sample['image_wh']),0)
                x1,y1,x2,y2=r['bbox'];ImageDraw.Draw(support).rectangle([x1,y1,x2-1,y2-1],fill=255)
                footprint=native_mask(support,256)
                ignored=bool(footprint.any() and not np.any(footprint & sample['validity']))
                predictions.append({'text':r['raw_text'],'bbox':r['bbox'],'ignored':ignored,
                                    'instance_id':r['instance_id']})
            t1=[{'text':f['t1']['text'],'ignored':False,'unlocalized':True}]
            if word_metrics(t1,gt) != old_by_id[sid]['t1']:
                raise ValueError('Frozen T1 metric replay changed')
            result[split][sid]={'yolo':word_metrics(predictions,gt),'deepsolo':old_by_id[sid]['deepsolo'],
                't1':word_metrics(t1,gt),'yolo_spatial':evaluate_predictions(predictions,gt,iou_threshold=.5),
                'ground_truth':gt,'yolo_predictions':predictions}
    seal(output/'word_branch_metrics.json',{'splits':result,'metric_protocol':cfg['metric'],
        'ignore_protocol':cfg['ignore_protocol'],'ground_truth_in_routing':False})
    return result


def evaluate(cfg,paths,plan,output,features,calibration):
    branches=metric_branches(cfg,paths,plan,output,features)
    report={'reproduction_mode':'approximate','claim_label':'approximate reproduction',
        'result_class':plan['result_class'],'plan_sha256':sha(output/'resolved_configuration.json'),
        'calibration_sha256':sha(output/'frozen_train_thresholds.json'),
        'metric_protocol':cfg['metric'],'ignore_protocol':cfg['ignore_protocol'],
        'paper_totaltext':{'resnet50_accuracy':.784,'vitaev2s_accuracy':.798,
            'comparable_metric':False,'reason':'Paper Table5 recognition accuracy lacks a defined denominator, matching and ignore protocol'},
        'splits':{},'paired_bootstrap':{},'test_tuning':False}
    table=[]
    for split, b in branches.items():
        ids=list(b);scores=score_images(features[split],calibration['reference'])
        points={'yolo_only':route_point(b,ids),'deepsolo_only':route_point(b,set())}
        source_t1={sid:'t1' if features[split][sid]['paper_decision'] is not None and
                   features[split][sid]['paper_decision']['semantic_winner']=='T1' else 'yolo' for sid in ids}
        policies={}
        for name,values in scores.items():
            for fraction,cut in calibration['cuts'][name].items():
                label=f'{name}/train_coverage_{fraction}'
                accepted={sid for sid in ids if cut is not None and values[sid] is not None and values[sid]>=cut}
                points[label]={**route_point(b,accepted,source_t1 if name=='paper_selected' else None),
                               'threshold':cut,'threshold_fit':'TRAIN80; transferred to TEST unchanged'}
                policies[label]=values
            if name.startswith('paper_') or name=='joined_t3':
                cut=cfg['paper_router']['tau'];accepted={sid for sid in ids if values[sid] is not None and values[sid]>=cut}
                points[name+'/paper_tau']={**route_point(b,accepted,source_t1 if name=='paper_selected' else None),'threshold':cut}
            for fraction in cfg['coverage_targets']:
                budget=round(len(ids)*fraction)
                label=f'{name}/same_budget_{budget}'
                accepted=budget_accept(values,ids,budget)
                points[label]={**route_point(b,accepted,source_t1 if name=='paper_selected' else None),
                    'threshold':None,'requested_budget':budget,'budget_rule':'descriptive GT-free rank; score descending then sample ID; not a transferred scalar threshold',
                    'rank_cut_score':min((values[sid] for sid in accepted),default=None)}
        oracles={str(round(len(ids)*f)):oracle(b,round(len(ids)*f)) for f in cfg['coverage_targets']}
        ideal=oracle(b)
        for label,p in points.items():
            p['oracle_gap_tn0']=ideal['combined']['accuracy_tn0']-p['combined']['accuracy_tn0']
            table.append({'split':split,'policy':label,'accepted_images':p['accepted_images'],
                'fallback_images':p['fallback_images'],'threshold':p.get('threshold'),
                'accepted_words':p['accepted_metrics']['detected_words'],
                'accepted_precision':p['accepted_metrics']['precision'],
                'accepted_complete_images':p['accepted_metrics']['complete_images'],
                **{k:p['combined'][k] for k in ('tp','fp','fn','detected_words','precision','recall','f1','accuracy_tn0')},
                'delta_tn0_vs_yolo':p['delta_tn0_vs_yolo']})
        report['splits'][split]={'images':len(ids),'gt_words':sum(r['yolo']['gt_words'] for r in b.values()),
            'points':points,'oracle':ideal,'equal_budget_oracles':oracles,
            'yolo_spatial_diagnostic':aggregate(r['yolo_spatial'] for r in b.values()),
            'scores':scores,'eligible_counts':{name:sum(v is not None for v in values.values()) for name,values in scores.items()}}
        for budget in sorted({round(len(ids)*f) for f in cfg['coverage_targets']}):
            a=points[f'clip/same_budget_{budget}'];c=points[f'paper_selected/same_budget_{budget}']
            report['paired_bootstrap'][f'{split}/same_budget_{budget}']=paired_bootstrap(b,a,c,
                cfg['bootstrap_iterations'],cfg['execution']['seed'])
    seal(output/'results.json',report)
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps({'output':str(output),'test':{k:v['combined'] for k,v in report['splits']['test']['points'].items()
        if k in ('yolo_only','deepsolo_only','clip/train_coverage_0.5','paper_selected/train_coverage_0.5','clip/same_budget_150','paper_selected/same_budget_150','paper_selected/paper_tau')}}),flush=True)
    return report


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    cfg,paths,plan,output=load_plan(args)
    e=cfg['execution'];random.seed(e['seed']);np.random.seed(e['seed']);torch.manual_seed(e['seed'])
    torch.cuda.manual_seed_all(e['seed']);torch.set_num_threads(e['cpu_threads'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        infer_train(cfg,paths,plan,output)
        train=score_split('train',cfg,paths,plan,output)
        cp=output/'frozen_train_thresholds.json'
        calibration=checked_shard(cp,sha(output/'resolved_configuration.json')) if cp.exists() else calibrate(cfg,plan,output,train)
        if calibration['test_features_present_at_freeze'] != 0:
            raise ValueError('Must freeze TRAIN-only cuts before TEST scoring')
        test=score_split('test',cfg,paths,plan,output)
        evaluate(cfg,paths,plan,output,{'train':train,'test':test},calibration)
    except Exception as error:
        write_json(output/'failure.json',{'status':'FAILED','error':str(error),'error_type':type(error).__name__,
            'traceback':traceback.format_exc(),'out_of_memory':isinstance(error,(torch.cuda.OutOfMemoryError,MemoryError)),
            'cuda_peak':_cuda_peak('cuda:0')})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--paths',type=Path,default=PATHS)
    p.add_argument('--output',type=Path);p.add_argument('--limit',type=int,choices=[1])
    run(p.parse_args())


if __name__=='__main__':main()
