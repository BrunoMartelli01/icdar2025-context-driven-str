"""APPROXIMATE text-completeness gate and paired YOLO640/960/1280 experiments."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from recognition.clip_router_evaluation import fixed_image_score, native_mask
from recognition.clip_router_prompt_ablation import budget_accept, coverage_threshold
from recognition.group_word_selection import reading_confidence
from recognition.paper_image_router_evaluation import word_metrics
from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
from recognition.text_coverage_router import coverage_features, fit_selector, scores
from recognition.yolo_router_comparison import aggregate
from scripts import run_yolo_router_comparison as base
from scripts import run_clip_router_e2e as source
from scripts.crop_clip_runtime import sha, array_sha, _cuda_peak
from scripts.run_clip_high_recall_admission import seal

REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/run_yolo_coverage_resolution.py', 'recognition/text_coverage_router.py', *base.CODE)


def prepare(args):
    cfg = source.read_approximate(args.config)
    paths = source.read_approximate(args.paths)
    if (args.mode != 'approximate' or cfg['reproduction_mode'] != args.mode
            or paths['reproduction_mode'] != args.mode or cfg['table_5_claim'] is not False
            or cfg['claim_label'] != 'approximate reproduction'
            or cfg['resolutions'] != [640, 960, 1280]
            or cfg['execution']['precision'] != 'float32' or cfg['execution']['device'] != 'cuda:0'
            or cfg['selection'] != {'joint_threshold': .6, 'nms_iou': .5}
            or cfg['detector_confidence_floor'] >= cfg['selection']['joint_threshold']**2):
        raise ValueError('Unsupported scientific configuration')
    output = (args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO):
        raise ValueError('Use an isolated external output directory')
    baseline = Path(paths['baseline_root'])
    old = base.checked(baseline/'resolved_configuration.json', cfg['baseline_plan_sha256'])
    old_results = base.checked(baseline/'results.json', cfg['baseline_results_sha256'])
    old_branches = base.checked(baseline/'word_branch_metrics.json', cfg['baseline_branches_sha256'])
    frozen = base.checked(baseline/'frozen_train_thresholds.json', old_results['calibration_sha256'])
    runtime_paths = source.read_approximate(REPO/paths['baseline_paths'])
    protected = [baseline, Path(paths['vitae_root']), *[Path(runtime_paths[k]) for k in
        ('source_root', 'paper_root', 'yolo_test_root', 'yolo_test_selection_root')]]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ValueError('Output must not contain or be inside frozen input directories')
    software = {p:importlib.metadata.version(p) for p in old['software']}
    if software != old['software']:
        raise ValueError('Paired experiment requires the frozen baseline software versions')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no automatic CPU fallback')
    vroot = Path(paths['vitae_root'])
    vr = base.checked(vroot/'results.json', cfg['vitae_results_sha256'])
    vp = base.checked(vroot/'resolved_configuration.json', cfg['vitae_plan_sha256'])
    if vr['status'] != 'SUCCESS' or vp['limit'] is not None:
        raise ValueError('Complete ViTAE fallback is required')
    base.validate_checkpoint(Path(runtime_paths['yolo_checkpoint']), old['yolo_config']['checkpoint_source'])
    base.model_identity(old['config']['clip'], Path(runtime_paths['clip_model']))
    values = source.qualify(SimpleNamespace(mode=args.mode, config=REPO/paths['qualification_config'],
        paths=REPO/paths['qualification_paths'], output_root=None))
    _, _, _, router_paths, _, _, _, _, _, _, _, qualified = values
    highroot = Path(runtime_paths['yolo_test_selection_root'])
    hp = base.checked(highroot/'resolved_configuration.json', cfg['high960_plan_sha256'])
    if hp['config']['highres_size'] != 960 or hp['ultralytics'] != old['software']['ultralytics']:
        raise ValueError('Cached960 runtime differs')
    fallback = {split: {r['sample_id']: r['word'] for r in vr['splits'][split]['per_image']}
                for split in ('train', 'test')}
    vgt = {split: {r['sample_id']: r['ground_truth'] for r in vp['images'][split]}
           for split in ('train', 'test')}
    samples, inputs, eval_data = {}, {}, {}
    for split, rows in qualified.items():
        ids = [r['observed']['sample_id'] for r in rows]
        if args.smoke:
            ids = frozen['partition']['calibration'][:1] if split == 'train' else ids[:1]
        samples[split] = []
        for r in rows:
            sid = r['observed']['sample_id']
            if sid not in ids:
                continue
            if r['ground_truth'] != old_branches['splits'][split][sid]['ground_truth'] or r['ground_truth'] != vgt[split][sid]:
                raise ValueError('Paired GT differs')
            mp = (Path(router_paths['segmentation_run'])/'validation_exports'/base.key(sid)/'maps.npz'
                  if split == 'train' else r['map_path'])
            fp = baseline/split/'features'/(base.key(sid)+'.json')
            base.checked_shard(fp, cfg['baseline_plan_sha256'])
            inputs[str(mp)] = sha(mp); inputs[str(fp)] = sha(fp)
            if split == 'test':
                cp = highroot/'ocr/high960'/(base.key(sid)+'.json')
                cpdata = base.read(cp)
                if cpdata['contract']['plan_sha256'] != cfg['high960_plan_sha256'] or cpdata['sample_id'] != sid:
                    raise ValueError('Cached960 sample contract differs')
                inputs[str(cp)] = sha(cp)
            samples[split].append({**{k:r['observed'][k] for k in ('sample_id','image_path','image_sha256')},
                'image_wh': r['image_wh'], 'orientation': r['orientation'], 'map_path': str(mp)})
            eval_data[sid] = r
    fit_ids = [s['sample_id'] for s in samples['train']] if args.smoke else frozen['partition']['calibration']
    val_ids = [] if args.smoke else frozen['partition']['validation']
    plan = {'schema_version':1, 'reproduction_mode':'approximate', 'claim_label':'approximate reproduction',
        'result_class':'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_COMPARISON',
        'table_5_claim':False, 'config':cfg, 'paths':paths, 'samples':samples,
        'calibration_ids':fit_ids, 'validation_ids':val_ids,
        'baseline_runtime_paths':runtime_paths, 'baseline_plan':old,
        'cached_input_sha256': inputs, 'code_sha256':{p:sha(REPO/p) for p in CODE},
        'config_sha256':sha(args.config), 'paths_sha256':sha(args.paths),
        'software':software, 'python':sys.version, 'cuda_build':torch.version.cuda,
        'deterministic_algorithms':True, 'tf32':False,
        'gpu':torch.cuda.get_device_name(0), 'test_labels_used_for_selection':False,
        'source_selection_caveat':'TEST has previously been inspected; exploratory comparison only'}
    root = output/('smoke' if args.smoke else 'full')/hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg, plan, root, eval_data, fallback, old_branches


def load_original_rgb(s):
    if sha(Path(s['image_path'])) != s['image_sha256']:
        raise ValueError('Original image changed')
    with Image.open(s['image_path']) as im:
        rgb = np.asarray(im.convert('RGB'))
    if list(rgb.shape[:2][::-1]) != s['image_wh']:
        raise ValueError('Stored-raster image frame changed')
    return rgb


def select(rows, cfg):
    return [{**r, 'instance_id':r['candidate_id'],
             'confidence':reading_confidence(r)['confidence_balanced']}
            for r in base.select_rows(rows,cfg)]


def observations(cfg, plan, root, split, resolution):
    """Reuse exact frozen640 and cached960 TEST; obtain all other real detector/OCR outputs."""
    paths = plan['baseline_runtime_paths'];ph = sha(root/'resolved_configuration.json')
    records = {};detector = recognizer = None;shapes=[]
    try:
        for i,s in enumerate(plan['samples'][split],1):
            sid=s['sample_id'];path=root/str(resolution)/split/'observations'/(base.key(sid)+'.json')
            if path.exists():records[sid]=base.checked_shard(path,ph);continue
            if resolution == 640:
                oldpath=Path(plan['paths']['baseline_root'])/split/'features'/(base.key(sid)+'.json')
                raw=base.checked_shard(oldpath,cfg['baseline_plan_sha256'])
                rows=raw['rows'];receipt={'source_cache':str(oldpath),'source_sha256':sha(oldpath),'new_inference':False}
            elif resolution == 960 and split == 'test':
                oldpath=Path(paths['yolo_test_selection_root'])/'ocr/high960'/(base.key(sid)+'.json')
                if sha(oldpath) != plan['cached_input_sha256'][str(oldpath)]:raise ValueError('960 cache changed')
                raw=base.read(oldpath);rows=select(raw['rows'],cfg)
                receipt={'source_cache':str(oldpath),'source_sha256':sha(oldpath),'new_inference':False,
                         'source_detector_device':'cpu','source_parseq_device':'cpu'}
            else:
                from ultralytics import YOLO
                rgb=load_original_rgb(s)
                if detector is None:
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    detector=YOLO(paths['yolo_checkpoint'])
                    detector.model.register_forward_pre_hook(lambda m,v:shapes.append(list(v[0].shape)))
                    recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),
                        checkpoint=Path(paths['parseq_checkpoint']),device='cpu',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        runtime_source_archive=Path(paths['parseq_archive'])))
                shapes.clear();start=time.monotonic();inf=plan['baseline_plan']['yolo_config']['inference']
                pred=detector.predict(np.ascontiguousarray(rgb[:,:,::-1]),imgsz=resolution,
                    conf=cfg['detector_confidence_floor'],iou=inf['nms_iou'],max_det=inf['max_detections'],
                    device='cuda:0',half=False,verbose=False)[0]
                torch.cuda.synchronize()
                slots,rejected=base.boxes_to_slots(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),
                    pred.boxes.cls.cpu().numpy(),tuple(s['image_wh']),cfg['detector_confidence_floor'])
                detector_seconds=time.monotonic()-start;rawrows=[];batch_shapes=[];start=time.monotonic()
                batch_size=cfg['execution']['parseq_batch_size']
                for j in range(0,len(slots),batch_size):
                    group=slots[j:j+batch_size]
                    crops=[np.ascontiguousarray(rgb[r['bbox'][1]:r['bbox'][3],r['bbox'][0]:r['bbox'][2]]) for r in group]
                    values=recognizer.recognize_candidate_batch(crops)
                    batch_shapes.append([len(crops),3,32,128])
                    if len(values)!=len(group):raise RuntimeError('Incomplete PARSeq batch')
                    for slot,crop,v in zip(group,crops,values):
                        rawrows.append({'candidate_id':f'{sid}:yolo{resolution}:{slot["source_index"]}',
                            'bbox':slot['bbox'],'detector_score':slot['score'],'crop_rgb_sha256':array_sha(crop),
                            'crop_shape':list(crop.shape),**v.to_dict()})
                rows=select(rawrows,cfg)
                receipt={'new_inference':True,'detector_device':'cuda:0','parseq_device':'cpu',
                    'detector_seconds':detector_seconds,'ocr_seconds':time.monotonic()-start,
                    'detector_peak':_cuda_peak('cuda:0'),'yolo_input_shapes':shapes.copy(),
                    'rejected_geometry':rejected,'raw_candidate_count':len(rawrows),
                    'parseq_input_shapes':batch_shapes,'memory_peak_scope':'cumulative detector stage'}
                base.save_shard(root/str(resolution)/split/'raw_ocr'/(base.key(sid)+'.json'),ph,
                    {'sample_id':sid,'image_sha256':s['image_sha256'],'rows':rawrows,**receipt})
            records[sid]=base.save_shard(path,ph,{'sample_id':sid,'image_sha256':s['image_sha256'],
                'rows':rows,'execution':receipt,'ground_truth_in_inference':False})
            if i%25==0 or i==len(plan['samples'][split]):print(f'[YOLO{resolution} {split} observations] {i}/{len(plan["samples"][split])}',flush=True)
    finally:
        if recognizer is not None:recognizer.release()
        del recognizer,detector;gc.collect();torch.cuda.empty_cache()
    return records


def feature_rows(cfg, plan, root, split, resolution, observed):
    ph=sha(root/'resolved_configuration.json');result={};scorer=model=None
    try:
        for i,s in enumerate(plan['samples'][split],1):
            sid=s['sample_id'];path=root/str(resolution)/split/'features'/(base.key(sid)+'.json')
            if path.exists():result[sid]=base.checked_shard(path,ph);continue
            rows=observed[sid]['rows'];rgb=load_original_rgb(s);crops=[]
            for r in rows:
                x1,y1,x2,y2=r['bbox'];crop=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                if array_sha(crop)!=r['crop_rgb_sha256']:raise ValueError('Original RGB crop changed')
                crops.append(crop)
            if resolution != 640 and rows:
                if model is None:
                    from transformers import CLIPModel,CLIPProcessor
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                    model,info=CLIPModel.from_pretrained(plan['baseline_runtime_paths']['clip_model'],
                        local_files_only=True,output_loading_info=True,use_safetensors=False)
                    if any(info.values()):raise ValueError('CLIP checkpoint incompatible: '+str(info))
                    model=model.eval().float().to('cuda:0')
                    proc=CLIPProcessor.from_pretrained(plan['baseline_runtime_paths']['clip_model'],local_files_only=True)
                    bc=plan['baseline_plan']['config']
                    clip_cfg={'textness_prompts':{'positive':bc['positive_prompts'],'negative':bc['negative_prompts']},
                        'word_templates':bc['word_templates'],'execution':{'batch_size':cfg['execution']['clip_batch_size']}}
                    scorer=base.WordAdvantageScorer(model,proc,clip_cfg,'cuda:0')
                values=scorer.score(rows,crops)
                if len(values)!=len(rows):raise RuntimeError('Incomplete CLIP batch')
                rows=[{**r,'clip_word_advantage':v} for r,v in zip(rows,values)]
            mp=Path(s['map_path'])
            if sha(mp)!=plan['cached_input_sha256'][str(mp)]:raise ValueError('Probability export changed')
            with np.load(mp,allow_pickle=False) as z:probabilities=z['probabilities'].copy()
            cov=coverage_features(probabilities,tuple(s['image_wh']),[r['bbox'] for r in rows],
                probability_threshold=cfg['coverage_probability_threshold'],minimum_component_area=cfg['coverage_minimum_component_area'])
            result[sid]=base.save_shard(path,ph,{'sample_id':sid,'image_sha256':s['image_sha256'],
                'rows':rows,'coverage':cov,'map_sha256':sha(mp),'ground_truth_used':False,
                'clip_peak':_cuda_peak('cuda:0') if model is not None else None})
            if i%50==0 or i==len(plan['samples'][split]):print(f'[features{resolution} {split}] {i}/{len(plan["samples"][split])}',flush=True)
    finally:
        del scorer,model;gc.collect();torch.cuda.empty_cache()
    return result


def evaluated_branches(features, eval_data, fallback, split):
    result={}
    for sid,f in features.items():
        s=eval_data[sid];pred=[]
        for r in f['rows']:
            support=Image.new('L',tuple(s['image_wh']),0);x1,y1,x2,y2=r['bbox']
            ImageDraw.Draw(support).rectangle([x1,y1,x2-1,y2-1],fill=255)
            mask=native_mask(support,256)
            ignored=bool(mask.any() and not np.any(mask & s['validity']))
            pred.append({'text':r['raw_text'],'bbox':r['bbox'],'ignored':ignored})
        result[sid]={'light':word_metrics(pred,s['ground_truth']),'deepsolo':fallback[split][sid]}
    return result


def with_confidence(features, reference):
    return {sid:{**f,'confidence_score':fixed_image_score(f['rows'],reference)} for sid,f in features.items()}


def evaluate_points(cfg, fit, features, branches, ids, baseline_cuts):
    winner=fit['winner'];mixture=scores(features,winner['reference'],winner['weight'],winner['coverage_field'])
    confidence={sid:f['confidence_score'] for sid,f in features.items()}
    points={}
    for name,values,cuts in [('confidence',confidence,baseline_cuts),('coverage',mixture,fit['scalar_thresholds'])]:
        for fraction in cfg['bypass_fractions']:
            for mode in ['budget','scalar']:
                threshold=cuts[str(fraction)]
                accepted=(budget_accept(values,ids,round(len(ids)*fraction)) if mode=='budget' else
                    {sid for sid in ids if values[sid] is not None and threshold is not None and values[sid]>=threshold})
                if mode=='budget' and len(accepted)!=round(len(ids)*fraction):
                    raise ValueError('Insufficient eligible images for requested bypass budget')
                points[f'{name}/{mode}_{fraction}']={'accepted_ids':sorted(accepted),
                    'accepted_images':len(accepted),'fallback_images':len(ids)-len(accepted),
                    'bypass_fraction':len(accepted)/len(ids),'threshold':threshold if mode=='scalar' else None,
                    'combined':aggregate([branches[sid]['light' if sid in accepted else 'deepsolo'] for sid in ids]),
                    'accepted':aggregate([branches[sid]['light'] for sid in ids if sid in accepted])}
    return {'images':len(ids),'light_only':aggregate([branches[s]['light'] for s in ids]),
        'deepsolo_only':aggregate([branches[s]['deepsolo'] for s in ids]),'points':points}


def run(args):
    cfg,plan,root,eval_data,fallback,old_branches=prepare(args)
    if args.check:
        print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'images':{s:len(r) for s,r in plan['samples'].items()}}));return
    seal(root/'resolved_configuration.json',plan)
    for name,digest in plan['code_sha256'].items():
        data=(REPO/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('Source changed during snapshot')
        target=root/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=digest:raise ValueError('Source snapshot changed')
        if not target.exists():target.write_bytes(data)
    random.seed(cfg['execution']['seed']);np.random.seed(cfg['execution']['seed']);torch.manual_seed(cfg['execution']['seed'])
    torch.cuda.manual_seed_all(cfg['execution']['seed']);torch.set_num_threads(cfg['execution']['cpu_threads'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    session = {'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'gpu':torch.cuda.get_device_name(0), 'total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'nvidia_smi':subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout,
        'seed':cfg['execution']['seed'],'precision':'float32','model_lifecycle':'sequential',
        'deterministic_algorithms':True,'cudnn_deterministic':True,'tf32':False}
    seal(root/'execution_sessions'/(str(time.time_ns())+'.json'),session)
    fitted={};train={};references={};branches={};cuts={}
    try:
        # Freeze every resolution's selector before constructing any current TEST feature.
        for resolution in cfg['resolutions']:
            observed=observations(cfg,plan,root,'train',resolution)
            f=feature_rows(cfg,plan,root,'train',resolution,observed)
            rows=[r for sid in plan['calibration_ids'] for r in f[sid]['rows']]
            ref={'native':sorted(r['confidence'] for r in rows),'word_advantage':sorted(r['clip_word_advantage'] for r in rows)}
            train[resolution]=with_confidence(f,ref);references[resolution]=ref
            b=evaluated_branches(f,eval_data,fallback,'train');branches[resolution]=b
            if resolution==640:
                for sid,r in b.items():
                    if r['light']!=old_branches['splits']['train'][sid]['yolo']:raise ValueError('Frozen640 TRAIN replay differs')
            fit=fit_selector({sid:train[resolution][sid] for sid in plan['calibration_ids']},
                {sid:b[sid] for sid in plan['calibration_ids']},plan['calibration_ids'],weights=cfg['mixture_weights'],
                coverage_fields=cfg['coverage_fields'],coverage_targets=cfg['bypass_fractions'],primary_coverage=cfg['primary_bypass_fraction'])
            fit['current_test_features_present_at_freeze']=len(list((root/str(resolution)/'test/features').glob('*.json')))
            # On resume TEST may already exist, but the previously sealed fit must match.
            fit_path=root/str(resolution)/'frozen_train_selector.json'
            if fit_path.exists():
                previous=base.read(fit_path)
                fit['current_test_features_present_at_freeze']=previous['current_test_features_present_at_freeze']
            elif fit['current_test_features_present_at_freeze']:
                raise ValueError('Refusing to fit a new selector after TEST features exist')
            fit['confidence_reference']=ref
            confidence={sid:v['confidence_score'] for sid,v in train[resolution].items()}
            fit['confidence_thresholds']={str(t):coverage_threshold(confidence,plan['calibration_ids'],t) for t in cfg['bypass_fractions']}
            seal(fit_path,fit);fitted[resolution]=fit;cuts[resolution]=fit['confidence_thresholds']
            print(f'[FROZEN {resolution}] coverage={fit["winner"]["coverage_field"]} weight={fit["winner"]["weight"]}',flush=True)
        results={'status':'SUCCESS','claim_label':'approximate reproduction','reproduction_mode':'approximate',
            'result_class':plan['result_class'],'metric':cfg['metric'],'official_evaluation':False,'resolutions':{}}
        for resolution in cfg['resolutions']:
            observed=observations(cfg,plan,root,'test',resolution)
            f=feature_rows(cfg,plan,root,'test',resolution,observed)
            test=with_confidence(f,references[resolution]);tb=evaluated_branches(f,eval_data,fallback,'test')
            if resolution==640:
                for sid,r in tb.items():
                    if r['light']!=old_branches['splits']['test'][sid]['yolo']:raise ValueError('Frozen640 TEST replay differs')
            item={'selected_weight':fitted[resolution]['winner']['weight'],
                  'selected_coverage_field':fitted[resolution]['winner']['coverage_field']}
            for name,features,b,ids in [('train',train[resolution],branches[resolution],list(train[resolution])),
                ('calibration',train[resolution],branches[resolution],plan['calibration_ids']),
                ('validation',train[resolution],branches[resolution],plan['validation_ids']),
                ('test',test,tb,list(test))]:
                if ids:item[name]=evaluate_points(cfg,fitted[resolution],features,b,ids,cuts[resolution])
            item['branch_metrics']={'train':branches[resolution],'test':tb}
            results['resolutions'][str(resolution)]=item
            point=item['test']['points']['coverage/budget_0.5']
            print(f'[RESULT {resolution}] {json.dumps(point["combined"])} bypass={point["accepted_images"]}',flush=True)
        seal(root/'results.json',results)
        print(json.dumps({'status':'SUCCESS','root':str(root)}),flush=True)
    except Exception as error:
        failure={'status':'FAILED','error':str(error),'traceback':traceback.format_exc(),'partial_results_are_not_final':True,
            'out_of_memory':isinstance(error,torch.cuda.OutOfMemoryError),'cuda_peak':_cuda_peak('cuda:0')}
        seal(root/'failures'/(hashlib.sha256(json.dumps(failure,sort_keys=True).encode()).hexdigest()[:16]+'.json'),failure)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,default=REPO/'configs/approximate/yolo_coverage_resolution_v1.json')
    parser.add_argument('--paths',type=Path,default=REPO/'configs/approximate/yolo_coverage_resolution_local_paths_v1.json')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())


if __name__=='__main__':main()
