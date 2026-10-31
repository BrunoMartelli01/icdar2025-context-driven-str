"""Approximate grounded-context ablations, frozen base branches and TRAIN-only fit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback
import warnings

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from PIL import Image
import torch
from sklearn.exceptions import ConvergenceWarning

from recognition.grounded_context_router import (
    inventory_features, verification_jobs, verification_features, local_views, feature_sets,
    generation_incomplete,
)
from recognition.trainable_image_router import feature_dict
from scripts import run_trainable_image_router as baseline
from scripts.run_yolo_f1_quality_study import checked, read, seal, sha
from scripts.run_yolo_router_comparison import checked_shard

REPO = Path(__file__).resolve().parents[1]
CODE = tuple(dict.fromkeys(('scripts/run_grounded_context_router.py', 'recognition/grounded_context_router.py',
    'recognition/grounded_context_runtime.py', 'recognition/residual_text_vqa.py',
    'scripts/contextual_lm_verifier_runtime.py', 'scripts/crop_clip_runtime.py',
    'scripts/run_yolo_router_comparison.py', *baseline.CODE)))


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config,args.paths)):
        raise ValueError('Explicit isolated approximate mode/configuration required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 experiment required')
    if cfg['model_kinds'] != ['ridge','trees'] or cfg['feature_sets'] != ['core','inventory','verified','visual','combined']:
        raise ValueError('Unexpected predeclared ablations')
    if cfg['execution']['precision'] != 'float32' or not cfg['execution']['device'].startswith('cuda'):
        raise ValueError('Explicit FP32 CUDA profile required; no substitution')
    if cfg['geometry_settings']['maximum_verified_words'] < 1 or cfg['inventory_settings']['embedding_dimension'] < 1:
        raise ValueError('Explicit positive probe and embedding dimensions required')
    if cfg['inventory_settings']['tasks'] != ['<OCR_WITH_REGION>','<DENSE_REGION_CAPTION>']:
        raise ValueError('Independent inventory task contract differs')
    source = Path(paths['baseline_root'])
    source_plan = checked(source/'resolved_configuration.json', cfg['baseline_source']['plan'])
    source_result = checked(source/'results.json', cfg['baseline_source']['results'])
    audit = checked(source/'independent_audit.json', cfg['baseline_source']['audit'])
    if source_result['status'] != 'SUCCESS' or audit['status'] != 'PASS' or audit['results_sha256'] != cfg['baseline_source']['results']:
        raise ValueError('Audited complete baseline is required')
    oldargs = argparse.Namespace(mode='approximate',config=Path(paths['baseline_config']), paths=Path(paths['baseline_paths']),
        output_root=None,smoke=args.smoke)
    oldcfg, oldplan, _, records, branches = baseline.prepare(oldargs)
    if cfg['sources'] != oldcfg['sources'] or cfg['feature_settings'] != oldcfg['feature_settings']:
        raise ValueError('Frozen baseline inputs/features differ')
    coverage = Path(oldplan['source_coverage_root']); coverage_hash = sha(coverage/'resolved_configuration.json')
    inputs = dict(oldplan['input_sha256']); observations = {}
    for split, samples in oldplan['samples'].items():
        observations[split] = {}
        for sample in samples:
            sid = sample['sample_id']; path = coverage/'640'/split/'features'/(sid.replace(':','__')+'.json')
            shard = checked_shard(path, coverage_hash)
            if shard['ground_truth_used'] is not False or shard['sample_id'] != sid or shard['image_sha256'] != sample['image_sha256']:
                raise ValueError('Observable word source identity differs')
            if sha(Path(sample['image_path'])) != sample['image_sha256']: raise ValueError('Original image changed')
            observations[split][sid] = {'sample': sample, 'rows': shard['rows'], 'parts': records[split][sid]}
            inputs[str(path)] = sha(path)
    for path, pin in [('resolved_configuration.json','plan'),('results.json','results'),('independent_audit.json','audit')]:
        inputs[str(source/path)] = cfg['baseline_source'][pin]
    inventory_path = Path(paths['inventory_snapshot']); verifier_path = Path(paths['verifier_snapshot'])
    if inventory_path.name != cfg['inventory_identity']['revision'] or verifier_path.name != cfg['verifier_identity']['revision']:
        raise ValueError('Pinned visual model revision differs')
    assets = [(inventory_path,cfg['inventory_identity']['file_sha256']),
              (verifier_path,{**cfg['verifier_identity']['configuration_file_sha256'],**cfg['verifier_identity']['weight_sha256']})]
    for snapshot, files in assets:
        for name, digest in files.items():
            if sha(snapshot/name) != digest: raise ValueError('Visual checkpoint/configuration changed: '+str(snapshot/name))
            inputs[str(snapshot/name)] = digest
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU or model substitution')
    output = Path(args.output_root or paths['output_root']).resolve()
    protected = [REPO,source,inventory_path,verifier_path,*(Path(oldplan['paths'][k]) for k in ('residual_root','quality_root'))]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('External outputs must be separate from sources and checkpoints')
    first = next(iter(observations['train'].values()))
    _, oldschemas, _ = feature_dict(first['parts']['residual'],first['parts']['quality'],first['sample']['image_wh'],cfg['feature_settings'])
    empty_inventory,_ = inventory_features({'quad_boxes':[],'labels':[]},{'bboxes':[],'labels':[]},[],[1,1],cfg['geometry_settings'])
    empty_inventory.update(inventory_ocr_incomplete=0., inventory_objects_incomplete=0.)
    _, schemas = feature_sets({},oldschemas,empty_inventory,verification_features([],0),[0.]*cfg['inventory_settings']['embedding_dimension'])
    reused=None
    if getattr(args,'reuse_train_inventory',None) is not None:
        reuse=args.reuse_train_inventory.resolve();original=read(reuse/'resolved_configuration.json')
        receipt=read(reuse/'train/inventory_receipt.json');model=receipt['model']
        for key in ('inventory_settings','inventory_identity','execution'):
            if original['config'][key]!=cfg[key]: raise ValueError('Reused neural inventory contract differs: '+key)
        if original['code_sha256']['recognition/grounded_context_runtime.py']!=sha(REPO/'recognition/grounded_context_runtime.py'):
            raise ValueError('Only byte-identical frozen inventory runtimes can share raw generations')
        if original['samples']['train']!=oldplan['samples']['train'] or receipt['plan_sha256']!=sha(reuse/'resolved_configuration.json'):
            raise ValueError('Reused TRAIN population/receipt differs')
        if model['model_state_sha256_before']!=model['model_state_sha256_after'] or model['optimizer_steps']!=0:
            raise ValueError('Reused inventory is not a qualified frozen inference')
        files={str(p.relative_to(reuse)):sha(p) for p in (reuse/'train/inventory').glob('*.json')}
        if len(files)!=len(oldplan['samples']['train']): raise ValueError('Reused TRAIN inventory is incomplete')
        reused={'root':str(reuse),'plan_sha256':sha(reuse/'resolved_configuration.json'),
            'receipt_sha256':sha(reuse/'train/inventory_receipt.json'),'files_sha256':files}
    plan = {**oldplan,'config':cfg,'paths':{**paths,'residual_root':oldplan['paths']['residual_root'],'quality_root':oldplan['paths']['quality_root']},
        'config_sha256':sha(args.config),'paths_sha256':sha(args.paths),'input_sha256':inputs,
        'families':[k+'_'+f for k in cfg['model_kinds'] for f in cfg['feature_sets']], 'feature_schemas':schemas,
        'code_sha256':{f:sha(REPO/f) for f in CODE},'baseline_plan_sha256':cfg['baseline_source']['plan'],
        'result_class':'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_GROUNDED_ROUTER',
        'new_visual_model_inference':True,'router_training':True,'base_models_frozen':True,
        'test_history':'Repeatedly explored TEST, not untouched validation; DeepSolo fine-tuned on Total-Text TRAIN',
        'gpu':torch.cuda.get_device_name(0),'cuda_build':torch.version.cuda,
        'software':{p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','scipy','scikit-learn','Pillow','einops')},
        'deterministic_algorithms':True,'tf32':False,'partial_verification_is_not_complete_image_certification':True,
        'reused_train_inventory':reused}
    tag = hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return cfg,plan,output/('smoke' if args.smoke else 'full')/tag,observations,branches


def rgb(sample):
    if sha(Path(sample['image_path'])) != sample['image_sha256']: raise ValueError('Image checksum drift')
    with Image.open(sample['image_path']) as im: value = np.asarray(im.convert('RGB'))
    if list(value.shape[:2][::-1]) != sample['image_wh']: raise ValueError('RGB orientation or shape differs')
    return value


def source_features(cfg, observed):
    p = observed['parts']
    return feature_dict(p['residual'],p['quality'],observed['sample']['image_wh'],cfg['feature_settings'])


def inventory_stage(cfg,plan,root,split,observed):
    from recognition.grounded_context_runtime import FrozenInventory
    target = root/split/'inventory_receipt.json'; ph = sha(root/'resolved_configuration.json')
    if target.exists():
        receipt = read(target)
        if receipt['plan_sha256'] != ph: raise ValueError('Inventory receipt plan differs')
        return
    if split == 'test' and not (root/'frozen_choices.json').exists(): raise ValueError('Freeze TRAIN choices before TEST inventory')
    if split=='train' and plan.get('reused_train_inventory'):
        reuse=plan['reused_train_inventory'];source=Path(reuse['root'])
        for sid,value in observed.items():
            relative='train/inventory/'+sid.replace(':','__')+'.json'
            original=checked(source/relative,reuse['files_sha256'][relative])
            if (original['plan_sha256']!=reuse['plan_sha256'] or original['sample_id']!=sid
                    or original['image_sha256']!=value['sample']['image_sha256']
                    or original['ground_truth_used'] or original['ocr_predictions_given_to_model']):
                raise ValueError('Reused inventory sample identity or inference contract differs')
            seal(root/relative,{**original,'plan_sha256':ph,'reused_source_sha256':reuse['files_sha256'][relative]})
        receipt=checked(source/'train/inventory_receipt.json',reuse['receipt_sha256'])
        seal(target,{**receipt,'plan_sha256':ph,'seconds':0.,'reused_source_seconds':receipt['seconds'],
            'reused_source_receipt_sha256':reuse['receipt_sha256']})
        print(f'[inventory train] reused {len(observed)} qualified raw outputs; no repeated neural generation',flush=True)
        return
    model = FrozenInventory(plan['paths']['inventory_snapshot'],cfg['inventory_settings'],cfg['execution']['device'])
    started = time.monotonic()
    try:
        for number,(sid,value) in enumerate(observed.items(),1):
            path = root/split/'inventory'/(sid.replace(':','__')+'.json')
            if path.exists():
                cached=read(path)
                if cached['plan_sha256'] != ph or cached['image_sha256'] != value['sample']['image_sha256']:
                    raise ValueError('Inventory cache identity changed')
            else:
                result = model.infer(rgb(value['sample']))
                seal(path,{'plan_sha256':ph,'sample_id':sid,'image_sha256':value['sample']['image_sha256'],
                    'frozen_choices_sha256':sha(root/'frozen_choices.json') if split=='test' else None,**result})
            if number%10 == 0 or number == len(observed): print(f'[inventory {split}] {number}/{len(observed)}',flush=True)
        receipt = model.finish()
        seal(target,{'plan_sha256':ph,'images':len(observed),'seconds':time.monotonic()-started,'model':receipt})
    finally: model.close()


def derived_inventory(cfg, value, raw):
    fields,evidence = inventory_features(raw['tasks']['<OCR_WITH_REGION>']['parsed'],raw['tasks']['<DENSE_REGION_CAPTION>']['parsed'],
        value['rows'],value['sample']['image_wh'],cfg['geometry_settings'])
    limit=cfg['inventory_settings']['generation']['max_new_tokens']
    fields['inventory_ocr_incomplete'] = float(generation_incomplete(raw['tasks']['<OCR_WITH_REGION>'],limit))
    fields['inventory_objects_incomplete'] = float(generation_incomplete(raw['tasks']['<DENSE_REGION_CAPTION>'],limit))
    return fields,evidence


def verification_stage(cfg,plan,root,split,observed):
    from recognition.grounded_context_runtime import FrozenLocalVerifier
    target = root/split/'verification_receipt.json'; ph = sha(root/'resolved_configuration.json')
    if target.exists():
        if read(target)['plan_sha256'] != ph: raise ValueError('Verifier receipt plan differs')
        return
    if split == 'test' and not (root/'frozen_choices.json').exists(): raise ValueError('Freeze before TEST verification')
    judge = FrozenLocalVerifier(plan['paths']['verifier_snapshot'],device=cfg['execution']['device'],
        plain_question=cfg['verification_question'],context_template='{caption}')
    started = time.monotonic(); questions = 0
    try:
        for number,(sid,value) in enumerate(observed.items(),1):
            name = sid.replace(':','__')+'.json'; path = root/split/'verification'/name
            if path.exists():
                cached = read(path)
                if cached['plan_sha256'] != ph or cached['image_sha256'] != value['sample']['image_sha256']:
                    raise ValueError('Verifier cache identity changed')
            else:
                independent = read(root/split/'inventory'/name); _,evidence = derived_inventory(cfg,value,independent)
                jobs = verification_jobs(value['rows'],value['parts']['residual']['word_evidence']['words'],
                    evidence,value['sample']['image_wh'],cfg['geometry_settings'])
                image = rgb(value['sample']); probes = []
                for job in jobs:
                    probes.append(judge.verify(local_views(image,job,cfg['geometry_settings']['mask_value']),job,cfg['verification_question']))
                cached = {'plan_sha256':ph,'sample_id':sid,'image_sha256':value['sample']['image_sha256'],'probes':probes,
                    'frozen_choices_sha256':sha(root/'frozen_choices.json') if split=='test' else None,'ground_truth_used':False}
                seal(path,cached)
            questions += sum(sum(len(x) for x in p['answers'].values()) for p in cached['probes'])
            if number%10 == 0 or number == len(observed): print(f'[local verification {split}] {number}/{len(observed)}',flush=True)
        if judge.provenance['native_forward_check'] is None:
            # A resumed completed stage must still qualify this fresh model before its receipt.
            for sid,value in observed.items():
                probes=read(root/split/'verification'/(sid.replace(':','__')+'.json'))['probes']
                if probes:
                    judge.verify(local_views(rgb(value['sample']),probes[0],cfg['geometry_settings']['mask_value']),probes[0],cfg['verification_question'])
                    break
        seal(target,{'plan_sha256':ph,'images':len(observed),'questions':questions,'seconds':time.monotonic()-started,'model':judge.finish()})
    finally: judge.close()


def features(cfg,plan,root,split,observed):
    result = {}; ph = sha(root/'resolved_configuration.json')
    binding = sha(root/'frozen_choices.json') if split == 'test' else None
    for sid,value in observed.items():
        name=sid.replace(':','__')+'.json'; independent=read(root/split/'inventory'/name); local=read(root/split/'verification'/name)
        for record in (independent,local):
            if (record['plan_sha256']!=ph or record['image_sha256']!=value['sample']['image_sha256']
                    or record['frozen_choices_sha256']!=binding): raise ValueError('Probe identity/freeze binding mismatch')
        base,schema,eligible=source_features(cfg,value); inv,evidence=derived_inventory(cfg,value,independent)
        verify=verification_features(local['probes'],len(value['rows']))
        vector,names=feature_sets(base,schema,inv,verify,independent['embedding'])
        if names!=plan['feature_schemas']: raise ValueError('Grounded feature schema drift')
        if not all(np.isfinite(v) for v in vector.values()): raise ValueError('Nonfinite router input')
        f={'plan_sha256':ph,'sample_id':sid,'image_sha256':value['sample']['image_sha256'],'values':vector,'eligible':eligible,
            'controls':{k:value['parts']['residual']['scores'][k] for k in cfg['controls']},
            'per_image_ground_truth_used':False,'actual_deepsolo_outputs_used':False,'frozen_choices_sha256':binding,
            'inventory_evidence':evidence,'source_probe_sha256':{'inventory':sha(root/split/'inventory'/name),'verification':sha(root/split/'verification'/name)}}
        seal(root/split/'features'/name,f); result[sid]=f
    return result


def run(args):
    warnings.simplefilter('error',ConvergenceWarning)
    cfg,plan,root,observed,branches=prepare(args)
    print(json.dumps({'status':'PREFLIGHT_PASS','root':str(root),'gpu':plan['gpu'],
        'feature_counts':{k:len(v) for k,v in plan['feature_schemas'].items()}}),flush=True)
    if args.preflight: return
    seal(root/'resolved_configuration.json',plan)
    for name,digest in plan['code_sha256'].items():
        path=root/'code_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and sha(path)!=digest: raise ValueError('Frozen code differs')
        if not path.exists(): path.write_bytes((REPO/name).read_bytes())
    torch.set_num_threads(cfg['execution']['cpu_threads']); seed=cfg['execution']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        inventory_stage(cfg,plan,root,'train',observed['train'])
        verification_stage(cfg,plan,root,'train',observed['train'])
        train=features(cfg,plan,root,'train',observed['train'])
        frozen,models,_=baseline.fit_and_freeze(cfg,plan,root,train,branches['train'])
        inventory_stage(cfg,plan,root,'test',observed['test'])
        verification_stage(cfg,plan,root,'test',observed['test'])
        test=features(cfg,plan,root,'test',observed['test'])
        result={'status':'SUCCESS','reproduction_mode':'approximate','claim_label':'approximate reproduction',
            'table_5_claim':False,'result_class':plan['result_class'],'plan_sha256':sha(root/'resolved_configuration.json'),
            'frozen_choices_sha256':sha(root/'frozen_choices.json'),'router_training':True,
            'base_models_frozen':True,'original_word_outputs_changed':False,'new_visual_model_inference':True,'splits':{}}
        for split,feature,source,ids in [('fit',train,branches['train'],plan['fit_ids']),
                ('validation',train,branches['train'],plan['validation_ids']),('train',train,branches['train'],sorted(train)),
                ('test',test,branches['test'],sorted(test))]:
            result['splits'][split],predictions=baseline.evaluate_split(cfg,plan,feature,source,ids,models,frozen)
            seal(root/'predictions'/(split+'.json'),predictions)
        result['paired_intervals']=baseline.intervals(cfg,result,branches['test'])
        seal(root/'results.json',result)
        print(json.dumps({'status':'SUCCESS','root':str(root),'results_sha256':sha(root/'results.json')}),flush=True)
    except Exception as error:
        # Preserve partial outputs and report every inference/model/OOM failure; never substitute.
        failure={'status':'FAILED','plan_sha256':sha(root/'resolved_configuration.json'),
            'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()}
        filename='failure_'+hashlib.sha256(json.dumps(failure,sort_keys=True).encode()).hexdigest()[:12]+'.json'
        seal(root/filename,failure)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--paths',type=Path,required=True)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--smoke',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--reuse-train-inventory',type=Path,help='Explicit qualified immutable TRAIN raw-output source; identical neural contract required')
    run(parser.parse_args())
