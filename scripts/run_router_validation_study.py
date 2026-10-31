"""Approximate matched router validation without refitting after calibration."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from scripts.run_yolo_f1_quality_study import read,seal,sha
from scripts.run_unet_token_router import freeze,wait_memory
from scripts.report_unet_token_router import verify_frozen_evidence
from scripts.unet_token_router_stages import load_features,light_rows,light_predictions
from scripts.token_context_data import annotations,key
from scripts.token_context_training import fit_router,load_model,score_model,point,ranked
from scripts.replay_router_improved_deepsolo import independent_route,metric_counts,independent_counts
from recognition.f1_quality_study import aggregate
from recognition.unet_token_routing import weak_word_labels,apply_threshold,fallback_counts
from recognition.router_validation import expert_holdout_ids,independent_partitions,validation_cuts,require_same_model

REPO=Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode!='approximate' or any(p.resolve().parent!=REPO/'configs/approximate' for p in (args.config,args.paths)):
        raise ValueError('Explicit isolated approximate configuration required')
    cfg=read(args.config);paths=read(args.paths);source=Path(paths['source_root'])
    if cfg['reproduction_mode']!='approximate' or paths['reproduction_mode']!='approximate':
        raise ValueError('Mode mismatch')
    for name,digest in cfg['source_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Qualified source changed: '+name)
    original=read(source/'resolved_configuration.json')
    evidence=verify_frozen_evidence(original,source)
    contract=read(paths['unet_contract'])
    if sha(paths['unet_contract'])!=original['config']['assets']['unet_contract']:
        raise ValueError('Validation split belongs to a different expert')
    if sha(paths['expert_manifest'])!=contract['resolved_config']['word_manifest_sha256']:
        raise ValueError('Expert training manifest changed')
    ordered=re.findall(r'^- id: (.+)$',Path(paths['expert_manifest']).read_text(),re.M)
    val=expert_holdout_ids(ordered,contract['resolved_config']['seed'],contract['resolved_config']['validation_count'],contract['validation_ids_sha256'])
    partitions=independent_partitions(original['samples']['train'],val,original['samples']['test'])
    model_cfg=json.loads(json.dumps(original['config']))
    model_cfg['training'].update(cfg['training']);model_cfg['execution'].update(cfg['execution'])
    if args.smoke:
        partitions={s:ids[:cfg['smoke'][s]] for s,ids in partitions.items()}
        model_cfg['training'].update(epochs=2,checkpoints=[1,2]);cfg['seeds']=cfg['seeds'][:1]
    model_cfg['validation_study']=cfg
    code={str(p.relative_to(REPO)):sha(p) for folder in original['config']['snapshot_packages'] for p in sorted((REPO/folder).rglob('*.py'))}
    plan=dict(reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='SMOKE_TEST' if args.smoke else 'APPROXIMATE_HELDOUT_VALIDATION_ROUTER',
        config=model_cfg,paths=paths,partitions=partitions,source_evidence_recheck=evidence,
        source_plan_sha256=sha(source/'resolved_configuration.json'),expert_manifest_sha256=sha(paths['expert_manifest']),
        unet_contract_sha256=sha(paths['unet_contract']),code_sha256=code,
        configuration_sha256=sha(args.config),paths_sha256=sha(args.paths),
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,numpy_version=np.__version__,
        deployment_protocol='Fit only fit IDs; select epoch and cuts on held-out validation with same weights; never refit after calibration',
        test_historically_explored=True,test_used_for_selection=False)
    tag=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return plan,Path(paths['output_root'])/('smoke' if args.smoke else 'full')/tag


def summary(branches,ids,accepted):
    p=point(branches,ids,accepted);p.update(fallback_counts(p));return p


def train_and_freeze(plan,root,original,source):
    if (root/'frozen_choices.json').exists():return
    if (root/'TEST_predictions.json').exists():raise ValueError('Cannot fit after TEST scoring')
    cfg=plan['config'];settings=cfg['validation_study'];parts=plan['partitions']
    features=load_features(original,source,'train');data=read(source/'train/branches.json')['branches']
    samples={s['sample_id']:s for s in original['samples']['train']};choices={}
    for model in cfg['backbones']:
        labels={s:dict(word_labels=weak_word_labels(light_rows(original,source,'train',model,s),annotations(samples[s]),cfg['labels'])) for s in parts['fit']}
        choices[model]={}
        for seed in settings['seeds']:
            name=model+'__seed'+str(seed);receipt=root/'training'/(name+'.json')
            if receipt.exists():found=read(receipt)['checkpoints']
            else:
                found=fit_router(plan,root,features[model],{},labels,data[model],parts['fit'],'token_context',name,seed,
                    cfg['training']['epochs'],parts['validation'])
                seal(receipt,dict(checkpoints=found,fitting_ids=parts['fit'],validation_ids=parts['validation']))
            candidates={e:summary(data[model],parts['validation'],ranked(v['scores'],parts['validation'],settings['primary_acceptance'])) for e,v in found.items()}
            epoch=max(candidates,key=lambda e:(candidates[e]['combined']['f1'],-int(e)))
            chosen=found[epoch]
            for v in found.values():
                if sha(root/v['model_path'])!=v['model_sha256']:raise ValueError('Candidate checkpoint changed')
            choices[model][str(seed)]=dict(epoch=int(epoch),model_path=chosen['model_path'],model_sha256=chosen['model_sha256'],
                validation_scores=chosen['scores'],validation_count_predictions=chosen['count_predictions'],candidate_points=candidates,
                policies=validation_cuts(chosen['scores'],settings['acceptance_targets'],chosen['model_sha256']))
            print(f'[VALIDATION freeze] {model} seed={seed} epoch={epoch}',flush=True)
    seal(root/'frozen_choices.json',dict(plan_sha256=sha(root/'resolved_configuration.json'),choices=choices,
        fit_ids=parts['fit'],validation_ids=parts['validation'],test_used_for_selection=False,refit_after_calibration=False))
    del features;gc.collect();torch.cuda.empty_cache()


def evaluate(plan,root,original,source):
    if (root/'results.json').exists():return
    frozen=read(root/'frozen_choices.json');cfg=plan['config'];parts=plan['partitions'];output={}
    predictions={m:{} for m in cfg['backbones']}
    for split,source_split in [('fit','train'),('validation','train'),('test','test')]:
        features=load_features(original,source,source_split)
        branches=read(source/source_split/'branches.json')['branches'];ids=parts[split];output[split]={}
        for model in cfg['backbones']:
            output[split][model]={}
            for seed,choice in frozen['choices'][model].items():
                actual_sha=sha(root/choice['model_path'])
                if actual_sha!=choice['model_sha256']:raise ValueError('Weights changed after calibration')
                network,preprocessing=load_model(plan,root,choice['model_path'])
                scores,counts=score_model(network,preprocessing,features[model],{},ids,cfg)
                if split=='validation' and scores!=choice['validation_scores']:
                    raise ValueError('Selected checkpoint does not replay its own validation predictions')
                del network;gc.collect();torch.cuda.empty_cache()
                predictions[model].setdefault(seed,{})[split]=dict(scores=scores,count_predictions=counts)
                thresholds=[]
                for policy in choice['policies']:
                    require_same_model(policy,actual_sha)
                    thresholds.append(dict(policy=policy,**summary(branches[model],ids,apply_threshold(scores,policy))))
                quotas={str(q):summary(branches[model],ids,ranked(scores,ids,q)) for q in cfg['validation_study']['acceptance_targets']}
                output[split][model][seed]=dict(thresholds=thresholds,descriptive_equal_quotas=quotas,
                    all_light=dict(metrics=aggregate(branches[model][s]['light'] for s in ids),deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.),
                    all_deepsolo=dict(metrics=aggregate(branches[model][s]['deepsolo'] for s in ids),deepsolo_images=len(ids),total_images=len(ids),deepsolo_percentage=100.))
        del features;gc.collect();torch.cuda.empty_cache()
    seal(root/'TEST_predictions.json',dict(frozen_choices_sha256=sha(root/'frozen_choices.json'),predictions=predictions))
    seal(root/'results.json',dict(status='SUCCESS',claim_label='approximate reproduction',reproduction_mode='approximate',
        result_class=plan['result_class'],splits=output,primary_seed=cfg['validation_study']['seeds'][0],
        plan_sha256=sha(root/'resolved_configuration.json'),frozen_choices_sha256=sha(root/'frozen_choices.json'),
        threshold_selection='Held-out VALIDATION only, with identical deployed weights',refit_after_calibration=False,
        routing_scope='Logical routing on cached experts; actual new DeepSolo inference=0'))


def audit(plan,root,source):
    frozen=read(root/'frozen_choices.json');d=read(root/'results.json');checks=0
    fit=set(plan['partitions']['fit']);validation=set(plan['partitions']['validation']);test=set(plan['partitions']['test'])
    if fit&validation or fit&test or validation&test:raise ValueError('Partition overlap')
    for model,seeds in frozen['choices'].items():
        for seed,c in seeds.items():
            p=root/c['model_path'];saved=torch.load(p,map_location='cpu',weights_only=True)
            if sha(p)!=c['model_sha256'] or set(saved['preprocessing']['training_ids'])!=fit or saved['preprocessing']['seed']!=int(seed):
                raise ValueError('Actual checkpoint training scope/identity mismatch')
            for policy in c['policies']:require_same_model(policy,sha(p));checks+=1
    for split,populations in d['splits'].items():
        branches=read(source/('test' if split=='test' else 'train')/'branches.json')['branches'];ids=plan['partitions'][split]
        for model,seeds in populations.items():
            for value in seeds.values():
                for p in [*value['thresholds'],*value['descriptive_equal_quotas'].values()]:
                    independent=independent_route(branches[model],ids,p['accepted_ids'])
                    if any(p['combined'][k]!=v for k,v in independent.items()):raise ValueError('Independent validation route audit failed')
                    if p['deepsolo_images']+p['accepted_images']!=len(ids):raise ValueError('Routing denominator mismatch')
                    checks+=1
    seal(root/'independent_audit.json',dict(status='PASS',checks=checks,results_sha256=sha(root/'results.json'),
        scope='Actual checkpoint training scope and immutable validation calibration; independent route counts'))


def execute(root):
    plan=read(root/'resolved_configuration.json');source=Path(plan['paths']['source_root']);original=read(source/'resolved_configuration.json')
    for name,digest in plan['config']['validation_study']['source_sha256'].items():
        if sha(source/name)!=digest:raise ValueError('Frozen source changed after preparation: '+name)
    if sha(plan['paths']['expert_manifest'])!=plan['expert_manifest_sha256'] or sha(plan['paths']['unet_contract'])!=plan['unet_contract_sha256']:
        raise ValueError('Frozen expert validation identity changed')
    cfg=plan['config'];torch.set_num_threads(cfg['execution']['cpu_threads'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    try:
        wait_memory(plan,'router');train_and_freeze(plan,root,original,source)
        evaluate(plan,root,original,source);audit(plan,root,source)
        seal(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),audit_sha256=sha(root/'independent_audit.json')))
    except BaseException as error:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error=str(error),traceback=traceback.format_exc(),
            out_of_memory=isinstance(error,torch.cuda.OutOfMemoryError),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode');p.add_argument('--config',type=Path);p.add_argument('--paths',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--run-root',type=Path);args=p.parse_args()
    if args.run_root:root=args.run_root.resolve()
    else:
        plan,root=prepare(args);print(json.dumps(dict(status='PREFLIGHT_PASS',root=str(root),populations={k:len(v) for k,v in plan['partitions'].items()})),flush=True)
        if args.preflight:return
        freeze(plan,root)
    if REPO==(root/'code_snapshot').resolve():execute(root)
    else:subprocess.run([sys.executable,'-u','-m','scripts.run_router_validation_study','--run-root',str(root)],
        cwd=root/'code_snapshot',env={**os.environ,'PYTHONPATH':str(root/'code_snapshot')},check=True)


if __name__=='__main__':main()
