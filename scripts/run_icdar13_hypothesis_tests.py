"""Matched APPROXIMATE vocabulary, crop, cross-fitting and TT-replay experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import joblib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from recognition.icdar13_hypothesis_tests import (
    crossfit_folds, vocabulary_rows, independent_counts, raw_evidence_features,
    gt_crop_diagnosis, normalize,
)
from recognition.icdar13_matched_router import router_rows, exact_accepted, validation_fallback_count
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.icdar_budget_router import observation
from recognition.lightweight_unet_router import fit_head, paired_utility
from recognition.f1_quality_study import aggregate
from recognition.trainable_image_router import oracle
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.icdar_budget_stages import score, route
from scripts.run_yolo_f1_quality_study import seal
from scripts.run_icdar13_matched_word_training import model, recognizer, read_rows
from scripts.train_icdar13_word_refit import save_checkpoint
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.finish_icdar13_word_refit import full_maps
from segmentation.icdar13_word_refit import WordDataset
from segmentation.icdar13_matched_training import emit_words, matched_word_loss
from segmentation.data import load_segmentation_manifest

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode != 'approximate' or args.config.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit APPROXIMATE namespace required')
    c=read(args.config)
    if c['mode'] != args.mode: raise ValueError('Mode mismatch')
    for p,d in c['pinned_inputs'].items():
        if sha(p)!=d: raise ValueError('Pinned input changed: '+p)
    light=read(Path(c['light_root'])/'resolved_configuration.json')
    source=light['paths']
    for name in ['tt_checkpoint','tt_contract']:
        if sha(source[name])!=source[name+'_sha256']:raise ValueError('Changed qualified initialization artifact: '+name)
    source_parent=read(source['parent_plan'])
    for name in ['parseq_archive','parseq_checkpoint']:
        if sha(source[name])!=source_parent['config']['pinned_inputs'][source[name]]:raise ValueError('Changed qualified PARSeq artifact: '+name)
    selected=read(Path(c['light_root'])/'selection_frozen.json')['winner']
    parent=read(Path(c['router_root'])/'resolved_configuration.json')
    records=light['train_records']+light['test_records']
    parts=copy.deepcopy(light['partitions'])
    if args.smoke:
        parts={k:v[:(8 if k=='fit' else 4)] for k,v in parts.items()}
    lookup={r['sample_id']:r for r in records}
    folds=crossfit_folds([lookup[s] for s in parts['fit']],2 if args.smoke else c['folds'],c['fold_seed'])
    ttroot=Path(c['tt_data_root']);manifest=load_segmentation_manifest(ttroot/c['tt_manifest'],require_checksums=True)
    fit,_,_= _split_indices(manifest,c['tt_split_seed'],c['tt_validation_count'])
    tt=[]
    for i in fit:
        s=asdict(manifest.samples[i]);r=dict(sample_id=s['sample_id'],split='fit')
        for dest,source in [('image','image'),('mask','mask'),('validity','validity_annotation'),('separator','separation_annotation'),('owners','word_owner_annotation')]:
            path=ttroot/str(s[source+'_path'])
            if sha(path)!=s[source+'_sha256']: raise ValueError('Changed TT source: '+str(path))
            r[dest+'_path']=str(path);r[dest+'_sha256']=s[source+'_sha256']
        with Image.open(r['image_path']) as im:
            rgb=np.asarray(im.convert('RGB'));r['image_wh']=list(im.size)
            r['rgb_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
        tt.append(r)
    if {r['rgb_sha256'] for r in tt}&{r['rgb_sha256'] for r in records}:
        raise ValueError('TT replay overlaps an IC13 population')
    groups=[{lookup[s]['rgb_sha256'] for s in ids} for ids in parts.values()]
    if any(a&b for i,a in enumerate(groups) for b in groups[i+1:]): raise ValueError('Population RGB overlap')
    for r in light['train_records']:
        for k in ['image','mask','validity','separator','owners']:
            if sha(r[k+'_path'])!=r[k+'_sha256']: raise ValueError('Changed IC target')
    root=Path(c['output_root'])/(args.run_name or ('smoke' if args.smoke else 'full'))
    root.mkdir(parents=True,exist_ok=False)
    code={str(f.relative_to(REPO)):sha(f) for folder in ['models','segmentation','scripts','recognition','localize','caption','scoring','pipeline','research'] for f in (REPO/folder).rglob('*.py')}
    plan=dict(mode='approximate',claim_label='approximate reproduction',config=c,light=light,
        selected=selected,parent_router=parent,records=records,partitions=parts,folds=folds,tt_records=tt,
        smoke=args.smoke,result_class='smoke-test result' if args.smoke else 'approximate reproduction',
        config_sha256=sha(args.config),code_sha256=code,epochs=1 if args.smoke else c['epochs'],
        historical_test_exposure=True,test_used_for_selection=False,
        hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),precision='float32',physical_batch=light['config']['training']['physical_batch_size'],
            mixed_sequential_domain_backward=True,image_hw=light['config']['image_hw'],parseq_batch=c['parseq_batch_size']))
    seal(root/'resolved_configuration.json',plan)
    for p,d in code.items():
        out=root/'code_snapshot'/p;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/p,out)
        if sha(out)!=d: raise ValueError('Source changed during snapshot')
    shutil.copytree(REPO/'configs',root/'code_snapshot/configs')
    return root


def lookup(plan): return {r['sample_id']:r for r in plan['records']}


def cached(plan,sid):
    p=Path(plan['config']['router_root'])/'experts'/(key(sid)+'.json');d=read(p)
    if 'raw_rows' not in d:
        p=Path(d['reused_source'])
        if sha(p)!=d['reused_source_sha256']: raise ValueError('Changed reused prediction')
        d=read(p)
    return d


def count_rows(rows,r):
    gt,valid=annotation(r)
    return independent_counts(predictions(rows,r,valid),gt)


def lexicon(plan):
    p=read(plan['config']['fallback_profile'])
    for k,path in p['lexicon_paths'].items():
        if sha(path)!=p['lexicon_sha256'][k]: raise ValueError('Changed Generic resource')
    return DeepSoloLexicon.from_files(p['lexicon_paths']['vocabulary'],p['lexicon_paths']['pairs'],distance_exclusive=p['distance_exclusive'])


def vocabulary_tests(plan,root):
    c=plan['config'];records=lookup(plan);lex=lexicon(plan);results={};details={}
    for split,ids in plan['partitions'].items():
        totals={p:[] for p in c['generic_policies']};changes={p:Counter() for p in c['generic_policies']}
        for sid in ids:
            r=records[sid];d=cached(plan,sid);emitted=emit_words(d['raw_rows'],plan['selected']['settings']['minimum_transcript_probability'])
            rows={p:vocabulary_rows(emitted,lex,p,c['conservative_vocabulary']) for p in c['generic_policies']}
            details[sid]={p:dict(rows=v,counts=count_rows(v,r)) for p,v in rows.items()}
            for p,v in rows.items():
                totals[p].append(details[sid][p]['counts'])
                changes[p].update(changed=sum(a['raw_text']!=b['raw_text'] for a,b in zip(emitted,v)))
        results[split]={p:dict(counts=aggregate(totals[p]),changes=dict(changes[p]),deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.) for p in totals}
    # Selection only; prefer unmodified then earlier predeclared policy on exact ties.
    chosen=max(c['generic_policies'],key=lambda p:results['selection'][p]['counts']['f1'])
    teacher=read(c['teacher_counts'])['samples'];ids=plan['partitions']['test']
    old=read(Path(c['router_root'])/'results.json');accepted=set(old['primary']['accepted_ids'])&set(ids)
    if plan['smoke']:
        accepted=set(ids[:len(ids)-validation_fallback_count(len(ids),233,c['primary_deepsolo_images'])])
    routes={}
    for policy in c['generic_policies']:
        b={s:dict(light=details[s][policy]['counts'],deepsolo=teacher[s]['counts']) for s in ids}
        upper=oracle(b,ids,ids,len(accepted))
        routes[policy]=dict(frozen_gate_control=route(b,ids,accepted),oracle=route(b,ids,upper['accepted_ids']),certificate=upper['certificate_max_residual'])
    seal(root/'vocabulary_details.json',details)
    seal(root/'vocabulary_results.json',dict(mode='approximate',claim_label='approximate reproduction',results=results,
        selected_policy=chosen,selection_split='selection',test_used_for_selection=False,routes=routes,
        actual_new_deepsolo_calls=0,scope='Same emitted words and boxes; fixed original gate decisions; CPU correction only'))
    print(json.dumps(dict(stage='vocabulary_complete',selected=chosen,test=results['test'][chosen])),flush=True)


def infer_sample(net,ocr,plan,r,out):
    c=plan['light']['config'];torch.cuda.reset_peak_memory_stats()
    prob=full_maps(net,r,c);settings=plan['selected']['settings']
    _,raw,expansion=read_rows(prob,r,settings,ocr,plan['config']['parseq_batch_size'])
    mp=out/'maps'/(key(r['sample_id'])+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(mp,probabilities=prob)
    rows=router_rows(raw,settings['minimum_transcript_probability'])
    d=dict(raw_rows=raw,rows=rows,maps_path=str(mp),maps_sha256=sha(mp),image_sha256=r['image_sha256'],
        counts=count_rows(rows,r),expansion=expansion,ground_truth_used_for_prediction=False,
        gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved())
    seal(out/'samples'/(key(r['sample_id'])+'.json'),d)
    return d


def evaluate_net(net,ocr,plan,ids,out):
    rec=lookup(plan);net.eval();counts=[]
    for sid in ids: counts.append(infer_sample(net,ocr,plan,rec[sid],out)['counts'])
    value=aggregate(counts);seal(out/'metrics.json',dict(counts=value,deepsolo_images=0,total_images=len(ids),deepsolo_percentage=0.))
    return value


def fit_net(plan,root,name,fit_ids,holdout_ids,mixed,ocr):
    out=root/name
    if (out/'complete.json').exists(): return read(out/'complete.json')
    if (out/'epochs.json').exists(): raise ValueError('Incomplete training exists; preserve it and explicitly restore a qualified checkpoint')
    c=plan['light']['config'];t=c['training'];seed=plan['config']['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    net=model(plan['light'],plan['light']['paths']['tt_checkpoint'])
    frozen_encoder={n:v.detach().cpu().clone() for n,v in net.state_dict().items() if n.startswith('encoder.')}
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=t['learning_rate'],weight_decay=t['weight_decay'],betas=tuple(t['adam_betas']),eps=t['adam_epsilon'])
    rec=lookup(plan);fit=WordDataset([rec[s] for s in fit_ids],c,augment=True)
    tt=WordDataset(plan['tt_records'],c,augment=True) if mixed else None
    evaluation_epochs=([0,1] if plan['smoke'] else plan['config']['mixed_validation_epochs']) if mixed else []
    best=-1.;best_epoch=None;history=[];out.mkdir(parents=True,exist_ok=True)
    if mixed:
        value=evaluate_net(net,ocr,plan,holdout_ids,out/'selection/epoch_000');best=value['f1'];best_epoch=0
        save_checkpoint(out/'best.pt',dict(model_state_dict=net.state_dict(),epoch=0,validation=value))
    start=time.monotonic()
    for epoch in range(1,plan['epochs']+1):
        fit.transform.set_epoch(epoch-1)
        loader=DataLoader(fit,batch_size=t['physical_batch_size'],num_workers=t['num_workers'],shuffle=True,
            generator=torch.Generator().manual_seed(seed+epoch),pin_memory=True)
        if mixed:
            tt.transform.set_epoch(epoch-1)
            replay=iter(DataLoader(tt,batch_size=t['physical_batch_size'],num_workers=t['num_workers'],shuffle=True,
                generator=torch.Generator().manual_seed(plan['config']['fold_seed']+epoch),pin_memory=True))
        net.train();seen=[];donors=[];total=0.;steps=0;torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            local={k:v.to(c['device']) if torch.is_tensor(v) else v for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss=matched_word_loss(net(local['image']),local,c,'totaltext_bce_dice')
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite IC loss')
            weight=plan['config']['tt_replay_weight'] if mixed else 0.
            ((1-weight)*loss).backward()
            if mixed:
                donor=next(replay)
                # Match the short last IC batch; target order/optimizer steps stay identical.
                size=len(batch['sample_id']);donors.extend(donor['sample_id'][:size])
                domain={k:v[:size].to(c['device']) if torch.is_tensor(v) else v[:size] for k,v in donor.items()}
                extra=matched_word_loss(net(domain['image']),domain,c,'totaltext_bce_dice')
                if not torch.isfinite(extra): raise FloatingPointError('Nonfinite TT replay loss')
                (weight*extra).backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()): raise FloatingPointError('Nonfinite gradient')
            opt.step();seen.extend(batch['sample_id']);total+=float(loss.detach())*len(batch['sample_id']);steps+=1
        if sorted(seen)!=sorted(fit_ids) or set(fit_ids)&set(holdout_ids): raise ValueError('Fitting/holdout exposure violation')
        row=dict(epoch=epoch,fit_samples=len(seen),tt_samples=len(donors),optimizer_steps=steps,ic_loss=total/len(seen),
            fit_order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
            tt_order_sha256=hashlib.sha256(('\n'.join(donors)+'\n').encode()).hexdigest(),
            gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved())
        if epoch in evaluation_epochs:
            value=evaluate_net(net,ocr,plan,holdout_ids,out/('selection/epoch_'+str(epoch).zfill(3)));row['selection']=value
            if value['f1']>best:
                best=value['f1'];best_epoch=epoch;save_checkpoint(out/'best.pt',dict(model_state_dict=net.state_dict(),epoch=epoch,validation=value))
        history.append(row);(out/'epochs.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(dict(row,stage='training',model=name,epochs=plan['epochs'],seconds=time.monotonic()-start)),flush=True)
    save_checkpoint(out/'last.pt',dict(model_state_dict=net.state_dict(),epoch=plan['epochs'],fit_ids=fit_ids,excluded_ids=holdout_ids))
    if not mixed:
        evaluate_net(net,ocr,plan,holdout_ids,out/'heldout')
    else:
        del net;torch.cuda.empty_cache();net=model(plan['light'],out/'best.pt')
        for split in ['calibration','test']:evaluate_net(net,ocr,plan,plan['partitions'][split],out/split)
    state=net.state_dict()
    if any(not torch.equal(state[n].cpu(),v) for n,v in frozen_encoder.items()): raise ValueError('Frozen encoder or BN state changed')
    value=dict(status='COMPLETE',fit_ids=fit_ids,excluded_ids=holdout_ids,epochs=plan['epochs'],
        frozen_encoder_tensors_verified=len(frozen_encoder),source_checkpoint_sha256=plan['light']['paths']['tt_checkpoint_sha256'],
        selected_epoch=best_epoch if mixed else plan['epochs'],checkpoint_sha256=sha(out/('best.pt' if mixed else 'last.pt')),
        peak_allocated=max(v['gpu_allocated'] for v in history),peak_reserved=max(v['gpu_reserved'] for v in history))
    seal(out/'complete.json',value);del net,opt;torch.cuda.empty_cache()
    return value


def crop_tests(plan,root,ocr):
    rec=lookup(plan);result={};c=plan['config'];torch.cuda.reset_peak_memory_stats()
    for split in ['selection','calibration','test']:
        details=[];totals={str(p):Counter() for p in c['gt_crop_padding']}
        for sid in plan['partitions'][split]:
            r=rec[sid];gt,_=annotation(r);d=cached(plan,sid);emitted=emit_words(d['raw_rows'],plan['selected']['settings']['minimum_transcript_probability'])
            with Image.open(r['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            for pad in c['gt_crop_padding']:
                crops=[]
                for g in gt:
                    x1,y1,x2,y2=g['bbox'];x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(rgb.shape[1],x2+pad);y2=min(rgb.shape[0],y2+pad)
                    crops.append(np.ascontiguousarray(rgb[y1:y2,x1:x2]))
                readings=[]
                for start in range(0,len(crops),c['parseq_batch_size']):
                    readings.extend(v.to_dict() for v in ocr.recognize_candidate_batch(crops[start:start+c['parseq_batch_size']]))
                diagnoses=gt_crop_diagnosis(gt,d['raw_rows'],emitted,readings,c['crop_iou_threshold'])
                for row in diagnoses:
                    totals[str(pad)].update(gt_words=1,gt_crop_correct=int(row['gt_crop_correct']))
                    totals[str(pad)].update({row['group']:1,row['group']+'_gt_readable':int(row['gt_crop_correct'])})
                details.append(dict(sample_id=sid,padding=pad,words=diagnoses,gt_readings=readings))
        seal(root/'gt_crops'/(split+'.json'),details)
        result[split]={p:dict(v,identity_accuracy=v['gt_crop_correct']/v['gt_words'],deepsolo_images=0,total_images=len(plan['partitions'][split]),deepsolo_percentage=0.) for p,v in totals.items()}
    seal(root/'crop_results.json',dict(mode='approximate',claim_label='approximate reproduction',scope='Privileged fixed-GT-identity diagnostic; not a deployable score or absolute ceiling',results=result,
        gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved(),model='Qualified PARSeq',precision='float32',batch=c['parseq_batch_size']))
    print(json.dumps(dict(stage='crop_diagnosis_complete',test=result['test'])),flush=True)


def features_and_branches(plan,root,ids,source):
    c=plan['config'];teacher=read(c['teacher_counts'])['samples'];rec=lookup(plan);features={};branches={};keys={}
    for sid in ids:
        r=rec[sid]
        if source=='oof' and sid in plan['partitions']['fit']:
            index=next(i for i,f in enumerate(plan['folds']) if sid in f);d=read(root/f'fold_{index}'/'heldout/samples'/(key(sid)+'.json'));mp=d['maps_path']
        else:
            archive=read(Path(c['router_root'])/'experts'/(key(sid)+'.json'));mp=archive['maps_path'];d=cached(plan,sid)
        with np.load(mp) as z:prob=z['probabilities']
        raw=d['raw_rows'];rows=router_rows(raw,plan['selected']['settings']['minimum_transcript_probability'])
        groups=observation(rows,r['image_wh'],prob,'square_letterbox',settings=c['features']);groups['raw']=raw_evidence_features(raw,rows)
        features[sid]=dict(groups=groups,scale=len(rows)+1,eligible=bool(rows));keys[sid]=r['rgb_sha256']
        branches[sid]=dict(light=count_rows(rows,r),deepsolo=teacher[sid]['counts'])
        if branches[sid]['light']['gt_words']!=branches[sid]['deepsolo']['gt_words']: raise ValueError('Expert GT mismatch')
    return features,branches,keys


def train_gates(plan,root):
    c=plan['config'];p=plan['partitions'];ids=p['fit']+p['selection']+p['calibration']+p['test'];result={}
    for arm,source,raw in [('seen_control','seen',False),('crossfit_control','oof',False),('crossfit_raw_no_guard','oof',True)]:
        features,branches,keys=features_and_branches(plan,root,ids,source)
        if raw:
            for f in features.values():f['eligible']=True
        train=p['fit'];reference=aggregate(branches[s]['deepsolo'] for s in train)['f1']
        target=np.array([paired_utility(branches[s],reference)/features[s]['scale'] for s in train]);weights=np.array([features[s]['scale'] for s in train]);trials=[]
        groupsets=[['ocr','tokens','maps','raw']] if raw else c['feature_groups']
        directory=root/'routers'/arm;directory.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(directory/'features.npz',ids=np.asarray(ids),targets=target,weights=weights,
            eligible=np.array([features[s]['eligible'] for s in ids]),scale=np.array([features[s]['scale'] for s in ids]),
            **{g:np.stack([features[s]['groups'][g] for s in ids]) for g in features[ids[0]]['groups']})
        seal(directory/'observations.json',dict(branches=branches,tie_keys=keys,fit_ids=train,
            selection_ids=p['selection'],calibration_ids=p['calibration'],test_ids=p['test'],source=source))
        val_n=validation_fallback_count(len(p['selection']),233,c['primary_deepsolo_images'])
        for groups in groupsets:
            x=np.stack([np.concatenate([features[s]['groups'][g] for g in groups]) for s in train])
            for index,recipe in enumerate(c['heads']):
                bundle=fit_head(groups,x,target,weights,recipe,c['seed']);name='_'.join(groups)+'_'+str(index)+'.joblib';joblib.dump(bundle,directory/name)
                scores=score(bundle,features,p['selection']);accepted=exact_accepted(scores,{s:keys[s] for s in p['selection']},val_n);v=route(branches,p['selection'],accepted)
                trials.append(dict(groups=groups,recipe=recipe,scalars=bundle['stored_scalar_count'],bytes=(directory/name).stat().st_size,path=name,sha256=sha(directory/name),selection=v))
        best=max(v['selection']['combined']['f1'] for v in trials)
        winner=min([v for v in trials if v['selection']['combined']['f1']>=best-c['selection_tolerance']],key=lambda v:(v['scalars'],v['bytes'],-v['selection']['combined']['f1'],v['path']))
        seal(directory/'selection_frozen.json',dict(winner=winner,trials=trials,test_used_for_selection=False))
        bundle=joblib.load(directory/winner['path']);test_scores=score(bundle,features,p['test']);n=validation_fallback_count(len(p['test']),233,c['primary_deepsolo_images'])
        accepted=exact_accepted(test_scores,{s:keys[s] for s in p['test']},n);test=route(branches,p['test'],accepted)
        fits={split:dict(counts=aggregate(branches[s]['light'] for s in p[split]),deepsolo_images=0,total_images=len(p[split]),deepsolo_percentage=0.) for split in p}
        result[arm]=dict(winner=winner,test=test,light_populations=fits)
        seal(directory/'test_scores.json',dict(scores=test_scores,branches={s:branches[s] for s in p['test']},tie_keys={s:keys[s] for s in p['test']}))
        print(json.dumps(dict(stage='router_complete',arm=arm,test=test['combined'],deepsolo_images=n,total_images=len(p['test']))),flush=True)
    seal(root/'router_results.json',dict(mode='approximate',claim_label='approximate reproduction',results=result,actual_new_deepsolo_calls=0,test_used_for_selection=False))


def execute(root,stage):
    plan=read(root/'resolved_configuration.json');c=plan['config']
    if plan['mode']!='approximate': raise ValueError('Approximate only')
    for path,digest in plan['code_sha256'].items():
        if sha(REPO/path)!=digest: raise ValueError('Frozen scientific source changed: '+path)
    for path,digest in c['pinned_inputs'].items():
        if sha(path)!=digest: raise ValueError('Pinned input changed: '+path)
    torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    if stage in ['cpu','all'] and not (root/'vocabulary_results.json').exists():vocabulary_tests(plan,root)
    if stage in ['gpu','all']:
        if torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']: raise RuntimeError('Insufficient free GPU memory')
        ocr=recognizer(plan['light'])
        try:
            if not (root/'crop_results.json').exists():crop_tests(plan,root,ocr)
            for i,heldout in enumerate(plan['folds']):
                fit=[s for s in plan['partitions']['fit'] if s not in heldout]
                fit_net(plan,root,f'fold_{i}',fit,heldout,False,ocr)
            fit_net(plan,root,'mixed_totaltext',plan['partitions']['fit'],plan['partitions']['selection'],True,ocr)
        finally:ocr.release()
    if stage in ['routers','all'] and not (root/'router_results.json').exists():train_gates(plan,root)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path);p.add_argument('--run-root',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--run-name',help='New isolated output directory name; never overwrites an existing run')
    p.add_argument('--stage',choices=['prepare','cpu','gpu','routers','all'],default='all');args=p.parse_args()
    if args.run_root is None and args.config is None:p.error('Provide --config to prepare a new run or --run-root for its frozen snapshot')
    root=args.run_root or prepare(args)
    try:
        if args.stage!='prepare':execute(root,args.stage)
    except BaseException as exc:
        path=root/('failure_'+str(time.time_ns())+'.json');path.write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
    print(json.dumps(dict(stage=args.stage,run_root=str(root),status='SUCCESS')),flush=True)


if __name__=='__main__':main()
