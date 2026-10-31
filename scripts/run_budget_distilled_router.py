"""Approximate DeepSolo-supervised cheap gate; validation choices precede TEST."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from sklearn.decomposition import PCA

from recognition.budget_distilled_router import (observation_features,paired_target,fit_small_head,
    predict_small_head,model_cost,rank_accept,predict_observation)
from recognition.router_validation import validation_cuts,require_same_model
from recognition.unet_token_routing import apply_threshold,fallback_counts
from recognition.trainable_image_router import oracle
from scripts.run_yolo_f1_quality_study import read,seal,sha
from scripts.run_unet_token_router import freeze
from scripts.token_context_data import key,rgb
from scripts.unet_token_router_stages import light_predictions
from scripts.token_context_training import point
from scripts.replay_router_improved_deepsolo import metric_counts,independent_counts,independent_route

REPO=Path(__file__).resolve().parents[1]


def log(stage,**values):print(json.dumps(dict(stage=stage,**values)),flush=True)


def summary(branches,ids,accepted):
    out=point(branches,ids,accepted);out.update(fallback_counts(out));return out


def prepare(args):
    if args.mode!='approximate' or any(p.resolve().parent!=REPO/'configs/approximate' for p in (args.config,args.paths)):
        raise ValueError('Explicit isolated approximate configurations required')
    cfg=read(args.config);paths=read(args.paths)
    if cfg['reproduction_mode']!='approximate' or paths['reproduction_mode']!='approximate':raise ValueError('Mode mismatch')
    source=Path(paths['source_root']);validation=Path(paths['validation_root'])
    for scope,root in [('source',source),('validation',validation)]:
        for name,digest in cfg['inputs_sha256'][scope].items():
            if sha(root/name)!=digest:raise ValueError('Qualified input changed: '+str(root/name))
    original=read(source/'resolved_configuration.json');vplan=read(validation/'resolved_configuration.json')
    parts=vplan['partitions']
    if cfg.get('expert_unseen_adaptation'):
        adaptation=cfg['expert_unseen_adaptation'];held=parts['validation']
        ranked=sorted(held,key=lambda s:(hashlib.sha256((str(cfg['seed'])+':adapt:'+s).encode()).digest(),s))
        added=ranked[:adaptation['training_count']]
        if not added or len(added)>=len(held):raise ValueError('Adaptation requires separate expert-unseen training and validation')
        cfg['expert_unseen_adaptation']['training_ids']=sorted(added)
        parts=dict(fit=sorted(added if adaptation.get('expert_unseen_only') else parts['fit']+added),validation=sorted(set(held)-set(added)),test=parts['test'])
    if args.smoke:parts={s:ids[:cfg['smoke'][s]] for s,ids in parts.items()}
    by_id={s['sample_id']:s for split in original['samples'].values() for s in split}
    sets=[{by_id[s]['rgb_sha256'] for s in parts[k]} for k in ('fit','validation','test')]
    if any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):raise ValueError('RGB split leakage')
    if sha(paths['unet_checkpoint'])!=original['config']['assets']['unet_checkpoint']:
        raise ValueError('Different U-Net checkpoint')
    for split in ('train','test'):
        receipt=read(source/split/'unet_receipt.json')
        for relative,digest in receipt['files_sha256'].items():
            if sha(source/relative)!=digest:raise ValueError('Source OCR/map shard changed: '+relative)
    code={str(p.relative_to(REPO)):sha(p) for folder in original['config']['snapshot_packages'] for p in sorted((REPO/folder).rglob('*.py'))}
    plan=dict(reproduction_mode='approximate',claim_label='approximate reproduction',config=cfg,paths=paths,
        result_class='SMOKE_TEST' if args.smoke else 'APPROXIMATE_RESEARCH_EXTENSION',
        source_plan_sha256=sha(source/'resolved_configuration.json'),partitions=parts,code_sha256=code,
        config_sha256=sha(args.config),paths_sha256=sha(args.paths),source_unet=original['config']['unet'],
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,numpy_version=np.__version__,
        test_used_for_selection=False,test_historically_explored=True,
        snapshot_protocol='Executed immutable source; validation-only candidate selection and same-weight cuts; no refit',
        actual_new_deepsolo_inference=0)
    # The existing snapshot helper reads this namespace without scientific fallback.
    plan['config']['snapshot_packages']=original['config']['snapshot_packages']
    tag=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()[:12]
    return plan,Path(paths['output_root'])/('smoke' if args.smoke else 'full')/tag


def shared_context_stage(plan,root,split):
    from models.adaspot_gap import AdaSpotGapAGUNet
    from segmentation.word_slot_data import image_tensor
    from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
    from scripts.contextual_lm_verifier_runtime import state_digest
    from PIL import Image
    p=root/(split+'_shared.npz');receipt=root/(split+'_shared_receipt.json')
    if receipt.exists():
        if sha(p)!=read(receipt)['npz_sha256']:raise ValueError('Shared context cache changed')
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    cfg=plan['config'];source=Path(plan['paths']['source_root']);original=read(source/'resolved_configuration.json')
    source_split='test' if split=='test' else 'train'
    ids=plan['partitions']['test'] if split=='test' else plan['partitions']['fit']+plan['partitions']['validation']
    samples={s['sample_id']:s for s in original['samples'][source_split]}
    if split=='test' and not (root/'frozen_choices.json').exists():raise ValueError('Freeze model before TEST context extraction')
    if 'shared_feature_source_root' in plan['paths']:
        prior=Path(plan['paths']['shared_feature_source_root']);name=split+'_shared.npz'
        for relative,digest in cfg['shared_cache_sha256'].items():
            if sha(prior/relative)!=digest:raise ValueError('Pinned shared context changed')
        source_receipt=read(prior/(split+'_shared_receipt.json'))
        if sha(prior/name)!=source_receipt['npz_sha256']:raise ValueError('Unqualified pooled context cache')
        with np.load(prior/name,allow_pickle=False) as z:
            if not set(ids)<=set(z.files):raise ValueError('Shared context population changed')
            result={k:z[k] for k in ids}
        np.savez_compressed(p,**result)
        seal(receipt,dict(source_receipt,npz_sha256=sha(p),images=len(ids),source_images=source_receipt['images'],reused_source_root=str(prior),actual_new_unet_forwards=0))
        return result
    device=cfg['execution']['device']
    if device.startswith('cuda') and torch.cuda.mem_get_info()[0]<cfg['execution']['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free GPU memory for frozen U-Net context extraction')
    saved=torch.load(plan['paths']['unet_checkpoint'],map_location='cpu',weights_only=False)
    if saved['run_contract']!=read(original['paths']['unet_contract']):raise ValueError('Expert contract mismatch')
    model=AdaSpotGapAGUNet('control',skip_floor=plan['source_unet']['skip_floor'],initial_gap_bias=plan['source_unet']['initial_gap_bias'])
    model.load_state_dict(saved['model_state_dict'],strict=True);del saved
    model.requires_grad_(False).eval().to(device);before=state_digest(model)
    if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    rng=np.random.default_rng(cfg['seed']);projection=torch.tensor(rng.choice([-1.,1.],(2048,cfg['shared_context']['projection_dim']))/math.sqrt(2048),dtype=torch.float32,device=device)
    encoded=[];handle=model.encoder.layer4.register_forward_hook(lambda module,args,output:encoded.append(output))
    output={};errors=[];pool_seconds=[];started=time.monotonic()
    try:
        for index,sid in enumerate(ids,1):
            sample=samples[sid];encoded.clear()
            image=rgb(sample)
            with torch.inference_mode():
                evidence=model.forward_evidence(image_tensor(Image.fromarray(image)).to(device))
                predicted=evidence['words'][0].sigmoid();feature=evidence['features'][0];encoder=encoded[0][0]
                with np.load(source/source_split/'unet_maps'/(key(sid)+'.npz'),allow_pickle=False) as z:
                    cached=z['probabilities']
                error=float(np.max(np.abs(predicted.cpu().numpy()-cached)));errors.append(error)
                if error>cfg['shared_context']['map_replay_tolerance']:raise ValueError('Fresh frozen U-Net maps do not replay: '+str(error))
                if device.startswith('cuda'):torch.cuda.synchronize()
                tick=time.perf_counter()
                canvas=torch.tensor(np.asarray(letterbox_canvas_mask(tuple(sample['image_wh']),native_size=predicted.shape[-1]),bool),device=device)
                pools=[feature[:,canvas].mean(1),feature[:,canvas].std(1,unbiased=False)]
                for weights in predicted:
                    weights=weights*canvas
                    pools.append((feature*weights[None]).sum((1,2))/weights.sum().clamp_min(1e-12))
                encoder_pools=[encoder.mean((1,2))]
                hh,ww=encoder.shape[-2:]
                for y in range(2):
                    for x in range(2):encoder_pools.append(encoder[:,y*hh//2:(y+1)*hh//2,x*ww//2:(x+1)*ww//2].mean((1,2)))
                pools.extend(v@projection for v in encoder_pools)
                context=torch.cat(pools).cpu().numpy()
                if device.startswith('cuda'):torch.cuda.synchronize()
                pool_seconds.append(time.perf_counter()-tick)
                output[sid]=context
            if index%cfg['execution']['log_every']==0 or index==len(ids):log('shared_context',split=split,images=index,total=len(ids),seconds=time.monotonic()-started)
        after=state_digest(model)
        if after!=before:raise ValueError('Frozen U-Net state changed')
        peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()) if device.startswith('cuda') else None
        np.savez_compressed(p,**output)
        seal(receipt,dict(status='PASS',images=len(ids),npz_sha256=sha(p),state_before=before,state_after=after,
            input_shape=[1,3,512,512],batch=1,precision='float32',device=device,cuda_peak=peak,
            maximum_probability_replay_error=max(errors),pooling_median_ms=1000*float(np.median(pool_seconds)),
            pooling_p95_ms=1000*float(np.quantile(pool_seconds,.95)),seconds=time.monotonic()-started,
            pooling_scope='Additional shared-feature pooling only; U-Net forward and OCR excluded',strict_checkpoint_loading=True))
    finally:
        handle.remove();del model;gc.collect()
        if device.startswith('cuda'):torch.cuda.empty_cache()
    return output


def training_feature_stage(plan,root,shared):
    cfg=plan['config'];source=Path(plan['paths']['source_root']);original=read(source/'resolved_configuration.json')
    raw=read(source/'train/branches.json');branches=raw['branches']['unet'];records=raw['records']
    samples={s['sample_id']:s for s in original['samples']['train']};ids=plan['partitions']['fit']+plan['partitions']['validation']
    reference=sum(branches[s]['deepsolo']['tp'] for s in plan['partitions']['fit'])*2/sum(branches[s]['deepsolo']['detected_words']+branches[s]['deepsolo']['gt_words'] for s in plan['partitions']['fit'])
    features={};views=[];names=None;groups=None;started=time.monotonic()
    for i,sid in enumerate(ids,1):
        sample=samples[sid];d=read(source/'train/unet'/(key(sid)+'.json'));rows=d['rows'];candidates=d['candidates']
        with np.load(source/'train/unet_maps'/(key(sid)+'.npz'),allow_pickle=False) as z:p=z['probabilities']
        value,schema,family=observation_features(rows,candidates,sample['image_wh'],p,cfg['features'])
        if names is None:names,groups=schema,family
        if names!=schema or groups!=family:raise ValueError('Feature schema changed')
        features[sid]=dict(x=value,shared=shared[sid],scale=math.sqrt(len(rows)+1),eligible=bool(rows) and all(r['raw_text'].strip() for r in rows))
        if sid not in plan['partitions']['fit']:continue
        native=light_predictions(candidates,sample,original['config']['metric']['native_ignore_size'])
        lookup={r['candidate_id']:pred for r,pred in zip(candidates,native)}
        def counted(selected):
            pp=[lookup[r['candidate_id']] for r in selected];gt=records[sid]['ground_truth']
            a=metric_counts(pp,gt,'casefold_alnum')
            if a!=independent_counts(pp,gt,'casefold_alnum'):raise ValueError('Synthetic-view independent metric disagreement')
            return a
        if counted(rows)!=branches[sid]['light']:raise ValueError('Primary expert endpoint changed')
        adaptation=cfg.get('expert_unseen_adaptation',{});image_weight=adaptation.get('weight',1.) if sid in adaptation.get('training_ids',[]) else 1.
        views.append(dict(sid=sid,x=value,scale=math.sqrt(len(rows)+1),target=paired_target(branches[sid]['light'],branches[sid]['deepsolo'],reference),weight=image_weight,real=True,
                          candidate_ids=[r['candidate_id'] for r in rows],light_counts=branches[sid]['light']))
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256((str(cfg['seed'])+sid).encode()).digest()[:8],'little'))
        rejected=[r for r in candidates if r['candidate_id'] not in {q['candidate_id'] for q in rows}]
        for drop in cfg['augmentation']['drop_probabilities']:
            selected=[r for r in rows if rng.random()>=drop]
            selected += [r for r in rejected if rng.random()<cfg['augmentation']['add_rejected_probability']]
            x,names2,_=observation_features(selected,candidates,sample['image_wh'],p,cfg['features'])
            if names2!=names:raise ValueError('Augmented observation schema changed')
            counts=counted(selected)
            views.append(dict(sid=sid,x=x,scale=math.sqrt(len(selected)+1),target=paired_target(counts,branches[sid]['deepsolo'],reference),
                              weight=image_weight*cfg['augmentation']['total_weight']/len(cfg['augmentation']['drop_probabilities']),real=False,
                              candidate_ids=[r['candidate_id'] for r in selected],light_counts=counts))
        if (i+1)%cfg['execution']['log_every']==0:log('training_observations',images=i+1,total=len(ids),seconds=time.monotonic()-started)
    seal(root/'feature_schema.json',dict(names=names,groups=groups,inference_ground_truth_used=False,reference_f1=reference))
    archive=root/'training_observations.npz'
    np.savez_compressed(archive,x=np.stack([v['x'] for v in views]),
        target=np.array([v['target'] for v in views]),scale=np.array([v['scale'] for v in views]),
        weights=np.array([v['weight'] for v in views]),real=np.array([v['real'] for v in views]),
        parent_ids=np.array([v['sid'] for v in views]),population_ids=np.array(ids),
        population_x=np.stack([features[s]['x'] for s in ids]),shared=np.stack([features[s]['shared'] for s in ids]))
    seal(root/'training_observations.json',dict(npz_sha256=sha(archive),views=[{k:v for k,v in row.items() if k!='x'} for row in views],
         reference_f1=reference,annotation_scope='Only fit IDs produce labels; validation/TEST never produce augmentation labels'))
    return features,views,branches,reference,names,groups


def fit_and_freeze(plan,root,features,views,branches,reference,names,groups):
    cfg=plan['config'];fit=plan['partitions']['fit'];validation=plan['partitions']['validation']
    n=min(cfg['shared_context']['pca_dimensions'],len(fit)-1)
    pca=PCA(n_components=n,svd_solver='full').fit(np.stack([features[s]['shared'] for s in fit]))
    projection=dict(mean=pca.mean_.tolist(),components=pca.components_.tolist(),fitting_ids=fit)
    trials=[];best=None;started=time.monotonic()
    for feature_set in cfg['feature_sets']:
        indices=[i for i,g in enumerate(groups) if g=='ocr' or feature_set!='ocr']
        use_shared=feature_set=='shared_context'
        def matrix(vs):
            rows=[]
            for v in vs:
                x=v['x'][indices]
                if use_shared:x=np.r_[x,(features[v['sid']]['shared']-pca.mean_)@pca.components_.T]
                rows.append(x)
            return np.stack(rows)
        val_views=[dict(sid=s,x=features[s]['x']) for s in validation];xv=matrix(val_views)
        for augmented in cfg['augmentation']['arms']:
            fitting=[v for v in views if augmented or v['real']];x=matrix(fitting)
            y=np.array([v['target']/v['scale'] for v in fitting]);weights=np.array([v['weight'] for v in fitting])
            for ri,recipe in enumerate(cfg['recipes']):
                name=f'{feature_set}__aug{int(augmented)}__r{ri}';tick=time.monotonic()
                head=fit_small_head(x,y,weights,recipe,cfg['seed']);scores0=predict_small_head(head,xv)
                scores={s:float(q*features[s]['scale']) if features[s]['eligible'] else None for s,q in zip(validation,scores0)}
                points={str(q):summary(branches,validation,rank_accept(scores,validation,q)) for q in cfg['selection_quotas']}
                objective=float(np.mean([p['combined']['f1'] for p in points.values()]))
                bundle=dict(head=head,features=cfg['features'],schema=names,feature_indices=indices,
                    shared_projection=projection if use_shared else None,feature_set=feature_set,
                    training_ids=fit,validation_ids=validation,augmented_training=augmented,reference_f1=reference,
                    training_views=len(fitting),target='Paired exact lexical utility divided by sqrt(selected rows+1)',
                    deployment_inputs='Predicted OCR/candidates, image dimensions, predicted maps and optional same-forward U-Net pooling')
                path=root/'candidates'/(name+'.json');seal(path,bundle)
                cost=model_cost(head);trial=dict(name=name,path=str(path.relative_to(root)),sha256=sha(path),validation_scores=scores,
                    validation_points=points,selection_objective=objective,feature_set=feature_set,augmented=augmented,
                    recipe=recipe,feature_dimension=x.shape[1],model_cost=cost,serialized_bytes=path.stat().st_size,
                    training_seconds=time.monotonic()-tick)
                trials.append(trial)
                # Select by validation F1; exact ties prefer no shared pooling and smaller head.
                quality=(objective,-int(use_shared),-path.stat().st_size,-len(trials))
                if best is None or quality>best[0]:best=(quality,trial,bundle)
                log('fit',name=name,validation_f1=100*objective,seconds=time.monotonic()-tick)
    _,choice,deployment=best
    # Bind scalar cuts to the exact fitted model, then serialize a deployment bundle.
    policies=validation_cuts(choice['validation_scores'],cfg['acceptance_targets'],choice['sha256'])
    deployment=dict(deployment,policies={str(p['validation_target_fraction']):p for p in policies},head_sha256=choice['sha256'])
    seal(root/'deployment.json',deployment)
    seal(root/'trials.json',dict(trials=trials,selection_quotas=cfg['selection_quotas'],test_used_for_selection=False))
    seal(root/'frozen_choices.json',dict(choice=choice,policies=policies,candidate_sha256=choice['sha256'],deployment_sha256=sha(root/'deployment.json'),
        fit_ids=fit,validation_ids=validation,projection=projection,refit_after_calibration=False,test_used_for_selection=False,seconds=time.monotonic()-started))
    log('FROZEN',choice=choice['name'],validation_f1=100*choice['selection_objective'])


def evaluate(plan,root,shared,features,branches):
    cfg=plan['config'];source=Path(plan['paths']['source_root']);original=read(source/'resolved_configuration.json')
    frozen=read(root/'frozen_choices.json');deployment=read(root/'deployment.json');choice=frozen['choice']
    if sha(root/choice['path'])!=choice['sha256'] or sha(root/'deployment.json')!=frozen['deployment_sha256']:raise ValueError('Model changed after calibration')
    original_test=read(source/'test/branches.json');branches_test=original_test['branches']['unet']
    samples={s['sample_id']:s for s in original['samples']['test']};ids=plan['partitions']['test'];scores={};timings=[];feature_timings=[]
    per_image={};checks=0
    for sid in ids:
        sample=samples[sid];data=read(source/'test/unet'/(key(sid)+'.json'))
        with np.load(source/'test/unet_maps'/(key(sid)+'.npz'),allow_pickle=False) as z:p=z['probabilities']
        tick=time.perf_counter();actual=predict_observation(deployment,data['rows'],data['candidates'],sample['image_wh'],p,shared.get(sid));timings.append(time.perf_counter()-tick)
        scores[sid]=actual['score'];per_image[sid]=actual
        checks+=1
    output={}
    for split,data,ss,features0 in [('validation',branches,frozen['choice']['validation_scores'],features),('test',branches_test,scores,None)]:
        members=plan['partitions'][split];policies=[];quotas={};oracles={}
        for policy in frozen['policies']:
            require_same_model(policy,choice['sha256'])
            selected=apply_threshold(ss,policy);result=summary(data,members,selected)
            if any(result['combined'][k]!=v for k,v in independent_route(data,members,selected).items()):raise ValueError('Independent route mismatch')
            checks+=1;policies.append(dict(policy=policy,**result))
        for q in cfg['acceptance_targets']:
            selected=rank_accept(ss,members,q);quotas[str(q)]=summary(data,members,selected)
            eligible=[s for s in members if ss[s] is not None]
            privileged=oracle({s:data[s] for s in members},members,eligible,len(selected))
            oracles[str(q)]=dict(**summary(data,members,privileged['accepted_ids']),optimality=privileged)
            checks+=1
        output[split]=dict(thresholds=policies,predeclared_batch_quotas=quotas,conditional_gt_oracles=oracles,
            all_light=dict(metrics=summary(data,members,members)['combined'],deepsolo_images=0,total_evaluated_images=len(members),deepsolo_percentage=0.),
            all_deepsolo=dict(metrics=summary(data,members,[])['combined'],deepsolo_images=len(members),total_evaluated_images=len(members),deepsolo_percentage=100.))
    # Frozen existing token controls use the identical expert outputs and TEST IDs.
    baseline=read(Path(plan['paths']['validation_root'])/'TEST_predictions.json')['predictions']
    controls={}
    for backbone in ('unet','yolo'):
        seed=str(read(Path(plan['paths']['validation_root'])/'results.json')['primary_seed'])
        pred=baseline[backbone][seed]['test']['scores'];pred={s:pred[s] for s in ids}
        data=original_test['branches'][backbone]
        controls[backbone]=summary(data,ids,rank_accept(pred,ids,.5))
    # Grouped paired intervals condition on the frozen routes, not training/search variance.
    groups={}
    for sid in ids:groups.setdefault(samples[sid]['rgb_sha256'],[]).append(sid)
    group_values=list(groups.values());rng=np.random.default_rng(cfg['seed']);deltas=[]
    new=set(output['test']['predeclared_batch_quotas']['0.5']['accepted_ids']);old=set(controls['unet']['accepted_ids'])
    for _ in range(cfg['bootstrap_draws']):
        drawn=[s for i in rng.integers(0,len(group_values),len(group_values)) for s in group_values[i]]
        def f1(accepted):
            tp=pred=gt=0
            for s in drawn:
                b=branches_test[s]['light' if s in accepted else 'deepsolo'];tp+=b['tp'];pred+=b['detected_words'];gt+=b['gt_words']
            return 2*tp/(pred+gt)
        deltas.append(100*(f1(new)-f1(old)))
    results=dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',result_class=plan['result_class'],
        selected=choice['name'],splits=output,token_baselines_at_half=controls,
        paired_delta_vs_unet_token_50=dict(lower95_pp=float(np.quantile(deltas,.025)),upper95_pp=float(np.quantile(deltas,.975)),conditional_on_fixed_routes=True),
        deployment_cost=dict(**choice['model_cost'],serialized_bytes=(root/'deployment.json').stat().st_size,feature_dimension=choice['feature_dimension'],
            head_and_observation_median_ms=1000*float(np.median(timings)),head_and_observation_p95_ms=1000*float(np.quantile(timings,.95)),
            timing_scope='CPU observations, selected projection and portable head; excludes OCR/U-Net forward/shared pooling, warm cache',
            clip_required=False,blip_required=False,new_expert_forward_passes_for_routing=0,requires_shared_pooling=deployment['shared_projection'] is not None),
        actual_new_deepsolo_calls=0,routing_scope='Logical cached expert routing; no full-pipeline latency claim',
        test_used_for_selection=False,refit_after_calibration=False,plan_sha256=sha(root/'resolved_configuration.json'),frozen_choices_sha256=sha(root/'frozen_choices.json'))
    seal(root/'test_predictions.json',dict(scores=scores,per_image=per_image));seal(root/'results.json',results)
    seal(root/'independent_audit.json',dict(status='PASS',checks=checks,results_sha256=sha(root/'results.json'),deployment_sha256=sha(root/'deployment.json'),
        actual_training_ids=deployment['training_ids'],validation_ids=deployment['validation_ids'],test_not_used_for_selection=True))
    seal(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json'),audit_sha256=sha(root/'independent_audit.json')))
    log('COMPLETE',selected=choice['name'],f1_at_half=100*output['test']['predeclared_batch_quotas']['0.5']['combined']['f1'],cost=results['deployment_cost'])


def execute(root):
    plan=read(root/'resolved_configuration.json');cfg=plan['config'];torch.set_num_threads(cfg['execution']['cpu_threads'])
    if (root/'completion.json').exists():
        done=read(root/'completion.json')
        if done['status']!='SUCCESS' or done['results_sha256']!=sha(root/'results.json') or done['audit_sha256']!=sha(root/'independent_audit.json'):
            raise ValueError('Completed immutable run changed')
        log('ALREADY_COMPLETE',root=str(root));return
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;cv_threads=1
    import cv2
    cv2.setNumThreads(cv_threads)
    try:
        for scope,location in [('source','source_root'),('validation','validation_root')]:
            for name,digest in cfg['inputs_sha256'][scope].items():
                if sha(Path(plan['paths'][location])/name)!=digest:raise ValueError('Pinned evidence changed after preparation: '+name)
        shared=shared_context_stage(plan,root,'train')
        features,views,branches,reference,names,groups=training_feature_stage(plan,root,shared)
        if not (root/'frozen_choices.json').exists():fit_and_freeze(plan,root,features,views,branches,reference,names,groups)
        # Validation-only choice is immutable before accessing TEST observations/outcomes.
        selected=read(root/'deployment.json')
        test_shared=shared_context_stage(plan,root,'test') if selected['shared_projection'] is not None else {}
        evaluate(plan,root,test_shared,features,branches)
    except BaseException as error:
        peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()) if torch.cuda.is_initialized() else None
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error=str(error),traceback=traceback.format_exc(),out_of_memory=isinstance(error,torch.cuda.OutOfMemoryError),cuda_peak=peak))
        raise


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode');p.add_argument('--config',type=Path);p.add_argument('--paths',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--run-root',type=Path);args=p.parse_args()
    if args.run_root:root=args.run_root.resolve()
    else:
        plan,root=prepare(args);log('PREFLIGHT_PASS',root=str(root),populations={k:len(v) for k,v in plan['partitions'].items()})
        if args.preflight:return
        freeze(plan,root)
    if REPO==(root/'code_snapshot').resolve():execute(root)
    else:subprocess.run([sys.executable,'-u','-m','scripts.run_budget_distilled_router','--run-root',str(root)],cwd=root/'code_snapshot',
                        env={**os.environ,'PYTHONPATH':str(root/'code_snapshot')},check=True)


if __name__=='__main__':main()
