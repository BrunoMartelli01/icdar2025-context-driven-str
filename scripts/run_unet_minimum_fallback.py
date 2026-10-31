"""APPROXIMATE U-Net/corrector to DeepSolo routing at minimum development cost."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
import torch

from localize.independent_word_slots import boundary_bbox,detector_boundary_in_original,EmptyDetectorBoundaryError
from models.grounded_word_recovery import GroundedVerifier,VisualEditTransducer
from recognition.deepsolo_word_f1 import recognition_confidence
from recognition.grounded_word_recovery import decide
from recognition.trainable_image_router import fit_model,predict
from recognition.unet_minimum_fallback import (exact_counts,independent_counts,aggregate_counts,
    routing_curve,minimum_oracle,fallback_ids,observable_features)
from scripts.grounded_word_editor import edited_candidates
from scripts.run_grounded_word_recovery import native_features,score_rows,gpu_start
from scripts.run_selective_word_repair_tests import gpu_end
from scripts.run_word_evidence_graphs import state_digest

REPO=Path(__file__).resolve().parents[1]
PHASES=('fit','calibration','evaluation','test')

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,ensure_ascii=False);stream.write('\n')
def status(root,stage,**values):
    value=dict(stage=stage,status='SUCCESS' if stage=='COMPLETE' else 'RUNNING',pid=os.getpid(),**values)
    (root/'status.json').write_text(json.dumps(value,indent=2)+'\n');print(stage,json.dumps(values),flush=True)
def verify(path,expected):
    if sha(path)!=expected:raise ValueError('Scientific input differs: '+str(path))
def tag(sid):return sid.replace(':','__')

def prepare(root,c,smoke):
    root.mkdir(exist_ok=False,parents=True)
    for path,expected in c['input_sha256'].items():verify(path,expected)
    source=Path(c['paths']['grounded_run'])
    manifest=read(source/'artifact_manifest.json')['artifacts']
    for path,item in manifest.items():verify(source/path,item['sha256'])
    data=pickle.loads((source/'prepared.pkl').read_bytes())
    plan=read(Path(c['paths']['expert_run'])/'resolved_configuration.json')
    train={s['sample_id']:s for s in plan['samples']['train']};test={s['sample_id']:s for s in plan['samples']['test']}
    excluded=[]
    for s in data['fit']:
        if s['sample_id'] not in train:
            if s['sample_id'] not in c['known_duplicate_train_ids']:raise ValueError('Unexpected missing fitting image')
            excluded.append(s['sample_id'])
    if sorted(excluded)!=sorted(c['known_duplicate_train_ids']):raise ValueError('Known duplicate exclusion changed')
    data['fit']=[s for s in data['fit'] if s['sample_id'] not in excluded]
    if smoke:data={p:rows[:c['smoke_images'][p]] for p,rows in data.items()}
    paired={};inputs={}
    for phase,samples in data.items():
        for s in samples:
            q=(test if phase=='test' else train)[s['sample_id']]
            if q['image_sha256']!=s['image_sha256'] or q['image_wh']!=list(reversed(s['image_hw'])):
                raise ValueError('Expert RGB/coordinate mismatch: '+s['sample_id'])
            verify(s['image_path'],s['image_sha256']);verify(s['annotation_path'],s['annotation_sha256'])
            paired[(phase,s['sample_id'])]=q
            prefix=Path(c['paths']['expert_run'])/('test' if phase=='test' else 'train')
            for name in ('deepsolo','clip'):
                path=prefix/name/(tag(s['sample_id'])+'.json');value=read(path)
                if value['contract']['image_sha256']!=s['image_sha256'] or value['contract']['sample_id']!=s['sample_id']:
                    raise ValueError('Cache contract mismatch: '+str(path))
                if name=='deepsolo':
                    if not value['response']['success'] or value['ground_truth_used_in_inference']:
                        raise ValueError('Failed/GT-contaminated expert response')
                else:
                    if value['ground_truth_used']:raise ValueError('GT-contaminated image features')
                    array=path.with_suffix('.npz');verify(array,value['arrays_sha256']);inputs[str(array)]=sha(array)
                inputs[str(path)]=sha(path)
    (root/'prepared.pkl').write_bytes(pickle.dumps(dict(data=data,paired=paired)))
    write(root/'resolved_config.json',c);write(root/'input_shards.json',inputs)
    write(root/'protocol.json',dict(mode='approximate',claim_label='approximate reproduction',
        result_class='smoke-test result' if smoke else 'approximate reproduction',
        metric='Micro one-to-one original bboxIoU>=0.5 and exact Unicode casefold, punctuation retained; all care GT and all outputs; no ignore neutralization',
        populations={k:len(v) for k,v in data.items()},excluded_fitting_pixel_duplicates=excluded,
        paper_target=c['primary_target'],paper_metric_equivalence='UNRESOLVED; nominal target only',
        fallback='Logical image-wise replacement by complete cached DeepSolo outputs; no online latency claim',
        test_policy_selection=False,posthoc_test_curve='Descriptive minimum for frozen rankings, explicitly not a development-selected operating point',
        historical_test_exposure=True,independent_validation=False,seed=c['seed']))
    return data

def load(root):return pickle.loads((root/'prepared.pkl').read_bytes())

def light_stage(root,c):
    tick=time.monotonic();source=Path(c['paths']['grounded_run']);data=load(root)['data']
    old=read(source/'resolved_config.json');features=native_features(source,data)
    memory=gpu_start(old,'frozen U-Net recovery head inference');v=old['verifier'];g=old['editor']
    verifier=GroundedVerifier(old['visual_dimensions'],v['hidden'],v['layers'],old['maximum_bytes']).cuda().eval()
    verifier.load_state_dict(torch.load(source/'verifier/best.pt',weights_only=True),strict=True);verifier.requires_grad_(False)
    editor=VisualEditTransducer(old['visual_dimensions'],g['hidden'],g['layers'],old['maximum_bytes'],visual=False).cuda().eval()
    editor.load_state_dict(torch.load(source/'editor/text/best.pt',weights_only=True),strict=True);editor.requires_grad_(False)
    before=[state_digest(verifier),state_digest(editor)]
    vp=read(source/'verifier/selection_frozen.json')['replacement']['policy'];ep=read(source/'editor/text/selection_frozen.json')['policy']
    replay=0
    for phase,samples in data.items():
        outputs={};all_scores={}
        original=read(source/f'{phase}_editor_text_selected_predictions.json') if phase in ('evaluation','test') else None
        for i,s in enumerate(samples):
            sid=s['sample_id'];z=features[phase][sid]
            values=score_rows(verifier,s,s['pool'],(z['image'],z['context']),old)[:,-1]
            scores=dict(zip([p['id'] for p in s['pool']],map(float,values)))
            prefix=decide(s['baseline_gnn'],s['pool'],scores,**vp,overlap=old['compatibility_iou'],replacements=True)
            pp,sc,details=edited_candidates(editor,verifier,s,z,old,'text',prefix)
            scores.update(sc)
            output=decide(prefix,pp,scores,**ep,overlap=old['compatibility_iou'],replacements=True)
            if original is not None:
                if output!=original[sid]:raise ValueError('Frozen light branch does not replay: '+sid)
                replay+=1
            outputs[sid]=output;all_scores[sid]=scores
            if (i+1)%100==0:status(root,'LIGHT',split=phase,images=i+1)
        write(root/f'{phase}_light_predictions.json',outputs);write(root/f'{phase}_verification_scores.json',all_scores)
    after=[state_digest(verifier),state_digest(editor)]
    if before!=after:raise ValueError('Frozen recovery heads changed')
    write(root/'light_qualification.json',dict(replayed_heldout_images=replay,heads_before=before,heads_after=after,
        new_native_OCR_calls=0,new_ENCODER_calls=0,new_DeepSolo_calls=0,
        memory=gpu_end(memory,'Frozen RGB verifier and text edit transducer; cached PARSeq features',
            old['runtime']['editor_batch_size'],[old['runtime']['editor_batch_size'],2,16,384],tick)))
    del verifier,editor,features;gc.collect();torch.cuda.empty_cache()

def convert_heavy(response,image_wh,orientation):
    outputs=[]
    for d in response['detections']:
        points=detector_boundary_in_original(d['geometry']['boundary_points'],tuple(image_wh),orientation)
        try:box=boundary_bbox(points,tuple(image_wh));invalid=None
        except EmptyDetectorBoundaryError as error:box=None;invalid=dict(unclipped=error.unclipped_bbox,clipped=error.clipped_bbox)
        outputs.append(dict(text=d['text'],bbox=box,score=d['score'],instance_id=d['source_index'],
            recognition_confidence=recognition_confidence(d['raw']['recognition_class_probabilities']),invalid_envelope=invalid))
    if len({p['instance_id'] for p in outputs})!=len(outputs):raise ValueError('Duplicate expert output identity')
    return outputs

def pairs_stage(root,c):
    loaded=load(root);data=loaded['data'];pairs=loaded['paired'];source=Path(c['paths']['grounded_run'])
    features=native_features(source,data);inputs=read(root/'input_shards.json');all_rows={};all_arrays={};checks=0;names=None
    for phase,samples in data.items():
        predictions=read(root/f'{phase}_light_predictions.json');scores=read(root/f'{phase}_verification_scores.json')
        rows={};tabular=[];vision=[]
        for s in samples:
            sid=s['sample_id'];q=pairs[(phase,sid)];prefix=Path(c['paths']['expert_run'])/('test' if phase=='test' else 'train')
            path=prefix/'deepsolo'/(tag(sid)+'.json');verify(path,inputs[str(path)]);raw=read(path)
            heavy=convert_heavy(raw['response'],q['image_wh'],q['orientation'])
            count=exact_counts(predictions[sid],s['words'],c['matching_iou']);ind=independent_counts(predictions[sid],s['words'],c['matching_iou'])
            if any(count[k]!=ind[k] for k in ind):raise ValueError('Independent light matching differs')
            choices={}
            for name,policy in c['heavy_profiles'].items():
                pp=[p for p in heavy if p['score']>=policy['score_threshold'] and p['recognition_confidence']>=policy['recognition_threshold']]
                m=exact_counts(pp,s['words'],c['matching_iou']);ind=independent_counts(pp,s['words'],c['matching_iou'])
                if any(m[k]!=ind[k] for k in ind):raise ValueError('Independent heavy matching differs')
                choices[name]=dict(predictions=pp,counts=m);checks+=1
            clip_path=prefix/'clip'/(tag(sid)+'.npz');verify(clip_path,inputs[str(clip_path)])
            with np.load(clip_path,allow_pickle=False) as z:
                clip=dict(patches=z['patches'],patch_valid=z['patch_valid'])
                value=observable_features(s['image_hw'],s['pool'],predictions[sid],scores[sid],
                    features[phase][sid]['image'],clip,c['feature_quantiles'])
            if names is None:names=value['names']
            if names!=value['names']:raise ValueError('Feature schema differs')
            tabular.append(value['tabular']);vision.append(value['vision'])
            rows[sid]=dict(sample_id=sid,image_sha256=s['image_sha256'],pixel_group=q['rgb_sha256'],
                ground_truth=s['words'],light=dict(predictions=predictions[sid],counts=count),heavy=choices)
            checks+=1
        all_rows[phase]=rows;write(root/f'{phase}_paired_rows.json',rows)
        for key,values in [('tabular',tabular),('vision',vision)]:all_arrays[phase+'_'+key]=np.asarray(values,np.float64)
    np.savez_compressed(root/'features.npz',**all_arrays)
    write(root/'feature_schema.json',dict(tabular_names=names,dimensions={k:list(v.shape) for k,v in all_arrays.items()},
        visual='RGB-only cached CLIP patches + frozen native PARSeq crop columns; no other OCR outputs',
        actual_DeepSolo_output_used_as_inference_feature=False,ground_truth_used_as_inference_feature=False))
    write(root/'pair_audit.json',dict(status='PASS',independent_image_branch_checks=checks))
    if not c['smoke']:
        actual=aggregate_counts([r['light']['counts'] for r in all_rows['test'].values()],0,len(all_rows['test']))
        if (actual['tp'],actual['predictions'],actual['gt_words'])!=(1251,1774,2204):raise ValueError('62.8959% source baseline differs')

def fractions(c):return [i/c['budget_steps'] for i in range(c['budget_steps']+1)]

def fit_stage(root,c):
    data=load(root)['data'];rows={p:read(root/f'{p}_paired_rows.json') for p in PHASES}
    x=np.load(root/'features.npz');ids={p:[s['sample_id'] for s in data[p]] for p in PHASES}
    profiles=list(c['heavy_profiles']);fit=rows['fit'];cal=rows['calibration']
    y=np.asarray([[v for profile in profiles for v in (
        fit[s]['heavy'][profile]['counts']['tp']-fit[s]['light']['counts']['tp'],
        fit[s]['heavy'][profile]['counts']['predictions']-fit[s]['light']['counts']['predictions'])] for s in ids['fit']],float)
    groups=sorted(set(fit[s]['pixel_group'] for s in ids['fit']),key=lambda v:hashlib.sha256((str(c['seed'])+v).encode()).hexdigest())
    group_fold={g:i%c['folds'] for i,g in enumerate(groups)}
    fold=np.asarray([group_fold[fit[s]['pixel_group']] for s in ids['fit']])
    write(root/'folds.json',dict(folds={s:int(f) for s,f in zip(ids['fit'],fold)},grouped_by_decoded_RGB=True))
    trials=[];catalog=[];settings=dict(seed=c['seed'],trees=c['trees'])
    for feature_set in c['feature_sets']:
        xx=x['fit_'+feature_set]
        for number,recipe in enumerate(c['recipes']):
            name=feature_set+'_'+str(number);oof=np.zeros_like(y)
            for f in range(c['folds']):
                train=np.flatnonzero(fold!=f);val=np.flatnonzero(fold==f)
                model=fit_model(xx[train],y[train],[ids['fit'][i] for i in train],recipe,settings)
                oof[val]=predict(model,xx[val])
            model=fit_model(xx,y,ids['fit'],recipe,settings);write(root/'models'/(name+'.json'),model)
            cal_pred=predict(model,x['calibration_'+feature_set])
            np.savez_compressed(root/'models'/(name+'_development.npz'),oof=oof,calibration=cal_pred)
            catalog.append(dict(name=name,feature_set=feature_set,recipe=recipe))
            for j,profile in enumerate(profiles):
                for target in c['selection_targets']:
                    score_oof={s:float(2*v[2*j]-target*v[2*j+1]) for s,v in zip(ids['fit'],oof)}
                    score_cal={s:float(2*v[2*j]-target*v[2*j+1]) for s,v in zip(ids['calibration'],cal_pred)}
                    _,fc=routing_curve({s:fit[s]['light']['counts'] for s in ids['fit']},
                        {s:fit[s]['heavy'][profile]['counts'] for s in ids['fit']},score_oof,ids['fit'])
                    _,cc=routing_curve({s:cal[s]['light']['counts'] for s in ids['calibration']},
                        {s:cal[s]['heavy'][profile]['counts'] for s in ids['calibration']},score_cal,ids['calibration'])
                    for fraction in fractions(c):
                        a=fc[len(fallback_ids(score_oof,fraction))];b=cc[len(fallback_ids(score_cal,fraction))]
                        trials.append(dict(model=name,profile=profile,target=target,fraction=fraction,oof=a,calibration=b))
            status(root,'ROUTER_FIT',model=name)
    selected={}
    for target in c['selection_targets']:
        for margin in c['selection_margins']:
            eligible=[t for t in trials if t['target']==target and t['oof']['f1']>target+margin and t['calibration']['f1']>target+margin]
            if eligible:
                chosen=min(eligible,key=lambda t:(t['fraction'],-t['calibration']['f1'],-t['oof']['f1'],t['model'],t['profile']))
                selected[str(target)+'_'+str(margin)]=dict(qualified=True,margin=margin,**chosen)
            else:
                candidates=[t for t in trials if t['target']==target]
                chosen=max(candidates,key=lambda t:(min(t['oof']['f1'],t['calibration']['f1']),-t['fraction']))
                selected[str(target)+'_'+str(margin)]=dict(qualified=False,margin=margin,**chosen)
    write(root/'development_trials.json',trials);write(root/'model_catalog.json',catalog)
    # No held-out routed-policy scores or metrics have been computed at this point.
    write(root/'selection_frozen.json',dict(selection_uses_TEST=False,selected=selected,
        model_sha256={v['name']:sha(root/'models'/(v['name']+'.json')) for v in catalog},
        criterion='minimum DeepSolo fraction with OOF and calibration F1 strictly above nominal target plus declared margin'))

def report_stage(root,c):
    data=load(root)['data'];choice=read(root/'selection_frozen.json');catalog={r['name']:r for r in read(root/'model_catalog.json')}
    features=np.load(root/'features.npz');results={};checks=0
    for phase in ('evaluation','test'):
        rows=read(root/f'{phase}_paired_rows.json');ids=[s['sample_id'] for s in data[phase]];results[phase]={}
        light={s:rows[s]['light']['counts'] for s in ids}
        controls=dict(light_only=aggregate_counts(light.values(),0,len(ids)))
        for profile in c['heavy_profiles']:
            heavy={s:rows[s]['heavy'][profile]['counts'] for s in ids}
            controls['all_DeepSolo_'+profile]=aggregate_counts(heavy.values(),len(ids),len(ids))
        outcomes={};curves={};oracles={}
        for label,p in choice['selected'].items():
            model=read(root/'models'/(p['model']+'.json'));info=catalog[p['model']]
            prediction=predict(model,features[phase+'_'+info['feature_set']]);j=list(c['heavy_profiles']).index(p['profile'])
            scores={s:float(2*v[2*j]-p['target']*v[2*j+1]) for s,v in zip(ids,prediction)}
            heavy={s:rows[s]['heavy'][p['profile']]['counts'] for s in ids}
            order,curve=routing_curve(light,heavy,scores,ids);selected=fallback_ids(scores,p['fraction']);ss=set(selected)
            m=aggregate_counts([heavy[s] if s in ss else light[s] for s in ids],len(ss),len(ids))
            if abs(m['f1']-curve[len(ss)]['f1'])>1e-15:raise ValueError('Independent route aggregate differs')
            cases=[];recovered=lost=0
            for s in ids:
                before=set(light[s]['matched_gt']);after=set((heavy[s] if s in ss else light[s])['matched_gt'])
                recovered+=len(after-before);lost+=len(before-after)
                cases.append(dict(sample_id=s,action='DeepSolo' if s in ss else 'light',routing_score=scores[s],
                    recovered_gt_indices=sorted(after-before),lost_gt_indices=sorted(before-after),
                    counts=heavy[s] if s in ss else light[s]))
                checks+=1
            m.update(recovered_words=recovered,lost_words=lost)
            rng=np.random.default_rng(c['seed']);values=np.asarray([[r['tp'],r['predictions'],r['gt']] for r in [heavy[s] if s in ss else light[s] for s in ids]],float)
            draws=np.asarray([values[rng.integers(0,len(ids),len(ids))].sum(0) for _ in range(c['bootstrap_replicates'])])
            ci=np.quantile(2*draws[:,0]/(draws[:,1]+draws[:,2]),[.025,.975]).tolist()
            crossings=[v for v in curve if v['f1']>p['target']]
            descriptive=crossings[0] if crossings else None
            outcomes[label]=dict(frozen_development_policy=p,metrics=m,conditional_image_bootstrap_95_f1=ci,
                actual_online_DeepSolo_calls=0,logical_cached_DeepSolo_images=len(ss),
                target_exceeded=m['f1']>p['target'],posthoc_TEST_ranking_minimum=descriptive if phase=='test' else None)
            write(root/'routes'/(phase+'_'+label+'.json'),dict(per_image=cases,DeepSolo_ids=selected))
            curves[label]=dict(order=order,points=curve,scores=scores)
            oracle=minimum_oracle(light,heavy,ids,p['target']);oracles[label]=oracle
        results[phase]=dict(controls=controls,policies=outcomes)
        write(root/(phase+'_curves.json'),curves);write(root/(phase+'_oracle_diagnostics.json'),oracles)
    write(root/'results.json',dict(claim_label='approximate reproduction',results=results))
    write(root/'independent_audit.json',dict(status='PASS',pair_audit=read(root/'pair_audit.json'),image_route_checks=checks,
        complete_TEST_images=len(data['test']),selection_frozen_sha256=sha(root/'selection_frozen.json')))
    lines=['# U-Net corrected branch: minimum DeepSolo fallback','','Approximate reproduction; nominal paper target, not protocol-equivalent paper superiority.','',
        'All counts below are logical routes on cached complete DeepSolo predictions; actual online DeepSolo calls=0.','',
        '| Development-selected policy | TEST F1 % | DeepSolo images / all (%) | Target exceeded |','| --- | ---: | ---: | --- |']
    for label,v in results['test']['policies'].items():
        m=v['metrics'];lines.append(f"| {label}; {v['frozen_development_policy']['profile']} | {100*m['f1']:.4f} | {m['deepsolo_images']}/{m['images']} ({100*m['deepsolo_fraction']:.2f}%) | {v['target_exceeded']} |")
    lines+=['','Selection uses fitting OOF and calibration only. Full frozen-ranking TEST curves are descriptive; their minimum is TEST-dependent.','',
        'The minimum-cardinality GT oracle is a privileged lower bound and cannot be deployed. Every care word/output is retained; punctuation and original bbox matching are unchanged.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(root/'artifact_manifest.json',dict(artifacts={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','status.json')}))

def run(args):
    path=args.config.resolve()
    if args.mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Isolated explicit approximate configuration required')
    c=read(path)
    if c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':raise ValueError('Wrong mode')
    if c['matching_iou']!=.5 or c['online_DeepSolo_inference'] is not False:raise ValueError('Qualified cached spatial protocol required')
    root=args.output.resolve();c=dict(c,smoke=args.smoke)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=c['cublas_workspace_config']
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if args.stage in ('all','prepare'):prepare(root,c,args.smoke)
    elif read(root/'resolved_config.json')!=c:raise ValueError('Resolved configuration changed')
    try:
        if args.stage in ('all','light'):light_stage(root,c)
        if args.stage in ('all','pairs'):pairs_stage(root,c)
        if args.stage in ('all','fit'):fit_stage(root,c)
        if args.stage in ('all','report'):report_stage(root,c);status(root,'COMPLETE')
    except BaseException as error:
        write(root/'failure.json',dict(error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        status(root,'FAILED',error=str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--stage',default='all',choices=['all','prepare','light','pairs','fit','report'])
    run(p.parse_args())
