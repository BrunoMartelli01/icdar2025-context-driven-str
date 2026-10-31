"""Four isolated APPROXIMATE word-F1 experiments, with fixed qualified inputs."""
from __future__ import annotations

import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from localize.word_graph_recovery import (HypothesisScorer, addition_features,
    hypothesis_adjacency, preserve_and_add, marginal_training_targets,
    training_owner_targets, complete_word_loss)
from localize.graph_occlusion_followups import (ContourPointRefiner, cyclic_contour_loss, contour_boxes)
from localize.occlusion_four_tests import features, iou
from scripts.run_graph_occlusion_followups import source, paired_comparison
from scripts.run_graph_occlusion_research import baseline_filter, budget_and_filter, dataset_for, record_failure
from scripts.run_teacher_student_tests import frozen_selector, report, release, peaks
from scripts.run_occlusion_four_tests import load_rgb, score_record, preselect, labels_for
from scripts.run_mask_crop_budget import read, write, setup, utc, CropOCR
from scripts.evaluate_cached_totaltext_padding import crop_hash
from segmentation.data import file_sha256
from segmentation.training import seed_everything
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box

REPO=Path(__file__).resolve().parents[1]


def announce(out, phase, **values):
    write(out/'status.json',dict(status='SUCCESS' if phase=='REPORT_COMPLETE' else 'RUNNING',
        phase=phase,pid=os.getpid(),utc=utc(),**values))
    print(phase,json.dumps(values),flush=True)


def checkpoint(path, **values):
    temporary=path.with_suffix('.partial.pt');torch.save(values,temporary);os.replace(temporary,path)


def training_batch(examples, *, device, extra=()):
    """Keep the qualified supervised objective's FULL annotation contract."""
    fields=('image','mask','kernel','validity','separation','owners_native')+tuple(extra)
    missing={key for example in examples for key in fields if key not in example}
    if missing:raise ValueError('Missing supervised batch fields: '+str(sorted(missing)))
    return {key:torch.stack([example[key] for example in examples]).to(device) for key in fields}


def validate_namespace(mode, path, config):
    if (mode!='approximate' or REPO/'configs/approximate' not in path.resolve().parents
            or config.get('reproduction_mode')!='approximate'
            or config.get('experiment_kind')!='word_graph_recovery_v1'):
        raise ValueError('Explicit isolated APPROXIMATE word-graph configuration required')
    if config['metric_iou_threshold']!=.5 or config['budget']!=20:
        raise ValueError('Qualified word metric and unique-crop budget20 required')
    if config.get('require_independent_validation',True):
        raise ValueError('These historical cohorts are NOT independent. Supply a separately qualified experiment before requesting independent validation.')
    allowed={'PAPER_SPECIFIED','REPOSITORY_SPECIFIED','ASSUMPTION','HARDWARE_DEVIATION','UNRESOLVED'}
    if not config.get('new_decisions') or any(x['classification'] not in allowed for x in config['new_decisions']):
        raise ValueError('Scientific decisions must have one explicit classification')


def prepare(args):
    cp=args.config.resolve();c=read(cp);validate_namespace(args.mode,cp,c)
    for name,digest in c['input_sha256'].items():
        if file_sha256(Path(c['paths'][name]))!=digest:raise ValueError('Input changed: '+name)
    parent=read(Path(c['followup_run'])/'protocol.json')
    if read(Path(c['followup_run'])/'results.json')['status']!='SUCCESS':raise ValueError('Incomplete source follow-up')
    if read(Path(c['followup_run'])/'independent_audit.json')['status']!='PASS':raise ValueError('Source audit failed')
    original=read(Path(c['source_run'])/'protocol.json')
    for name in c['critical_frozen_files']:
        if file_sha256(REPO/name)!=original['source_code_sha256'][name]:raise ValueError('Frozen scientific implementation changed: '+name)
    for name,digest in c['filter_source_sha256'].items():
        if file_sha256(Path(c['filter_source'])/name)!=digest:raise ValueError('Reference filter changed: '+name)
    if file_sha256(Path(c['source_run'])/'results.json')!=c['source_results_sha256']:raise ValueError('Source results changed')
    index=copy.deepcopy(parent['identity']['index'])
    if [len(index[k]) for k in ('fit','calibration','evaluation','test')]!=[1030,100,125,300]:raise ValueError('Source populations changed')
    if args.smoke:index={k:v[:c['smoke'][k]] for k,v in index.items()}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(config_sha256=file_sha256(cp),index=index,smoke=args.smoke,
        code_sha256={str(p.relative_to(REPO)):file_sha256(p) for folder in ('models','localize','scripts','segmentation','recognition') for p in sorted((REPO/folder).rglob('*.py'))})
    if (out/'protocol.json').exists() and read(out/'protocol.json')['identity']!=identity:raise ValueError('Resume source identity changed')
    hardware=setup(c)
    if not (out/'protocol.json').exists():
        write(out/'protocol.json',dict(identity=identity,hardware=hardware,
            result_class='smoke-test result' if args.smoke else 'approximate reproduction; historically explored populations',
            metric=original['metric'],selection_uses_test=False,historical_test_exploration=True,
            source_calibration_exposure=True,seed_count=1,independent_generalization=False))
        write(out/'resolved_config.json',c)
    return c,out,index,read(Path(c['paths']['source_contract']))


@lru_cache(maxsize=2)
def followup_manifest(path):return read(Path(path))['artifacts']


@lru_cache(maxsize=8)
def followup_contour(root, manifest, phase, sid):
    prefix='contour_data/'+phase+'/'+sid.replace(':','__')
    paths=[Path(root)/(prefix+extension) for extension in ('.json','.npz')]
    pinned=followup_manifest(manifest)
    for p in paths:
        if file_sha256(p)!=pinned[str(p.relative_to(root))]['sha256']:raise ValueError('Frozen contour artifact changed: '+str(p))
    meta=read(paths[0])
    if file_sha256(paths[1])!=meta['arrays_sha256']:raise ValueError('Contour cache pair changed')
    with np.load(paths[1]) as z:arrays={k:z[k].copy() for k in z.files}
    return copy.deepcopy(meta),arrays


def contour_payload(c,phase,sid):
    return followup_contour(c['followup_run'],c['paths']['followup_artifact_manifest'],phase,sid)


def initial(c,index,directory):
    records,readings=source(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(readings)
    baseline={phase:(budget_and_filter(c,records[phase],ocr) if phase=='fit' else baseline_filter(c,records[phase],phase,ocr)) for phase in index}
    return records,ocr,baseline


def save_result(directory, name, phase, records, predictions, ocr):
    result=report(records,predictions,ocr.rows)
    write(directory/(phase+'_'+name+'_results.json'),result)
    write(directory/(phase+'_'+name+'_predictions.json'),predictions)
    return result


def memory(directory,label,batch,shape):
    write(directory/'memory.json',peaks(label,batch,shape))


def additions_stage(c,out,index,contract,smoke):
    directory=out/'additions';directory.mkdir(exist_ok=False)
    records,ocr,baseline=initial(c,index,directory);g=c['additions'];pre=frozen_selector(c,'pre');data={}
    fit_words,_=labels_for(records['fit'])
    for phase,entries in index.items():
        data[phase]={}
        for entry in entries:
            sid=entry['sample_id'];r=records[phase][sid];rgb=load_rgb(r)
            ids=preselect(r,'mk',score_record(r,'mk',ocr,pre,False),c['budget'])
            pool=[p for p in r['arms']['mk']['pool'] if p['id'] in ids]
            if len({p['crop_array_sha256'] for p in pool})>c['budget']:raise ValueError('OCR admission overflow')
            ocr.image(rgb,pool)
            _,z=contour_payload(c,phase,sid);field=z['feature_map'][:64];size=z['probabilities'].shape[1];visual=[]
            for p in pool:
                box=np.asarray(p['bbox_native'],float)*field.shape[1]/size
                a,b=np.floor(box[:2]).astype(int);d,e=np.ceil(box[2:]).astype(int)
                patch=field[:,max(0,b):max(b+1,e),max(0,a):max(a+1,d)].reshape(64,-1)
                if not patch.shape[1]:raise ValueError('Empty frozen visual field crop')
                x0,y0,x1,y1=p['padded_bbox_original'];gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
                bitmap=cv2.resize(gray,(8,4),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255
                visual.append(np.r_[patch.mean(1),patch.std(1),bitmap])
            x=addition_features(pool,baseline[phase][sid],r['features']['mk'],ocr.rows,visual,text_bins=g['text_bins'],overlap=c['filter']['overlap'])
            if not len(pool):x=np.empty((0,25+160+7+g['text_bins']),np.float32)
            adj=hypothesis_adjacency(pool,ocr.rows,neighbors=g['neighbors'],maximum_distance=g['maximum_distance'])
            z=dict(features=x,adjacency=adj,pool=pool)
            if phase=='fit':
                target,valid=marginal_training_targets(baseline[phase][sid],pool,fit_words[sid],ocr.rows,overlap=c['filter']['overlap'])
                z.update(target=target,valid=valid)
            data[phase][sid]=z
            path=directory/'data'/phase/(sid.replace(':','__')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,**{k:v for k,v in z.items() if isinstance(v,np.ndarray)})
        announce(out,'ADDITION_DATA',split=phase,images=len(entries))
    vectors=np.concatenate([z['features'] for z in data['fit'].values() if len(z['features'])])
    mean=torch.from_numpy(vectors.mean(0)).cuda();scale=torch.from_numpy(vectors.std(0)).clamp_min(g['minimum_feature_scale']).cuda()
    def score(model,phase):
        output={};model.eval()
        with torch.no_grad():
            for sid,z in data[phase].items():
                x=torch.from_numpy(z['features']).cuda();a=torch.from_numpy(z['adjacency']).cuda()
                values=model((x-mean)/scale,a).sigmoid().cpu().tolist()
                output[sid]=dict(zip([p['id'] for p in z['pool']],values))
        return output
    def predict(phase,scores,threshold):
        return {sid:copy.deepcopy(baseline[phase][sid]) if threshold=='KEEP' else preserve_and_add(
            baseline[phase][sid],z['pool'],scores[sid],threshold=threshold,overlap=c['filter']['overlap']) for sid,z in data[phase].items()}
    capacities={}
    for kind in ('mlp','gnn'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        branch=directory/kind;branch.mkdir();torch.cuda.reset_peak_memory_stats()
        model=HypothesisScorer(vectors.shape[1],g['hidden'],g['layers'],message_passing=kind=='gnn').cuda()
        capacities[kind]=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);history=[]
        base=report(records['calibration'],baseline['calibration'],ocr.rows)['aggregate']
        best=dict(epoch=0,threshold='KEEP',f1=base['f1'],proposals=base['proposals'])
        epochs=1 if smoke else g['epochs']
        for epoch in range(1,epochs+1):
            losses=[];model.train();items=list(data['fit'].values())
            for j in torch.randperm(len(items),generator=rng).tolist():
                z=items[j]
                if not z['valid'].any():continue
                x=torch.from_numpy(z['features']).cuda();a=torch.from_numpy(z['adjacency']).cuda();valid=torch.from_numpy(z['valid']).cuda()
                target=torch.from_numpy(z['target']).cuda();optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model((x-mean)/scale,a)[valid],target[valid])
                if not loss.isfinite():raise FloatingPointError('Nonfinite marginal word loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No eligible addition-training updates')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)));write(branch/'history.json',history)
            checkpoint(branch/'last.pt',state_dict=model.state_dict(),optimizer=optimizer.state_dict(),generator_state=rng.get_state(),epoch=epoch,mean=mean.cpu(),scale=scale.cpu())
            if epoch%g['calibration_every']==0 or epoch==epochs:
                scores=score(model,'calibration');trials=[]
                for threshold in g['thresholds']:
                    pp=predict('calibration',scores,threshold);a=report(records['calibration'],pp,ocr.rows)['aggregate']
                    trials.append(dict(threshold=threshold,aggregate=a))
                    if (a['f1'],-a['proposals'])>(best['f1'],-best['proposals']):
                        best=dict(epoch=epoch,threshold=threshold,f1=a['f1'],proposals=a['proposals'])
                        checkpoint(branch/'best.pt',state_dict=model.state_dict(),selection=best,mean=mean.cpu(),scale=scale.cpu())
                write(branch/f'epoch{epoch}_calibration.json',trials)
            announce(out,'ADDITION_TRAIN',arm=kind,**history[-1],best=best)
        write(branch/'selection_frozen.json',best)
        for phase in ('evaluation','test'):
            scores=score(model,phase);save_result(directory,kind+'_final',phase,records[phase],predict(phase,scores,g['raw_threshold']),ocr)
        if best['epoch']:model.load_state_dict(torch.load(branch/'best.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        for phase in ('evaluation','test'):
            scores=score(model,phase);save_result(directory,kind,phase,records[phase],predict(phase,scores,best['threshold']),ocr)
        memory(branch,'Hypothesis model + AdamW; frozen cached U-Net visual features and PARSeq readings',1,['admitted_candidates',vectors.shape[1]])
        del model,optimizer;release()
    if len(set(capacities.values()))!=1:raise ValueError('Unmatched hypothesis model capacity')
    for phase in ('evaluation','test'):save_result(directory,'baseline',phase,records[phase],baseline[phase],ocr)
    write(directory/'training_targets.json',dict(positive=sum(int(z['target'][z['valid']].sum()) for z in data['fit'].values()),
        eligible=sum(int(z['valid'].sum()) for z in data['fit'].values()),classification='ASSUMPTION',uses='FIT only; conditional incremental exact matching'))
    ocr.release();write(directory/'complete.json',dict(status='SUCCESS',parameters=capacities))


def adjusted_contour_records(model,c,entries,phase,contract,trace,*,padding=0.):
    trace.mkdir(parents=True,exist_ok=True);result={}
    for entry in entries:
        sid=entry['sample_id'];meta,z=contour_payload(c,phase,sid);r=meta['record'];rgb=load_rgb(entry);native=z['probabilities'].shape[1]
        if model is None:
            boxes=z['boxes'].copy();length=boxes[:,2:]-boxes[:,:2]
            boxes=np.concatenate((np.floor(boxes[:,:2]-padding*length),np.ceil(boxes[:,2:]+padding*length)),1)
            boxes=np.clip(boxes,0,native).astype(int).tolist();points=z['initial']
        else:
            with torch.no_grad():points=model(torch.from_numpy(z['feature_map'])[None],torch.from_numpy(z['initial']).cuda(),torch.from_numpy(z['boxes']).cuda(),native).cpu().numpy()
            boxes=contour_boxes(points,z['initial'],z['boxes'],native)
        for p,box in zip(r['arms']['mk']['pool'],boxes):
            p['bbox_native']=box;original=remap_letterbox_native_box(box,rgb.shape[:2][::-1],native_size=native)
            p['bbox']=original;p['padded_bbox_original']=original.copy();a,b,d,e=original
            p['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[b:e,a:d]))
        r['features']['mk']={p['id']:features(p,rgb,z['probabilities'][0],text_threshold=contract['resolved_config']['postprocessing']['text_threshold']) for p in r['arms']['mk']['pool']}
        result[sid]=r;write(trace/(sid.replace(':','__')+'.json'),dict(record=r,points=points.tolist()))
    return result


def contours_stage(c,out,index,contract,smoke):
    directory=out/'contours';directory.mkdir(exist_ok=False);g=c['contours_v3']
    records,ocr,baseline=initial(c,index,directory);ds,by=dataset_for(c,contract);train=[]
    for entry in index['fit']:
        sid=entry['sample_id'];_,z=contour_payload(c,'fit',sid);batch=ds[by[sid]]
        target,ids,valid=training_owner_targets(z['boxes'],batch['owners_native'].numpy(),batch['validity'][0].numpy(),
            minimum_iou=g['minimum_training_iou'],minimum_valid_fraction=g['minimum_valid_fraction'])
        train.append(dict(sid=sid,target=target,ids=ids,valid=valid,owners=batch['owners_native'].numpy()))
    base=report(records['calibration'],baseline['calibration'],ocr.rows)['aggregate']['f1'];capacities={}
    for kind in ('point','complete'):
        seed_everything(c['seed'],deterministic_algorithms=True);branch=directory/kind;branch.mkdir();torch.cuda.reset_peak_memory_stats()
        model=ContourPointRefiner(66,g['hidden'],g['layers'],message_passing=True,iterations=g['iterations'],maximum_step=g['maximum_step']).cuda()
        capacities[kind]=sum(p.numel() for p in model.parameters())
        opt=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay']);rng=torch.Generator().manual_seed(c['seed'])
        best=dict(epoch=0,f1=base,policy='identity');history=[];epochs=1 if smoke else g['epochs']
        for epoch in range(1,epochs+1):
            model.train();losses=[]
            for j in torch.randperm(len(train),generator=rng).tolist():
                t=train[j];_,z=contour_payload(c,'fit',t['sid'])
                if not z['valid'].any():continue
                boxes=torch.from_numpy(z['boxes']).cuda();opt.zero_grad(set_to_none=True)
                points=model(torch.from_numpy(z['feature_map'])[None],torch.from_numpy(z['initial']).cuda(),boxes,z['probabilities'].shape[1])
                loss=cyclic_contour_loss(points,torch.from_numpy(z['target']).cuda(),torch.from_numpy(z['valid']).cuda(),boxes,beta=g['loss_beta'])
                if kind=='complete':loss=loss+complete_word_loss(points,boxes,torch.from_numpy(t['target']).cuda(),torch.from_numpy(t['ids']).cuda(),
                    torch.from_numpy(t['owners']).cuda(),torch.from_numpy(t['valid']).cuda(),inside_weight=g['inside_weight'],extent_weight=g['extent_weight'],contamination_weight=g['contamination_weight'])
                if not loss.isfinite():raise FloatingPointError('Nonfinite complete-word contour loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True);opt.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No valid contour updates')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)));write(branch/'history.json',history)
            checkpoint(branch/'last.pt',state_dict=model.state_dict(),optimizer=opt.state_dict(),generator_state=rng.get_state(),epoch=epoch)
            if epoch%g['calibration_every']==0 or epoch==epochs:
                exported=adjusted_contour_records(model,c,index['calibration'],'calibration',contract,branch/f'epoch{epoch}_calibration')
                pp=budget_and_filter(c,exported,ocr);metric=report(exported,pp,ocr.rows);write(branch/f'epoch{epoch}_calibration/results.json',metric)
                if metric['aggregate']['f1']>best['f1']:
                    best=dict(epoch=epoch,f1=metric['aggregate']['f1'],policy='refiner');checkpoint(branch/'best.pt',state_dict=model.state_dict(),selection=best)
                ocr.release()
            announce(out,'CONTOUR_TRAIN',arm=kind,**history[-1],best=best)
        write(branch/'selection_frozen.json',best)
        for label,chosen in ((kind+'_final',True),(kind,bool(best['epoch']))):
            if label==kind and chosen:model.load_state_dict(torch.load(branch/'best.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                exported=adjusted_contour_records(model,c,index[phase],phase,contract,branch/label/phase) if chosen else records[phase]
                pp=budget_and_filter(c,exported,ocr);save_result(directory,label,phase,exported,pp,ocr)
        memory(branch,'Contour graph + AdamW; frozen CPU fields, CPU coordinate sampler, PARSeq at calibration',1,['proposals',g['points'],66])
        del model,opt;ocr.release();release()
    if len(set(capacities.values()))!=1:raise ValueError('Unmatched contour capacity')
    trials=[]
    for padding in g['padding_fractions']:
        exported=adjusted_contour_records(None,c,index['calibration'],'calibration',contract,directory/f'padding_{padding}/calibration',padding=padding)
        pp=budget_and_filter(c,exported,ocr);trials.append(dict(padding=padding,aggregate=report(exported,pp,ocr.rows)['aggregate']))
    selected=max(trials,key=lambda z:(z['aggregate']['f1'],-z['padding']))['padding']
    write(directory/'padding_selection_frozen.json',dict(padding=selected,trials=trials))
    for phase in ('evaluation','test'):
        exported=adjusted_contour_records(None,c,index[phase],phase,contract,directory/'padding'/phase,padding=selected)
        save_result(directory,'padding',phase,exported,budget_and_filter(c,exported,ocr),ocr)
        save_result(directory,'baseline',phase,records[phase],baseline[phase],ocr)
    ocr.release();write(directory/'complete.json',dict(status='SUCCESS',parameters=capacities))


def final_report(c,out,index,contract,smoke):
    from scipy.optimize import linear_sum_assignment
    sections={'additions':['baseline','mlp','mlp_final','gnn','gnn_final'],
        'contours':['baseline','point','point_final','complete','complete_final','padding'],
        'geometry':['baseline','unet','unet_final','proposal','proposal_final','self','self_final','graph','graph_final'],
        'regional':['baseline','gt','gt_final','fixed','fixed_final','ema','ema_final']}
    records,readings=source(c,index);checks=0;summary={};lines=['# Word graph recovery experiments','',
        '**Smoke-test result.**' if smoke else '**Approximate reproduction; historically explored cohorts.**','',
        'Primary: exact casefold retaining punctuation AND unpadded bbox IoU>=0.5. All care GT and outputs count. No independent generalization claim.','']
    for stage,names in sections.items():
        if read(out/stage/'complete.json')['status']!='SUCCESS':raise ValueError('Incomplete required stage: '+stage)
        for line in (out/stage/'ocr.jsonl').open():
            value=json.loads(line);key=value['crop_array_sha256']
            if key in readings and readings[key]['raw_text']!=value['raw_text']:raise ValueError('OCR transcript conflict')
            readings[key]=value
    for phase in ('evaluation','test'):
        gt,_=labels_for(records[phase]);results={};predictions={};comparisons={};rng=np.random.default_rng(c['seed'])
        for stage,names in sections.items():
            for name in names:
                key=stage+'/'+name;result=read(out/stage/(phase+'_'+name+'_results.json'));pp=read(out/stage/(phase+'_'+name+'_predictions.json'))
                by={r['sample_id']:r for r in result['per_image']}
                if set(pp)!=set(gt) or set(by)!=set(gt):raise ValueError('Incomplete evaluation population: '+key)
                for sid,pool in pp.items():
                    matrix=np.array([[iou(p['bbox'],w['bbox'])>=.5 and readings[p['crop_array_sha256']]['raw_text'].casefold()==w['text'].casefold() for w in gt[sid]] for p in pool],bool).reshape(len(pool),len(gt[sid]))
                    a,b=linear_sum_assignment(matrix.astype(int),maximize=True);count=int(matrix[a,b].sum())
                    if count!=by[sid]['correct'] or len(pool)!=by[sid]['proposals'] or len(gt[sid])!=by[sid]['gt_words']:raise ValueError('Independent word audit failed: '+key+' '+sid)
                    checks+=1
                results[key]=result;predictions[key]=pp
        for name in ('mlp','mlp_final','gnn','gnn_final'):
            for sid,pp in predictions['additions/'+name].items():
                base=predictions['additions/baseline'][sid]
                if pp[:len(base)]!=base:raise ValueError('Independent locked-output audit failed: '+name+' '+sid)
        reference=results['additions/baseline']
        for key,result in results.items():comparisons[key]=paired_comparison(result,reference,c,rng)
        matched={'additions/gnn':'additions/mlp','additions/gnn_final':'additions/mlp_final','contours/complete':'contours/point',
                 'contours/complete_final':'contours/point_final','geometry/graph':'geometry/self','geometry/proposal':'geometry/unet',
                 'geometry/graph_final':'geometry/self_final','geometry/proposal_final':'geometry/unet_final',
                 'regional/fixed':'regional/gt','regional/ema':'regional/fixed','regional/fixed_final':'regional/gt_final','regional/ema_final':'regional/fixed_final'}
        for key,control in matched.items():comparisons[key]['matched_control']=dict(name=control,**paired_comparison(results[key],results[control],c,rng))
        lines += ['## '+phase,'','| Arm | Word F1 % | Delta vs baseline | Recovered | Damaged | FP |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for key,result in results.items():
            a=result['aggregate'];q=comparisons[key]
            # Recovery/damage keys come from the qualified change comparator.
            lines.append(f"| {key} | {a['f1']*100:.3f} | {q['delta_f1_points']:+.3f} | {sum(len(r['recovered_gt_indices']) for r in q['per_image'])} | {sum(len(r['damaged_gt_indices']) for r in q['per_image'])} | {a['proposals']-a['correct']} |")
        write(out/(phase+'_results.json'),results);write(out/(phase+'_predictions.json'),predictions);write(out/(phase+'_comparisons.json'),comparisons)
        summary[phase]={k:v['aggregate'] for k,v in results.items()}
    write(out/'independent_audit.json',dict(status='PASS',image_arm_checks=checks,matcher='Independent SciPy rectangular linear assignment; maximum exact word matches'))
    write(out/'results.json',dict(status='SUCCESS',claim_label='smoke-test result' if smoke else 'approximate reproduction',smoke=smoke,results=summary))
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    announce(out,'ARTIFACT_HASHING',independent_image_arm_checks=checks)
    artifacts={str(p.relative_to(out)):dict(sha256=file_sha256(p),bytes=p.stat().st_size)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('artifact_manifest.json','status.json')}
    write(out/'artifact_manifest.json',dict(status='PASS',artifacts=artifacts,excluded=['mutable status.json']))
    announce(out,'REPORT_COMPLETE',independent_image_arm_checks=checks,hashed_artifacts=len(artifacts))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--stage',choices=['additions','contours','geometry','regional','report','all'],required=True);args=p.parse_args()
    try:
        c,out,index,contract=prepare(args)
        from scripts.word_graph_geometry_stages import geometry_stage
        from scripts.word_graph_regional_stages import regional_stage
        stages=dict(additions=additions_stage,contours=contours_stage,geometry=geometry_stage,regional=regional_stage,report=final_report)
        for name in stages if args.stage=='all' else [args.stage]:
            if name!='report' and (out/name/'complete.json').exists():
                if read(out/name/'complete.json')['status']!='SUCCESS':raise ValueError('Unsuccessful stage receipt')
                continue
            announce(out,name.upper());stages[name](c,out,index,contract,args.smoke)
    except Exception as exc:
        record_failure(args.output.resolve(),exc);raise


if __name__=='__main__':main()
