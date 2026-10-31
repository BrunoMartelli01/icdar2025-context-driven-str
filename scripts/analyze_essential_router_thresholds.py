"""Validation-frozen and explicitly post-hoc threshold frontiers on three datasets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.essential_portable_router import predict_portable_essential
from recognition.essential_threshold_frontier import (threshold_curve,apply_cut,maximum_f1,minimum_cost,pareto,
    fraction,simultaneous_bootstrap,common_thresholds,gain_identity,observable_length_rescale)
from recognition.unet_corrected_token_interfaces import lexical_counts,ranked_ids,route_summary
from scripts.run_essential_icdar_transfer import (score_population,pixels,truth_and_validity,ignored_rows)

read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def checked(p,pins):
    p=Path(p);digest=sha(p)
    if str(p) in pins and pins[str(p)]!=digest:raise ValueError('Changed source: '+str(p))
    pins[str(p)]=digest;return read(p)

def heavy_counts(raw,s,gt,valid,policy):
    from recognition.clip_router_evaluation import deepsolo_predictions
    from recognition.deepsolo_word_f1 import recognition_confidence,apply_policy
    if not raw['response']['success'] or raw['contract']['image_sha256']!=s['image_sha256']:raise ValueError('Mismatched validation fallback')
    rows=deepsolo_predictions(raw['response'],s['image_wh'],s['orientation'],valid,policy['score_threshold'])
    by={d['source_index']:d for d in raw['response']['detections']}
    for r in rows:r['recognition_confidence']=recognition_confidence(by[r['instance_id']]['raw']['recognition_class_probabilities'])
    rows=apply_policy(rows,policy)
    return lexical_counts([dict(raw_text=r['text'] if r.get('invalid_envelope') is None else '') for r in rows],gt,[r['ignored'] for r in rows])

def icdar_validation(c,pins):
    root=Path(c['paths']['icdar_transfer']);plan=checked(root/'resolved_configuration.json',pins)
    outputs,scores,inputs=score_population(root,plan,'calibration');pins.update(inputs)
    if c.get('score_mode')=='normalized_utility':
        scores={ds:{e:{s:observable_length_rescale(v,outputs[ds][e][s]) for s,v in values.items()}
            for e,values in experts.items()} for ds,experts in scores.items()}
    policy=read(plan['config']['paths']['deepsolo_profile'])['selected_policy'];branches={}
    from recognition.clip_router_evaluation import native_mask
    from scripts.diagnose_parseq_candidate_ranking_train import _task22_word_texts
    from localize.practical_remap_protocol import train_task22_word_boxes
    from segmentation.data import icdar13_task22_rgb_to_target_and_validity
    from recognition.scene_text_datasets import parse_icdar2015_annotation
    from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
    audit=[]
    for ds,samples in plan['population']['calibration'].items():
        branches[ds]={e:{} for e in c['experts']}
        for s in samples:
            sid=s['sample_id'];identity=sid.split(':')[-1]
            if ds=='icdar13':
                base=Path(c['paths']['icdar13_root'])/'raw/train_segmentation_gt';annotation=base/(identity+'_GT.txt');mask=base/(identity+'_GT.bmp')
                pins[str(annotation)]=sha(annotation);pins[str(mask)]=sha(mask)
                words=_task22_word_texts(annotation);boxes=train_task22_word_boxes(mask,annotation) if words else []
                if len(words)!=len(boxes):raise ValueError('Task2.2 word group/text mismatch')
                gt=[dict(text=t,bbox=list(b.bbox)) for t,b in zip(words,boxes)]
                with Image.open(mask) as im:_,valid_image=icdar13_task22_rgb_to_target_and_validity(im,annotation.read_text(encoding='utf-8-sig'))
                valid=native_mask(valid_image,256)
            else:
                base=Path(c['paths']['icdar2015_root'])/'datasets/icdar2015/train';annotation=base/'annotations'/(identity+'.txt');mask=base/'validity'/(identity+'.png')
                pins[str(annotation)]=sha(annotation);pins[str(mask)]=sha(mask)
                instances,_=clip_instances_to_pixel_extent(parse_icdar2015_annotation(annotation),tuple(s['image_wh']))
                gt=[dict(text=v.transcription,bbox=[v.bbox_xyxy[0],v.bbox_xyxy[1],v.bbox_xyxy[2]+1,v.bbox_xyxy[3]+1]) for v in instances if not v.ignore]
                with Image.open(mask) as im:valid=native_mask(im.convert('L'),256)
            p=root/'fallback_validation'/(sid.replace(':','__')+'.json');raw=checked(p,pins);heavy=heavy_counts(raw,s,gt,valid,policy)
            for expert in c['experts']:
                rows=outputs[ds][expert][sid];light=lexical_counts(rows,gt,ignored_rows(rows,s['image_wh'],valid))
                branches[ds][expert][sid]=dict(light=light,deepsolo=heavy)
            audit.append(dict(sample_id=sid,words=len(gt),annotation_sha256=pins[str(annotation)],mask_sha256=pins[str(mask)]))
    return branches,scores,audit

def totaltext(c,role,pins):
    paths=c['paths'];source=Path(paths['totaltext_features']);budget=Path(paths['totaltext_essential']);native=Path(paths['totaltext_native']);yolo=Path(paths['totaltext_yolo'])
    b=checked(budget/'frozen_choices.json',pins);r='selection' if role=='validation' else 'test'
    schema=checked(source/(r+'_feature_schema.json'),pins);npz=source/(r+'_features.npz')
    if sha(npz)!=schema['arrays_sha256']:raise ValueError('Changed TotalText feature cache')
    pins[str(npz)]=sha(npz)
    branches=checked(source/(r+'_branches.json'),pins);paired=checked(native/('train' if role=='validation' else 'test')/'branches.json',pins)['branches']['yolo']
    trace=checked(source/('train_feature_inputs.json' if role=='validation' else 'test_feature_inputs.json'),pins)
    records={s['sample_id']:s for s in checked(source/'resolved_configuration.json',pins)['records'][r]}
    ids=b['calibration_ids'] if role=='validation' else schema['ids']
    plan=checked(Path(paths['icdar_transfer'])/'resolved_configuration.json',pins);deployment=checked(plan['config']['paths']['router_deployment'],pins)
    with np.load(npz,allow_pickle=False) as z:x=np.concatenate([z['ocr'],z['mask']],1)
    predicted=((x-np.asarray(deployment['mean']))/np.asarray(deployment['scale']))@np.asarray(deployment['coef'])+deployment['intercept']
    all_scores={s:float(v)*(1 if c.get('score_mode')=='normalized_utility' else schema['word_scale'][s]) if schema['eligible'][s] else None for s,v in zip(schema['ids'],predicted)}
    scores=dict(unet={s:all_scores[s] for s in ids},yolo={});result=dict(unet={s:branches[s] for s in ids},yolo={s:paired[s] for s in ids})
    for sid in ids:
        p=yolo/('train' if role=='validation' else 'test')/'light'/(sid.replace(':','__')+'.json');raw=checked(p,pins);rows=raw['rows']
        if raw['ground_truth_used'] or raw['contract']['image_sha256']!=records[sid]['image_sha256']:raise ValueError('Changed native YOLO source')
        mp=Path(trace[sid]['maps_path'])
        if sha(mp)!=trace[sid]['maps_sha256']:raise ValueError('Changed TotalText maps')
        pins[str(mp)]=sha(mp)
        with np.load(mp,allow_pickle=False) as z:prob=z['probabilities']
        score=predict_portable_essential(deployment,sid,rows,records[sid]['image_wh'],prob)['score']
        scores['yolo'][sid]=observable_length_rescale(score,rows) if c.get('score_mode')=='normalized_utility' else score
        if result['unet'][sid]['deepsolo']!=result['yolo'][sid]['deepsolo']:raise ValueError('TotalText experts must share fallback counts')
    return result,scores

def validation_choices(c,branches,scores):
    result={};curves={};bands={}
    for ds in c['datasets']:
        result[ds]={};curves[ds]={};bands[ds]={}
        for expert in c['experts']:
            curve=threshold_curve(branches[ds][expert],scores[ds][expert]);curves[ds][expert]=curve
            band=simultaneous_bootstrap(curve,branches[ds][expert],scores[ds][expert],c['bootstrap_draws'],c['seed']);bands[ds][expert]=band
            baseline=curve[0];best=maximum_f1(curve);policies={}
            for loss in c['loss_budgets']:
                chosen=minimum_cost(curve,baseline,loss);policies['empirical_loss_'+str(loss)]=dict(threshold=chosen['threshold'],validation=chosen,allowed_f1_loss=loss)
                feasible=[p for p,lower in zip(curve,band['lower_gain']) if lower>=-loss]
                chosen=min(feasible,key=lambda p:(p['deepsolo_images'],-fraction(p)))
                policies['bootstrap_band_loss_'+str(loss)]=dict(threshold=chosen['threshold'],validation=chosen,allowed_f1_loss=loss,
                    guarantee='Approximate fixed-head bootstrap band only, not a population F1 guarantee')
            policies['maximum_validation_f1']=dict(threshold=best['threshold'],validation=best)
            policies['raw_positive_utility']=dict(threshold=0.,validation=apply_cut(curve,0.))
            for tolerance in c['near_best_tolerances']:
                chosen=minimum_cost(curve,best,tolerance);policies['near_validation_best_'+str(tolerance)]=dict(threshold=chosen['threshold'],validation=chosen)
            result[ds][expert]=policies
    common={}
    for expert in c['experts']:
        points=common_thresholds({ds:curves[ds][expert] for ds in c['datasets']})
        common[expert]=min((p for p in points if p['no_loss_all']),key=lambda p:p['deepsolo_images'])
    return result,curves,bands,common

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();c=read(args.config)
    if c['mode']!='approximate' or args.output.exists():raise ValueError('New explicit approximate threshold-analysis root required')
    out=args.output;out.mkdir();pins={str(args.config):sha(args.config)}
    if c.get('prepared_populations'):
        prepared=Path(c['prepared_populations']);vb=checked(prepared/'validation_branches.json',pins);vs=checked(prepared/'validation_scores.json',pins)
        va=[];checked(prepared/'deployment.json',pins);checked(prepared/'fit_metadata.json',pins)
    else:
        vb,vs,va=icdar_validation(c,pins);vb['total_text'],vs['total_text']=totaltext(c,'validation',pins)
    choices,vcurves,vbands,shared=validation_choices(c,vb,vs)
    frozen=dict(mode='approximate',claim_label='approximate reproduction',config=c,policies=choices,shared_validation_cuts=shared,
        validation_annotations_audit=va,validation_input_sha256=pins.copy(),test_used_for_choices=False,no_training=True)
    save(out/'validation_choices_frozen.json',frozen);save(out/'validation_curves.json',vcurves);save(out/'validation_bands.json',vbands)
    save(out/'validation_branches.json',vb);save(out/'validation_scores.json',vs)
    # Complete TEST frontiers and best-TEST existence checks are accessed only after validation choices are frozen.
    root=Path(c['paths']['icdar_transfer'])
    if c.get('prepared_populations'):
        prepared=Path(c['prepared_populations']);tb=checked(prepared/'test_branches.json',pins);ts=checked(prepared/'test_scores.json',pins)
    else:tb=checked(root/'branches.json',pins);ts=checked(root/'test_scores.json',pins)
    if c.get('score_mode')=='normalized_utility':
        records=checked(root/'evaluation_records.json',pins)
        ts={ds:{e:{s:observable_length_rescale(v,records[ds][s]['light'][e]['rows']) for s,v in values.items()}
            for e,values in experts.items()} for ds,experts in ts.items()}
    if not c.get('prepared_populations'):tb['total_text'],ts['total_text']=totaltext(c,'test',pins)
    result={};curves={};bands={}
    for ds in c['datasets']:
        result[ds]={};curves[ds]={};bands[ds]={}
        for expert in c['experts']:
            b,s=tb[ds][expert],ts[ds][expert];curve=threshold_curve(b,s);curves[ds][expert]=curve
            band=simultaneous_bootstrap(curve,b,s,c['bootstrap_draws'],c['seed']);bands[ds][expert]=band
            policies={name:dict(threshold=p['threshold'],validation=p['validation'],test=apply_cut(curve,p['threshold'])) for name,p in choices[ds][expert].items()}
            global_cut=shared[expert]['threshold'];policies['shared_validation_no_loss']=dict(threshold=global_cut,test=apply_cut(curve,global_cut))
            source=choices['total_text'][expert]['empirical_loss_0.0']['threshold'];policies['source_totaltext_validation_no_loss']=dict(threshold=source,test=apply_cut(curve,source))
            best=maximum_f1(curve);best_index=curve.index(best)
            no_loss=minimum_cost(curve,curve[0]);ids=sorted(s)
            half=route_summary(b,ids,ranked_ids(s,ids,.5))
            lower=band['lower_gain'][best_index]
            # The algebraic identity is checked against every reported operational route.
            for name,p in policies.items():
                accepted=[] if p['threshold'] is None else [sid for sid,v in s.items() if v is not None and v>=p['threshold']]
                identity=gain_identity(b,accepted)
                if identity['positive']!=(fraction(p['test'])>fraction(curve[0])):raise ValueError('Hybrid gain identity failed')
            result[ds][expert]=dict(all_deepsolo=curve[0],validation_selected=policies,exact_half=half,
                test_descriptive_best=best,test_descriptive_minimum_deepsolo_at_no_loss=no_loss,
                observed_test_threshold_exceeds_deepsolo=fraction(best)>fraction(curve[0]),
                test_best_simultaneous_lower_gain=lower,test_best_pointwise_gain95=band['pointwise_gain95'][best_index],
                pareto_points=pareto(curve),curves_evaluated=len(curve),
                warning='TEST-best cuts are descriptive and cannot be promoted to validation choices; the band is conditional and approximate.')
            print(ds,expert,'baseline',curve[0]['f1'],'best_TEST',best['f1'],'D/N',str(best['deepsolo_images'])+'/'+str(best['total_images']),
                'percent',best['deepsolo_percent'],'noLossD',no_loss['deepsolo_images'],'validationNoLoss',policies['empirical_loss_0.0']['test']['f1'],flush=True)
    common={}
    for expert in c['experts']:
        family=common_thresholds({ds:curves[ds][expert] for ds in c['datasets']});gains=[p for p in family if p['strict_gain_all']]
        common[expert]=dict(exists_test_cut_with_strict_gain_all=bool(gains),
            test_descriptive_minimum_fallback_no_loss_all=min((p for p in family if p['no_loss_all']),key=lambda p:p['deepsolo_images']),
            test_descriptive_minimum_fallback_strict_gain_all=min(gains,key=lambda p:p['deepsolo_images']) if gains else None,
            shared_validation_cut=shared[expert])
    report=dict(status='SUCCESS',mode='approximate',claim_label='approximate reproduction',score_mode=c.get('score_mode','absolute_utility'),datasets=result,common_threshold=common,
        frozen_choices_sha256=sha(out/'validation_choices_frozen.json'),input_sha256=pins,
        operational_model='Same frozen30-feature Ridge; thresholds and objective selected only on the declared validation populations.',
        limitations='Approximate lexical metric, historical TEST/source-model exposure, small validation, single fixed expert/gate, external pretraining overlap. No universal population-performance guarantee or full online latency measurement.')
    save(out/'test_curves.json',curves);save(out/'test_bands.json',bands);save(out/'test_branches.json',tb);save(out/'test_scores.json',ts);save(out/'results.json',report)
    with (out/'test_frontiers.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','expert','threshold','f1','tp','fp','fn','deepsolo_images','total_images','deepsolo_percent','accepted_images']);writer.writeheader()
        for ds,experts in curves.items():
            for expert,curve in experts.items():
                for point in curve:writer.writerow(dict(dataset=ds,expert=expert,**{k:point[k] for k in writer.fieldnames if k not in ['dataset','expert']}))
    save(out/'completion.json',dict(status='SUCCESS',results_sha256=sha(out/'results.json')))

if __name__=='__main__':main()
