"""Fit deployable image/word selectors on TRAIN; measure untouched TEST outcomes."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import time
import traceback

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from segmentation.word_slot_data import read,write,sha,sample_key
from recognition.word_slot_evaluation import (measure,aggregate,select,image_features,
                                              word_features,merge_words)
from recognition.scene_text_v1_evaluation import box_iou
from recognition.word_slot_text import normalize_word_text,CASEFOLD
from scripts.evaluate_word_slot_methods import raw_path,bootstrap_delta

REPO=Path(__file__).resolve().parents[1]


def threshold_selection(records,predictions,config):
    sweep=[];normalization=config['evaluation'].get('text_normalization',CASEFOLD)
    for presence,confidence in itertools.product(config['evaluation']['presence_thresholds'],config['evaluation']['confidence_thresholds']):
        rows=[measure(select(predictions[r['sample_id']],presence,confidence),r['ground_truth'],config['evaluation']['iou_threshold'],normalization) for r in records]
        metric=aggregate(rows);sweep.append((metric['f1'],-metric['fp'],metric['tp'],-presence,-confidence,presence,confidence))
    chosen=max(sweep);return chosen[-2:]


def image_router(calibration,raw,deep,config,router_config):
    normalization=router_config.get('text_normalization',CASEFOLD)
    folds=list(KFold(n_splits=min(router_config['folds'],len(calibration)),shuffle=True,random_state=router_config['seed']).split(calibration))
    full_thresholds=threshold_selection(calibration,raw,config)
    final_predictions={r['sample_id']:select(raw[r['sample_id']],*full_thresholds) for r in calibration}
    final_x=np.array([image_features(final_predictions[r['sample_id']],r['image_wh']) for r in calibration])
    deep_metrics=[measure(deep[r['sample_id']],r['ground_truth'],router_config['iou_threshold'],normalization) for r in calibration]
    base_f1=aggregate(deep_metrics)['f1'];final_metric=[measure(final_predictions[r['sample_id']],r['ground_truth'],router_config['iou_threshold'],normalization) for r in calibration]
    final_y=np.array([s['tp']-d['tp']-.5*base_f1*(s['predictions']-d['predictions']) for s,d in zip(final_metric,deep_metrics)])
    candidates=[]
    for alpha in router_config['ridge_alphas']:
        oof=np.zeros(len(calibration));student_metrics=[None]*len(calibration);fold_thresholds=[]
        for train_ids,val_ids in folds:
            train=[calibration[i] for i in train_ids];thresholds=threshold_selection(train,raw,config);fold_thresholds.append(list(thresholds))
            predicted={r['sample_id']:select(raw[r['sample_id']],*thresholds) for r in calibration}
            x=np.array([image_features(predicted[r['sample_id']],r['image_wh']) for r in calibration])
            sm=[measure(predicted[r['sample_id']],r['ground_truth'],router_config['iou_threshold'],normalization) for r in calibration]
            d_f1=aggregate([deep_metrics[i] for i in train_ids])['f1']
            y=np.array([s['tp']-d['tp']-.5*d_f1*(s['predictions']-d['predictions']) for s,d in zip(sm,deep_metrics)])
            model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x[train_ids],y[train_ids]);oof[val_ids]=model.predict(x[val_ids])
            for i in val_ids:student_metrics[i]=sm[i]
        thresholds=[float(oof.min()-1),float(oof.max()+1),0.]+[float(np.quantile(oof,q)) for q in router_config['image_threshold_quantiles']]
        for threshold in sorted(set(thresholds)):
            chosen=[s if score>=threshold else d for s,d,score in zip(student_metrics,deep_metrics,oof)]
            metric=aggregate(chosen);candidates.append({'alpha':alpha,'threshold':threshold,'oof':oof.tolist(),
                'metric':metric,'deepsolo_images':int((oof<threshold).sum()),'fold_student_thresholds':fold_thresholds})
    chosen=max(candidates,key=lambda c:(c['metric']['f1'],-c['deepsolo_images'],-c['alpha'],-c['threshold']))
    model=make_pipeline(StandardScaler(),Ridge(alpha=chosen['alpha']));model.fit(final_x,final_y)
    return model,{'chosen':chosen,'full_student_thresholds':list(full_thresholds),
           'scaler_mean':model[0].mean_.tolist(),'scaler_scale':model[0].scale_.tolist(),
           'ridge_coefficients':model[1].coef_.tolist(),'ridge_intercept':float(model[1].intercept_),
           'candidates':[{k:v for k,v in c.items() if k not in ('oof','fold_student_thresholds')} for c in candidates]}


def quality_training(records,student,deep,iou,normalization=CASEFOLD):
    x=[];y=[];bounds={}
    for record in records:
        sid=record['sample_id'];start=len(x)
        for expert,predictions,other in ((0,student[sid],deep[sid]),(1,deep[sid],student[sid])):
            for p in predictions:
                x.append(word_features(p,other,record['image_wh'],expert,normalization))
                compared=normalize_word_text(p['text'],normalization)
                correct=any(p.get('bbox') is not None and (compared or normalization==CASEFOLD) and compared==normalize_word_text(g['text'],normalization) and
                            box_iou(p['bbox'],g['bbox'])>=iou for g in record['ground_truth'])
                y.append(int(correct))
        bounds[sid]=(start,len(x),len(student[sid]))
    return np.asarray(x),np.asarray(y),bounds


def fit_quality(x,y,c):
    if not len(y):raise ValueError('No calibration word candidates')
    if len(np.unique(y))==1:
        # Defined empirical constant model for a degenerate training fold. This
        # condition is reported explicitly; it is not recovery from an error.
        return float(y.mean())
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c['quality_logistic_C'],max_iter=c['quality_logistic_max_iter'],random_state=c['seed']))
    model.fit(x,y);return model


def quality_probability(model,x):
    return np.full(len(x),model) if isinstance(model,float) else model.predict_proba(x)[:,1]


def word_router(calibration,student,deep,c):
    normalization=c.get('text_normalization',CASEFOLD)
    x,y,bounds=quality_training(calibration,student,deep,c['iou_threshold'],normalization);oof=np.empty(len(y));degenerate=0
    for train_ids,val_ids in KFold(n_splits=min(c['folds'],len(calibration)),shuffle=True,random_state=c['seed']).split(calibration):
        train_indices=[i for n in train_ids for i in range(*bounds[calibration[n]['sample_id']][:2])]
        val_indices=[i for n in val_ids for i in range(*bounds[calibration[n]['sample_id']][:2])]
        model=fit_quality(x[train_indices],y[train_indices],c);degenerate+=isinstance(model,float)
        oof[val_indices]=quality_probability(model,x[val_indices])
    trials=[]
    for append,replace,margin in itertools.product(c['append_thresholds'],c['replacement_thresholds'],c['replacement_margins']):
        metrics=[];changes=0
        for record in calibration:
            sid=record['sample_id'];start,end,n=bounds[sid]
            predictions,decisions=merge_words(student[sid],deep[sid],oof[start:start+n],oof[start+n:end],
                        append_threshold=append,replace_threshold=replace,margin=margin,conflict_iou=c['conflict_iou'],normalization=normalization)
            metrics.append(measure(predictions,record['ground_truth'],c['iou_threshold'],normalization));changes+=len(decisions)
        trials.append({'append_threshold':append,'replace_threshold':replace,'margin':margin,
                       'metric':aggregate(metrics),'changes':changes})
    chosen=max(trials,key=lambda t:(t['metric']['f1'],-t['changes'],t['append_threshold'],t['replace_threshold'],t['margin']))
    model=fit_quality(x,y,c)
    parameters={'kind':'empirical_constant','probability':model} if isinstance(model,float) else {
        'kind':'standard_scaler_logistic','scaler_mean':model[0].mean_.tolist(),'scaler_scale':model[0].scale_.tolist(),
        'logistic_coefficients':model[1].coef_.tolist(),'logistic_intercept':model[1].intercept_.tolist(),'classes':model[1].classes_.tolist()}
    return model,{'chosen':chosen,'sweep':trials,'parameters':parameters,'oof_degenerate_folds':degenerate,
                  'training_images':len(calibration),'training_candidates':len(y),'positive_candidates':int(y.sum()),
                  'pointwise_training_labels':'any exact text + bbox match; actual routed metrics always use one-to-one matching'}


def analyze(root,router_config,arms,phases):
    protocol=read(root/'protocol.json');config=protocol['config'];index=protocol['index'];calibration=index['calibration'];summary={};audits={};output_rows={}
    normalization=router_config.get('text_normalization',CASEFOLD)
    if normalization!=config['evaluation'].get('text_normalization',CASEFOLD):raise ValueError('Router and evaluation text normalization disagree')
    source_hashes={};start=time.perf_counter()
    for arm in arms:
        # TRAIN and TEST reuse sample_id strings for DIFFERENT image bytes.
        # Fit gates before even loading evaluation/test predictions; never join
        # across splits using sample_id alone.
        def load_phase(phase):
            student={};deep={}
            for record in index[phase]:
                sid=record['sample_id'];p=raw_path(root,arm,phase,sid);student[sid]=read(p)['predictions'];source_hashes[str(p.relative_to(root))]=sha(p)
                p=root/'deepsolo'/phase/(sample_key(sid)+'.json');deep[sid]=read(p)['predictions'];source_hashes[str(p.relative_to(root))]=sha(p)
            return student,deep
        calibration_student,calibration_deep=load_phase('calibration')
        image_model,image_audit=image_router(calibration,calibration_student,calibration_deep,config,router_config)
        word_model,word_audit=word_router(calibration,calibration_student,calibration_deep,router_config)
        audits[arm]={'image':image_audit,'word':word_audit}
        for phase in phases:
            student,deep=load_phase(phase)
            records=index[phase];p,c=image_audit['full_student_thresholds']
            image_student={r['sample_id']:select(student[r['sample_id']],p,c) for r in records}
            scores=image_model.predict(np.array([image_features(image_student[r['sample_id']],r['image_wh']) for r in records]))
            threshold=image_audit['chosen']['threshold'];unconstrained=scores>=threshold
            fixed=np.zeros(len(records),bool);fixed[np.argsort(-scores,kind='stable')[:int(len(records)*router_config['fixed_image_unet_fraction'])]]=True
            per={key:[] for key in ('image','image_fixed_half','word','deepsolo')}
            for number,record in enumerate(records):
                sid=record['sample_id'];ds=deep[sid];us=student[sid];chosen=word_audit['chosen']
                wx=np.array([word_features(p,ds,record['image_wh'],0,normalization) for p in us]+[word_features(p,us,record['image_wh'],1,normalization) for p in ds])
                probability=quality_probability(word_model,wx) if len(wx) else np.empty(0)
                mixed,decisions=merge_words(us,ds,probability[:len(us)],probability[len(us):],
                     append_threshold=chosen['append_threshold'],replace_threshold=chosen['replace_threshold'],margin=chosen['margin'],conflict_iou=router_config['conflict_iou'],normalization=normalization)
                predictions={'image':image_student[sid] if unconstrained[number] else ds,
                             'image_fixed_half':image_student[sid] if fixed[number] else ds,'word':mixed,'deepsolo':ds}
                for kind,values in predictions.items():
                    per[kind].append({'sample_id':sid,**measure(values,record['ground_truth'],router_config['iou_threshold'],normalization),
                        'action':('unet' if (unconstrained[number] if kind=='image' else fixed[number]) else 'deepsolo') if kind.startswith('image') else kind,
                        'router_score':float(scores[number]) if kind.startswith('image') else None})
                write(root/'router_decisions'/arm/phase/(sample_key(sid)+'.json'),{
                      'sample_id':sid,'image_router_score':float(scores[number]),'image_uses_unet':bool(unconstrained[number]),
                      'image_fixed_half_uses_unet':bool(fixed[number]),'word_router_decisions':decisions,
                      'word_student_probability':probability[:len(us)].tolist(),'word_deepsolo_probability':probability[len(us):].tolist(),
                      'word_routed_predictions':mixed,'ground_truth_used_for_decision':False})
            table={k:aggregate(rows) for k,rows in per.items()}
            for kind,table_row in table.items():
                table_row['logical_deepsolo_calls']=sum(r['action']=='deepsolo' for r in per[kind]) if kind.startswith('image') else len(records)
                if phase=='test':table_row['paired_bootstrap_delta_f1_95_interval']=bootstrap_delta(per[kind],per['deepsolo'],config['evaluation']['bootstrap_draws'],config['seed'])
            summary.setdefault(phase,{})[arm]=table;output_rows.setdefault(phase,{})[arm]=per
        print('ROUTERS',arm,flush=True)
    write(root/'router_fit_audits.json',audits);write(root/'router_results.json',summary);write(root/'router_per_image.json',output_rows)
    write(root/'router_execution.json',{'status':'SUCCESS','config':router_config,'config_digest':__import__('hashlib').sha256(json_bytes(router_config)).hexdigest(),
          'sklearn':sklearn.__version__,'seconds':time.perf_counter()-start,'prediction_source_sha256':source_hashes,
          'code_sha256':{str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'recognition/word_slot_evaluation.py')},
          'selection_uses_test':False,'latency_scope':'cached branch replay; logical selective calls only, not a measured end-to-end deployment'})


def json_bytes(value):
    import json
    return json.dumps(value,sort_keys=True).encode()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arms',nargs='+');parser.add_argument('--phases',nargs='+',default=['evaluation','test'],choices=['evaluation','test'])
    args=parser.parse_args();c=read(args.config);root=args.output.resolve()
    if REPO/'configs/approximate' not in args.config.resolve().parents or c['reproduction_mode']!=args.mode or c['experiment_kind']!='word_slot_router_v1':raise ValueError('Explicit APPROXIMATE router configuration required')
    arms=args.arms or ['pixel_control','pixel_pil',*read(root/'protocol.json')['config']['arms']]
    try:analyze(root,c,arms,args.phases)
    except Exception as error:
        write(root/('failure_router_'+str(time.time_ns())+'.json'),{'status':'FAILED','type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
