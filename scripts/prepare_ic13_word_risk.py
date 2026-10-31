"""Image-group OOF word confidence and discarded-candidate risk, FIT-only training."""

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from localize.independent_word_slots import overlap_iou
from recognition.lexical_budget_router import correct_rows
from scripts.run_lexical_budget_router import lexicon
from scripts.icdar_budget_data import annotation,read,key,sha
from scripts.run_yolo_f1_quality_study import seal


def word_features(row,wh):
    raw=row['original_raw_text'];text=row['raw_text'];p=np.asarray(row['token_probabilities']);logs=np.asarray(row['token_log_probabilities']);w,h=wh;a,b,c,d=row['bbox'];n=max(len(text),1)
    return [row['confidence'],row['detector_score'],math.sqrt(row['confidence']*row['detector_score']),math.log1p(len(raw)),math.log1p(len(text)),
        float(p.mean()) if len(p) else 0.,float(p.min()) if len(p) else 0.,float(np.quantile(p,.1)) if len(p) else 0.,float(logs.mean()) if len(logs) else 0.,row['sequence_confidence_native'],
        sum(c.isdigit() for c in text)/n,sum(c.isupper() for c in text)/n,sum(c.isalpha() for c in text)/n,float(raw!=text),(c-a)*(d-b)/(w*h),math.log((c-a)/(d-b)),(a+c)/(2*w),(b+d)/(2*h)]


def aggregate_risk(rows,prob,selected,wh,duplicate_iou):
    norm=lambda s:''.join(c for c in s.casefold() if c.isalnum());ids={r['candidate_id'] for r in selected};kept=[];extra=[]
    for row,value in zip(rows,prob):
        if row['candidate_id'] in ids:kept.append(value)
        elif not any(norm(row['raw_text'])==norm(s['raw_text']) and overlap_iou(row['bbox'],s['bbox'])>duplicate_iou for s in selected):extra.append(value)
    summarize=lambda p:[float(sum(p)),float(np.mean(p)) if p else 0.,float(min(p)) if p else 0.,float(np.quantile(p,.1)) if p else 0.,float(sum(1-x for x in p)),float(sum(x>=.5 for x in p))]
    return summarize(kept)+summarize(extra)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);plan=read(Path(c['original_router_root'])/'resolved_configuration.json');records={r['sample_id']:r for rows in plan['records']['icdar13'].values() for r in rows};fitids=plan['partitions']['icdar13']['fit'];settings=c['word_risk'];dataset={};policy=dict(enabled=True,confidence_cap=1.01,protect_numeric=False,unmatched_policy='source_reject');lexicons={s:lexicon(c,'icdar13',s) for s in ['train','test']};norm=lambda s:''.join(x for x in s.casefold() if x.isalnum())
    for sid,r in records.items():
        source=Path(settings['raw_ocr_root'])/(key(sid)+'.json');raw=read(source);rows,_=correct_rows(raw['rows'],lexicons['train' if ':train:' in sid else 'test'],policy)
        dataset[sid]=dict(rows=rows,x=[word_features(row,r['image_wh']) for row in rows],source_sha256=sha(source))
    for sid in fitids:
        gt,_=annotation(records[sid]);words={norm(g['text']) for g in gt if norm(g['text'])};dataset[sid]['labels']=[int(norm(row['raw_text']) in words) for row in dataset[sid]['rows']]
    def model():return HistGradientBoostingClassifier(max_iter=settings['iterations'],max_leaf_nodes=settings['leaves'],min_samples_leaf=settings['minimum_leaf'],l2_regularization=settings['l2'],early_stopping=False,random_state=c['seed'])
    def train(ids):
        x=np.concatenate([np.asarray(dataset[s]['x']).reshape(-1,18) for s in ids]);y=np.concatenate([dataset[s]['labels'] for s in ids]);m=model();m.fit(x,y);return m
    probabilities={};folds=[]
    for index,(trainix,valix) in enumerate(KFold(n_splits=settings['folds'],shuffle=True,random_state=c['seed']).split(fitids)):
        tr=[fitids[i] for i in trainix];val=[fitids[i] for i in valix];m=train(tr);path=root/'word_models'/('fold_'+str(index)+'.joblib');path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(m,path)
        for sid in val:probabilities[sid]=m.predict_proba(np.asarray(dataset[sid]['x']).reshape(-1,18))[:,1].tolist() if dataset[sid]['rows'] else []
        folds.append(dict(train_ids=tr,heldout_ids=val,path=str(path),sha256=sha(path)))
    full=train(fitids);path=root/'word_models/full_fit.joblib';joblib.dump(full,path)
    for sid in set(records)-set(fitids):probabilities[sid]=full.predict_proba(np.asarray(dataset[sid]['x']).reshape(-1,18))[:,1].tolist() if dataset[sid]['rows'] else []
    values={}
    for sid,r in records.items():
        light=read(Path(c['original_router_root'])/'experts/yolo'/(key(sid)+'.json'))['rows'];selected,_=correct_rows(light,lexicons['train' if ':train:' in sid else 'test'],policy);values[sid]=aggregate_risk(dataset[sid]['rows'],probabilities[sid],selected,r['image_wh'],settings['duplicate_iou'])
    seal(root/'word_risk_features.json',dict(mode='approximate',claim_label='approximate reproduction',values=values,folds=folds,full_fit_ids=fitids,full_model_path=str(path),full_model_sha256=sha(path),scope='Only FIT word labels train models; FIT image descriptors use out-of-fold scores; all other descriptors use full-FIT model, never GT'))
    seal(root/'word_training_evidence.json',dict(dataset=dataset,probabilities=probabilities,fit_ids=fitids,config_sha256=sha(a.config),code_sha256=sha(__file__)))
    print('OOF word risk prepared',len(fitids),'FIT images',len(values),'total images',flush=True)


if __name__=='__main__':main()
