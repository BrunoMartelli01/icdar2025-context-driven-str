"""Shared APPROXIMATE OCR/correction/risk/count-utility pipeline.

All input features are observable before invoking the fallback. Dataset-specific
artifacts are supplied as data; dataset identity never branches the predictor.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from recognition.lightweight_unet_router import ocr_summary, token_summary
from recognition.lexical_budget_router import lexical_features
from recognition.text_coverage_router import coverage_features


def normalize(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def validation_roles(partitions,policy):
    result={k:list(v) for k,v in partitions.items()}
    if policy=='pooled':
        if set(result['selection'])&set(result['calibration']):raise ValueError('Validation role overlap')
        result['selection']=sorted(result['selection']+result['calibration']);result['calibration']=[]
    elif policy!='separate':raise ValueError('Unknown validation policy')
    return result


def counts(rows, truth, ignored):
    if len(rows)!=len(ignored):raise ValueError('Prediction/ignore cardinalities differ')
    care=[r for r,skip in zip(rows,ignored) if not skip]
    a=Counter(normalize(r['raw_text']) for r in care if normalize(r['raw_text']))
    b=Counter(normalize(r['text']) for r in truth if normalize(r['text']))
    tp=sum((a&b).values())
    return dict(tp=tp,fp=len(care)-tp,fn=len(truth)-tp,detected_words=len(care),
        gt_words=len(truth),all_outputs=len(rows),ignored_outputs=sum(ignored))


def micro(rows):
    rows=list(rows);fields=('tp','fp','fn','detected_words','gt_words','all_outputs','ignored_outputs')
    total={k:sum(v[k] for v in rows) for k in fields}
    den=2*total['tp']+total['fp']+total['fn']
    if total['tp']+total['fp']!=total['detected_words'] or total['tp']+total['fn']!=total['gt_words']:
        raise ValueError('Inconsistent count boundary')
    return dict(total,images=len(rows),f1=2*total['tp']/den if den else 0.)


def choose_light(scores,keys,fallback_count,eligible=None):
    if set(scores)!=set(keys) or not 0<=fallback_count<=len(scores):raise ValueError('Invalid exact budget')
    if not all(np.isfinite(v) for v in scores.values()):raise ValueError('Nonfinite routing score')
    eligible=set(scores) if eligible is None else set(eligible)
    if not eligible<=set(scores):raise ValueError('Unknown eligible image')
    n=len(scores)-fallback_count
    if len(eligible)<n:raise ValueError('Guard cannot satisfy exact fallback budget')
    return sorted(eligible,key=lambda s:(-scores[s],keys[s],s))[:n]


def route(branches,accepted):
    accepted=set(accepted)
    if not accepted<=set(branches):raise ValueError('Unknown accepted image')
    n=len(branches)-len(accepted)
    return dict(combined=micro(v['light' if s in accepted else 'deepsolo'] for s,v in branches.items()),
        accepted_ids=sorted(accepted),deepsolo_images=n,total_images=len(branches),
        deepsolo_percentage=100*n/len(branches),scope='logical cached routing; no online latency measurement')


def grouped_folds(ids,keys,number,seed):
    groups={}
    for sid in ids:groups.setdefault(keys[sid],[]).append(sid)
    if len(groups)<number:raise ValueError('Too few independent RGB groups')
    order=sorted(groups,key=lambda k:hashlib.sha256(f'{seed}:{k}'.encode()).hexdigest())
    folds=[[] for _ in range(number)]
    for i,key in enumerate(order):folds[i%number].extend(groups[key])
    return [sorted(v) for v in folds]


def map_features(probabilities,image_wh,rows,settings):
    """Text and optional kernel coverage; missing channels are explicit, never fabricated."""
    p=np.asarray(probabilities)
    if p.ndim!=3 or p.shape[0] not in (1,2,3) or p.shape[1]!=p.shape[2]:
        raise ValueError('Expected square text / optional kernel probability maps')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid map probabilities')
    # The existing coverage primitive requires text and kernel. For a single
    # text map its first six outputs do not depend on the second argument;
    # a zero placeholder is excluded and marked missing in the returned schema.
    pair=p[:2] if p.shape[0]>=2 else np.concatenate([p,np.zeros_like(p)],axis=0)
    f=coverage_features(pair,image_wh,[r['bbox'] for r in rows],**settings)
    area=p.shape[-1]**2;out=[]
    for index,head in enumerate(('text','kernel')):
        if index>=p.shape[0]:out.extend([0.]*6);continue
        out.extend([f[head+'_coverage'],f[head+'_foreground_mass']/area,
            f[head+'_uncovered_pixels']/area,math.log1p(f[head+'_uncovered_components']),
            f[head+'_largest_uncovered_component']/area,float(f[head+'_mask_empty'])])
    return np.asarray(out+[1.,float(p.shape[0]>=2)],np.float64)


def geometry_features(rows,image_wh):
    w,h=image_wh;n=max(len(rows),1);boxes=[r['bbox'] for r in rows]
    widths=[max(0,b[2]-b[0])/w for b in boxes];heights=[max(0,b[3]-b[1])/h for b in boxes]
    aspects=[math.log(max(b[2]-b[0],1)/max(b[3]-b[1],1)) for b in boxes]
    values=[]
    for a in (widths,heights,aspects):values.extend([float(np.mean(a)) if a else 0.,float(np.std(a)) if a else 0.,min(a) if a else 0.,max(a) if a else 0.])
    values.extend([sum(normalize(r['raw_text']).isdigit() for r in rows)/n,
        sum(not normalize(r['raw_text']) for r in rows)/n])
    return np.asarray(values,np.float64)


def word_features(rows,traces,image_wh):
    w,h=image_wh;values=[]
    if len(rows)!=len(traces):raise ValueError('Every raw word needs its lexical trace')
    for r,t in zip(rows,traces):
        b=r['bbox'];text=normalize(r['raw_text']);p=np.asarray(r['token_probabilities'],float)
        match=t['potential_match'];length=max(len(text),1)
        values.append([r['confidence'],r['detector_score'],r.get('sequence_confidence_native',r['confidence']),
            float(p.mean()) if len(p) else 0.,float(p.min()) if len(p) else 0.,
            float(np.quantile(p,.1)) if len(p) else 0.,float(np.mean(p<.9)) if len(p) else 0.,
            math.log1p(len(text)),sum(c.isdigit() for c in text)/length,float(not text),
            max(0,b[2]-b[0])/w,max(0,b[3]-b[1])/h,(b[0]+b[2])/(2*w),(b[1]+b[3])/(2*h),
            float(match is not None),float(match['distance']) if match is not None else 2.,
            float(t['accepted']),float(t['changed'])])
    return np.asarray(values,np.float64).reshape(-1,18)


def risk_features(probabilities,traces):
    p=np.asarray(probabilities,float)
    if len(p)!=len(traces) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid word risk')
    accepted=np.array([t['accepted'] for t in traces],bool)
    result=[]
    for subset in (p,p[accepted],p[~accepted]):
        result.extend([float(np.mean(subset)) if len(subset) else 0.,min(subset) if len(subset) else 0.,
            float(np.quantile(subset,.25)) if len(subset) else 0.,float(np.sum(subset)),float(np.sum(1-subset))])
    return np.asarray(result+[float(not len(p))],np.float64)


def image_features(raw,corrected,traces,image_wh,maps):
    return dict(raw_ocr=ocr_summary(raw,image_wh),tokens=token_summary(raw,[.5,.9]),
        post_ocr=ocr_summary(corrected,image_wh),lexical=lexical_features(raw,corrected,traces),
        maps=maps,geometry=geometry_features(raw,image_wh))


def fit_count_head(x,y,weights,recipe,seed):
    if y.shape!=(len(x),2):raise ValueError('Two count-difference targets required')
    scale=StandardScaler().fit(x);z=scale.transform(x)
    if recipe['kind']=='ridge':
        models=[Ridge(alpha=recipe['alpha']).fit(z,y,sample_weight=weights)]
    elif recipe['kind']=='hgb':
        models=[HistGradientBoostingRegressor(max_iter=recipe['iterations'],max_leaf_nodes=recipe['leaves'],
            min_samples_leaf=recipe['minimum_leaf'],l2_regularization=recipe['l2'],early_stopping=False,
            random_state=seed).fit(z,y[:,i],sample_weight=weights) for i in range(2)]
    elif recipe['kind']=='extra_trees':
        models=[ExtraTreesRegressor(n_estimators=recipe['trees'],max_depth=recipe['depth'],
            min_samples_leaf=recipe['minimum_leaf'],random_state=seed,n_jobs=1).fit(z,y,sample_weight=weights)]
    else:raise ValueError('Unknown explicit common head recipe')
    return dict(scaler=scale,models=models,recipe=recipe)


def predict_counts(bundle,x):
    z=bundle['scaler'].transform(x)
    result=bundle['models'][0].predict(z) if len(bundle['models'])==1 else np.stack([m.predict(z) for m in bundle['models']],axis=1)
    if result.shape!=(len(x),2) or not np.isfinite(result).all():raise ValueError('Invalid predicted count differences')
    return result


def fit_word_risk(x,y,seed,settings):
    if len(np.unique(y))<2:raise ValueError('Word-risk fitting needs both label classes')
    model=HistGradientBoostingClassifier(max_iter=settings['iterations'],max_leaf_nodes=settings['leaves'],
        min_samples_leaf=settings['minimum_leaf'],l2_regularization=settings['l2'],
        early_stopping=False,random_state=seed)
    return model.fit(x,y)
