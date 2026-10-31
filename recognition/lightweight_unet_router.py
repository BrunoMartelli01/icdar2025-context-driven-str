"""Approximate small image gates with explicit feature cost and frozen calibration."""
from __future__ import annotations

import math
import numpy as np


def ocr_summary(rows, image_wh):
    """Same eighteen inference-only descriptors as the qualified token baseline."""
    w,h=image_wh
    if min(w,h)<=0:raise ValueError('Positive image dimensions required')
    native=[r['confidence'] for r in rows];det=[r['detector_score'] for r in rows]
    area=[(r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1])/(w*h) for r in rows]
    lengths=[len(r['raw_text']) for r in rows]
    result=[math.log1p(len(rows)),math.log(w/h),math.log1p(sum(lengths)),float(any(not r['raw_text'].strip() for r in rows))]
    for values in (native,det,area,lengths):
        result.extend([float(np.mean(values)) if values else 0.,float(min(values)) if values else 0.,float(max(values)) if values else 0.])
    result.extend([sum(1-v for v in native),sum(area)])
    return np.asarray(result,np.float64)


def token_summary(rows,low_probability_thresholds):
    values=[p for r in rows for p in r['token_probabilities']]
    minima=[min(r['token_probabilities']) for r in rows if r['token_probabilities']]
    if not values:return np.zeros(8,np.float64)
    p=np.asarray(values,np.float64)
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):raise ValueError('Invalid native token probabilities')
    if len(low_probability_thresholds)!=2 or not all(0<t<1 for t in low_probability_thresholds):
        raise ValueError('Two explicit native token-probability cutoffs required')
    return np.asarray([p.mean(),p.min(),np.quantile(p,.1),np.quantile(p,.25),
                       *[np.mean(p<t) for t in low_probability_thresholds],np.mean(minima),np.min(minima)],np.float64)


def mask_summary(probabilities,image_wh,rows,settings):
    from recognition.text_coverage_router import coverage_features
    f=coverage_features(probabilities,image_wh,[r['bbox'] for r in rows],**settings)
    side=probabilities.shape[-1];area=side*side;result=[]
    for head in ('text','kernel'):
        result.extend([f[head+'_coverage'],f[head+'_foreground_mass']/area,
                       f[head+'_uncovered_pixels']/area,math.log1p(f[head+'_uncovered_components']),
                       f[head+'_largest_uncovered_component']/area,float(f[head+'_mask_empty'])])
    return np.asarray(result,np.float64)


def context_summary(arrays,global_patch_count,groups=('local','global','blip')):
    result={}
    if 'local' in groups:
        text=arrays['word_text'];image=arrays['word_image']
        if text.shape!=image.shape:raise ValueError('Unpaired word context')
        if len(text):
            denominator=np.linalg.norm(text,axis=1)*np.linalg.norm(image,axis=1)
            cosine=(text*image).sum(1)/np.maximum(denominator,1e-12)
            result['local']=np.asarray([cosine.mean(),cosine.min(),cosine.max(),np.quantile(cosine,.1),np.quantile(cosine,.9),cosine.std()],np.float64)
        else:result['local']=np.zeros(6,np.float64)
    if 'global' in groups:
        # Only original global-view patches; neither tiles nor their extra online calls.
        valid=arrays['patch_valid'][:global_patch_count]
        if not valid.any():raise ValueError('No valid global image patches')
        result['global_']=arrays['patches'][:global_patch_count][valid].mean(0).astype(np.float64)
    if 'blip' in groups:result['blip']=arrays['queries'].mean(0).astype(np.float64)
    return result


def paired_utility(branch,reference_f1):
    return 2*(branch['light']['tp']-branch['deepsolo']['tp'])-reference_f1*(branch['light']['detected_words']-branch['deepsolo']['detected_words'])


def ranked(scores,ids,fraction):
    if not 0<=fraction<=1 or set(scores)!=set(ids):raise ValueError('Invalid routing population')
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(ids)*fraction)]


def conformal_correction(predicted,actual,alpha):
    """One-sided split-conformal nonconformity order statistic, including infinity."""
    predicted=np.asarray(predicted,np.float64);actual=np.asarray(actual,np.float64)
    if predicted.shape!=actual.shape or predicted.ndim!=1 or not len(actual) or not 0<alpha<1:
        raise ValueError('Finite paired calibration scores and alpha in(0,1) required')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():raise ValueError('Nonfinite calibration data')
    k=math.ceil((len(actual)+1)*(1-alpha))
    return float(np.sort(predicted-actual)[k-1]) if k<=len(actual) else math.inf


def lower_bound_accept(scores,correction):
    """A strict positive lower bound; zero utility abstains."""
    if math.isnan(correction):raise ValueError('NaN conformal correction')
    return sorted(s for s,v in scores.items() if v is not None and v-correction>0)


def calibration_threshold(scores,branches,ids,maximum_f1_loss,summarize):
    """Maximize CAL acceptance under an empirical aggregate F1-loss budget."""
    if set(scores)!=set(ids) or set(branches)!=set(ids) or maximum_f1_loss<0:
        raise ValueError('Calibration receives exactly its declared population')
    baseline=summarize(branches,ids,[]);best=dict(threshold=None,summary=baseline)
    for threshold in sorted({v for v in scores.values() if v is not None},reverse=True):
        accepted=[s for s in ids if scores[s] is not None and scores[s]>=threshold]
        summary=summarize(branches,ids,accepted)
        if summary['f1']+maximum_f1_loss+1e-15>=baseline['f1']:
            if (summary['accepted_images'],summary['f1'])>(best['summary']['accepted_images'],best['summary']['f1']):
                best=dict(threshold=float(threshold),summary=summary)
    return dict(**best,maximum_f1_loss=maximum_f1_loss,calibration_ids=sorted(ids),
                guarantee='Empirical CAL micro-F1 constraint only, no population guarantee')


def threshold_accept(scores,threshold):
    return [] if threshold is None else sorted(s for s,v in scores.items() if v is not None and v>=threshold)


def feature_cost(groups):
    """Additional frozen encoder families; OCR and masks come from the fixed expert."""
    clip=bool(set(groups)&{'local','global'});blip='blip' in groups
    return dict(additional_encoder_families=int(clip)+int(blip),clip_required=clip,blip_required=blip,
                requires_global_clip='global' in groups,requires_word_clip='local' in groups,
                requires_unet_maps='mask' in groups)


def fit_head(groups,x,y,weights,settings,seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler().fit(x);z=scaler.transform(x)
    if settings['kind']=='ridge':
        regressor=Ridge(alpha=settings['alpha']).fit(z,y,sample_weight=weights)
        stored_scalar_count=int(regressor.coef_.size+1+2*x.shape[1])
    elif settings['kind']=='hgb':
        regressor=HistGradientBoostingRegressor(max_iter=settings['iterations'],max_leaf_nodes=settings['leaves'],
            l2_regularization=settings['l2'],min_samples_leaf=settings['minimum_leaf'],
            early_stopping=False,random_state=seed).fit(z,y,sample_weight=weights)
        stored_scalar_count=int(2*x.shape[1]+1+sum(len(p.nodes)+np.count_nonzero(~p.nodes['is_leaf'].astype(bool)) for row in regressor._predictors for p in row))
    else:raise ValueError('Unknown explicit small head')
    return dict(groups=list(groups),scaler=scaler,regressor=regressor,settings=settings,
                stored_scalar_count=stored_scalar_count,feature_dimension=x.shape[1],cost=feature_cost(groups))


def predict_head(bundle,x):
    result=bundle['regressor'].predict(bundle['scaler'].transform(x))
    if not np.isfinite(result).all():raise ValueError('Nonfinite small-head prediction')
    return result


def predict_observation_gate(deployment,rows,image_wh,*,probabilities=None,context_arrays=None):
    """Online single-image boundary: only selected inference features are accessed."""
    settings=deployment['feature_settings'];bundle=deployment['bundle'];values=[]
    context=None
    for group in bundle['groups']:
        if group=='ocr':value=ocr_summary(rows,image_wh)
        elif group=='tokens':value=token_summary(rows,settings['token_probability_cutoffs'])
        elif group=='mask':
            if probabilities is None:raise ValueError('Selected gate requires predicted U-Net maps')
            value=mask_summary(probabilities,image_wh,rows,settings['map_features'])
        else:
            if context_arrays is None:raise ValueError('Selected gate requires its declared frozen context encoder features')
            if context is None:context=context_summary(context_arrays,settings['global_patch_count'],bundle['groups'])
            value=context['global_' if group=='global' else group]
        if group in deployment['projections']:value=deployment['projections'][group].transform(value[None])[0]
        values.append(value)
    x=np.concatenate(values)[None];prediction=float(predict_head(bundle,x)[0]);eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
    normalized=prediction if eligible else None;absolute=prediction*(len(rows)+1) if eligible else None
    decisions={}
    for name,p in deployment['policies'].items():
        if name.startswith('conformal'):
            decisions[name]=eligible and p['correction'] is not None and normalized-p['correction']>0
        else:decisions[name]=eligible and p['threshold'] is not None and absolute>=p['threshold']
    return dict(normalized_utility=normalized,predicted_utility=absolute,use_unet=decisions,
                additional_encoder_cost=bundle['cost'],ground_truth_used=False)
