"""GT-free conservative lexicon editing and cached-observation budget heads."""

import math
import numpy as np

from recognition.lightweight_unet_router import ocr_summary,token_summary


def correct_rows(rows,lexicon,policy):
    out=[];trace=[]
    for i,row in enumerate(rows):
        text=row['raw_text'];normalized=''.join(c for c in text if c.isalnum())
        active=policy['enabled'] and row['confidence']<policy['confidence_cap'] and not (policy['protect_numeric'] and normalized.isdigit())
        potential=lexicon.nearest(text)
        match=potential if active else None
        keep=not active or match is not None or policy['unmatched_policy']=='keep_unmatched'
        edited=match['text'] if match is not None else text
        if keep:out.append(dict(row,raw_text=edited,text=edited,original_raw_text=text,source_row=i))
        trace.append(dict(source_row=i,original_text=text,active=active,accepted=keep,match=match,potential_match=potential,
                          changed=keep and edited!=text,confidence=row['confidence']))
    return out,trace


def lexical_features(raw,corrected,trace):
    n=max(len(raw),1);edited=[t for t in trace if t['changed']];matched=[t for t in trace if t['potential_match'] is not None]
    return np.asarray([len(corrected)/n,sum(t['active'] for t in trace)/n,
        len(matched)/n,sum(t['potential_match'] is not None and t['potential_match']['distance']==0 for t in trace)/n,
        sum(t['potential_match'] is not None and t['potential_match']['distance']==1 for t in trace)/n,
        len(edited)/n,sum(not t['accepted'] for t in trace)/n,
        sum(t['potential_match'] is None for t in trace)/n,
        sum(t['confidence'] for t in edited)/n,
        sum(1-t['confidence'] for t in edited),
        sum(t['confidence'] for t in trace if not t['accepted'])/n,
        sum(len(r['raw_text']) for r in corrected)/(1+sum(len(r['raw_text']) for r in raw)),
        float(not raw),float(not corrected),math.log1p(len(matched)),math.log1p(len(edited))],np.float64)


def features(raw,corrected,trace,image_wh,*,use_consensus=False):
    result=dict(raw_ocr=ocr_summary(raw,image_wh),post_ocr=ocr_summary(corrected,image_wh),
                tokens=token_summary(raw,[.5,.9]),lexical=lexical_features(raw,corrected,trace))
    if use_consensus:
        if not raw:result['consensus']=np.zeros(8,np.float64)
        else:
            support=np.asarray([r['scale_support'] for r in raw]);conflicts=np.asarray([r['scale_text_conflicts'] for r in raw]);variation=[np.std(r['scale_joints']) for r in raw]
            result['consensus']=np.asarray([support.mean()/3,support.min()/3,np.mean(support>=2),np.mean(support==3),np.mean(1-1/support),np.mean(variation),np.mean(conflicts>0),np.mean(conflicts)],np.float64)
    return result


def fit(x,y,weights,recipe,seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
    scaler=StandardScaler().fit(x);z=scaler.transform(x)
    if recipe['kind']=='ridge':model=Ridge(alpha=recipe['alpha'])
    elif recipe['kind']=='hgb':model=HistGradientBoostingRegressor(max_iter=recipe['iterations'],max_leaf_nodes=recipe['leaves'],l2_regularization=recipe['l2'],min_samples_leaf=recipe['minimum_leaf'],early_stopping=False,random_state=seed)
    elif recipe['kind']=='extra_trees':model=ExtraTreesRegressor(n_estimators=recipe['trees'],max_depth=recipe['depth'],min_samples_leaf=recipe['minimum_leaf'],random_state=seed,n_jobs=1)
    else:raise ValueError('Unknown explicit head recipe')
    model.fit(z,y,sample_weight=weights)
    return dict(scaler=scaler,model=model,recipe=recipe)


def predict(bundle,x,scale):
    values=bundle['model'].predict(bundle['scaler'].transform(x))*scale
    if not np.isfinite(values).all():raise ValueError('Nonfinite budget router score')
    return values
