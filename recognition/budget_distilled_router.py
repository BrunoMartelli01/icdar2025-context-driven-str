"""Approximate CPU gate from existing OCR, map layout and shared U-Net context.

Only training targets use paired DeepSolo/annotation outcomes. Deployment never
loads DeepSolo, CLIP, BLIP, annotations or validity masks to choose a branch.
"""
from __future__ import annotations

import math
import numpy as np
import cv2


def observation_features(rows, candidates, image_wh, probabilities, settings):
    """Explicit inference observations, with a stable named feature schema."""
    w, h = image_wh
    p = np.asarray(probabilities)
    if min(w, h) <= 0 or p.ndim != 3 or p.shape[0] != 2 or p.shape[1] != p.shape[2]:
        raise ValueError('Positive dimensions and square two-head predicted maps required')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid predicted probabilities')
    if type(settings['grid_side']) is not int or not 1<=settings['grid_side']<=p.shape[-1]:
        raise ValueError('Spatial grid must be a positive integer within the map')
    if any(not math.isfinite(c) or not 0<c<1 for c in settings['map_cuts']+settings['sequence_cuts']):
        raise ValueError('Explicit map and sequence cutoffs must lie within(0,1)')
    side = p.shape[-1]; result = {}; groups = {}
    def add(name, value, group='ocr'):
        if not math.isfinite(float(value)):
            raise ValueError('Nonfinite inference feature: '+name)
        if name in result:
            raise ValueError('Duplicate feature name')
        result[name] = float(value); groups[name] = group
    def stats(name, values, group='ocr'):
        v = np.asarray(values, float)
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite OCR observation')
        for label, value in zip(('mean','min','q10','median','q90','max','std'),
                [v.mean(), v.min(), *np.quantile(v,[.1,.5,.9]), v.max(), v.std()] if len(v) else [0.]*7):
            add(name+'_'+label, value, group)
    add('log_words', math.log1p(len(rows))); add('log_candidates',math.log1p(len(candidates)))
    add('log_aspect',math.log(w/h)); add('log_long_side',math.log(max(w,h)))
    selected = {r['candidate_id'] for r in rows}
    if len(selected) != len(rows) or not selected <= {r['candidate_id'] for r in candidates}:
        raise ValueError('Selected words must have unique raw candidate identities')
    for scope, words in [('kept',rows),('raw',candidates)]:
        fields={name:[] for name in ('sequence','token_min','token_mean','surprisal','detector',
                                     'length','relative_height','aspect','fill','alnum_fraction','digit_fraction')}
        for r in words:
            a,b,c,d = r['bbox']
            if not 0 <= a < c <= w or not 0 <= b < d <= h:
                raise ValueError('Original-frame exclusive word envelope required')
            tokens=np.asarray(r['token_probabilities'],float)
            if not np.isfinite(tokens).all() or np.any(tokens<0) or np.any(tokens>1):
                raise ValueError('Invalid native OCR token probabilities')
            text=r['raw_text']; length=max(len(text),1)
            native=r.get('bbox_native'); native_area=(native[2]-native[0])*(native[3]-native[1]) if native else 1
            values=[r['sequence_confidence_native'],float(tokens.min()) if len(tokens) else 0.,
                    float(tokens.mean()) if len(tokens) else 0.,float(-np.log(np.maximum(tokens,1e-12)).sum()),
                    r['detector_score'],len(text),(d-b)/max(w,h),math.log((c-a)/(d-b)),
                    r.get('foreground_area',0)/max(native_area,1),
                    sum(ch.isalnum() for ch in text)/length,sum(ch.isdigit() for ch in text)/length]
            for name,value in zip(fields,values):fields[name].append(value)
        for name,values in fields.items():stats(scope+'_'+name,values)
        add(scope+'_log_characters',math.log1p(sum(len(r['raw_text']) for r in words)))
        add(scope+'_log_blank',math.log1p(sum(not r['raw_text'].strip() for r in words)))
        normalized=[''.join(c for c in r['raw_text'].casefold() if c.isalnum()) for r in words]
        add(scope+'_duplicates',len(normalized)-len(set(normalized)))
    for cutoff in settings['sequence_cuts']:
        add('raw_sequence_above_'+str(cutoff),sum(r['sequence_confidence_native']>=cutoff for r in candidates))
    # Exact center-pad map frame; no validity/owner/annotation input is accepted.
    from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
    canvas=np.asarray(letterbox_canvas_mask(tuple(image_wh),native_size=side),bool)
    covered=np.zeros((side,side),bool)
    for r in rows:
        if 'bbox_native' in r:
            a,b,c,d=map(int,r['bbox_native'])
        else:
            scale=side/max(w,h); ox=(max(w,h)-w)//2; oy=(max(w,h)-h)//2
            x,y,xx,yy=r['bbox'];a,b=math.floor((x+ox)*scale),math.floor((y+oy)*scale)
            c,d=math.ceil((xx+ox)*scale),math.ceil((yy+oy)*scale)
        if not 0<=a<c<=side or not 0<=b<d<=side:raise ValueError('Native crop outside map')
        covered[b:d,a:c]=True
    area=max(int(canvas.sum()),1)
    add('rectangle_fraction',np.sum(covered&canvas)/area,'layout')
    for head,values in zip(('text','kernel'),p):
        total=float(values[canvas].sum())
        add(head+'_mass',total/area,'layout')
        add(head+'_uncovered_soft',float(values[canvas&~covered].sum())/max(total,1e-12),'layout')
        add(head+'_uncertain',np.sum((values>.2)&(values<.8)&canvas)/area,'layout')
        for cutoff in settings['map_cuts']:
            fg=(values>=cutoff)&canvas; missing=fg&~covered
            count,_,components,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),connectivity=8)
            mc,_,ms,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),connectivity=8)
            prefix=head+'_'+str(cutoff)
            for name,value in [('foreground',fg.sum()/area),('missing',missing.sum()/area),
                               ('components',math.log1p(count-1)),('missing_components',math.log1p(mc-1)),
                               ('largest_missing',float(ms[1:,cv2.CC_STAT_AREA].max())/area if mc>1 else 0.)]:
                add(prefix+'_'+name,value,'layout')
        grid=settings['grid_side']
        for y in range(grid):
            for x in range(grid):
                ys=slice(y*side//grid,(y+1)*side//grid);xs=slice(x*side//grid,(x+1)*side//grid)
                valid=canvas[ys,xs]; local=values[ys,xs]
                add(f'{head}_grid_{y}_{x}_mass',float(local[valid].sum())/area,'layout')
                add(f'{head}_grid_{y}_{x}_missing',float(local[valid&~covered[ys,xs]].sum())/area,'layout')
    return np.asarray(list(result.values()),np.float64),list(result),[groups[n] for n in result]


def paired_target(light, heavy, reference_f1):
    """F1-relative utility; actual heavy decisions are training supervision only."""
    return 2*(light['tp']-heavy['tp'])-reference_f1*(light['detected_words']-heavy['detected_words'])


def export_tree(tree):
    t=tree.tree_
    return dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
                feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,:].tolist())


def tree_predict(tree,x):
    values=[]
    for row in np.asarray(x,np.float32):
        node=0
        while tree['left'][node]!=-1:
            node=tree['left'][node] if row[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        values.append(tree['value'][node])
    return np.asarray(values,float)


def fit_small_head(x,y,weights,recipe,seed):
    """Return a portable JSON model and check it against sklearn immediately."""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor
    x=np.asarray(x,float);y=np.asarray(y,float);weights=np.asarray(weights,float)
    if x.ndim!=2 or y.shape!=(len(x),) or weights.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(weights<=0):
        raise ValueError('Invalid small-head fitting tensors')
    mean=x.mean(0);scale=x.std(0);scale[scale==0]=1;z=(x-mean)/scale
    kind=recipe['kind'];out=dict(recipe=recipe,mean=mean.tolist(),scale=scale.tolist())
    if kind=='ridge':
        model=Ridge(alpha=recipe['alpha'],solver='svd').fit(z,y,sample_weight=weights)
        out.update(coef=model.coef_.tolist(),intercept=float(model.intercept_))
    elif kind=='extra_trees':
        model=ExtraTreesRegressor(n_estimators=recipe['trees'],max_depth=recipe['depth'],min_samples_leaf=recipe['minimum_leaf'],
                                 random_state=seed,n_jobs=1,max_features=1.).fit(z,y,sample_weight=weights)
        out['trees']=[export_tree(t) for t in model.estimators_]
    elif kind=='boosted':
        model=GradientBoostingRegressor(n_estimators=recipe['trees'],max_depth=recipe['depth'],min_samples_leaf=recipe['minimum_leaf'],
            learning_rate=recipe['learning_rate'],loss='huber',alpha=.9,random_state=seed).fit(z,y,sample_weight=weights)
        out.update(trees=[export_tree(t[0]) for t in model.estimators_],initial=float(model.init_.constant_.ravel()[0]),learning_rate=recipe['learning_rate'])
    else:raise ValueError('Unknown explicit small-head recipe')
    if not np.allclose(predict_small_head(out,x),model.predict(z),rtol=1e-10,atol=1e-10):
        raise ValueError('Portable head differs from its fitted estimator')
    return out


def predict_small_head(model,x):
    z=(np.asarray(x,float)-model['mean'])/model['scale'];kind=model['recipe']['kind']
    if not np.isfinite(z).all():raise ValueError('Nonfinite deployment features')
    if kind=='ridge':return z@np.asarray(model['coef'])+model['intercept']
    predictions=np.asarray([tree_predict(t,z)[:,0] for t in model['trees']])
    if kind=='extra_trees':return predictions.mean(0)
    if kind=='boosted':return model['initial']+model['learning_rate']*predictions.sum(0)
    raise ValueError('Unknown portable head')


def model_cost(model):
    if model['recipe']['kind']=='ridge':
        return dict(coefficients=len(model['coef'])+1,trees=0,nodes=0,maximum_comparisons=0)
    trees=model['trees']
    return dict(coefficients=0,trees=len(trees),nodes=sum(len(t['left']) for t in trees),
                maximum_comparisons=len(trees)*model['recipe']['depth'])


def rank_accept(scores,ids,fraction):
    if not 0<=fraction<=1 or set(scores)!=set(ids):raise ValueError('Invalid budget population')
    return sorted((s for s in ids if scores[s] is not None),key=lambda s:(-scores[s],s))[:round(len(ids)*fraction)]


def predict_observation(deployment,rows,candidates,image_wh,probabilities,shared_context=None):
    """Single-image inference; arguments deliberately exclude expert outcomes."""
    values,names,_=observation_features(rows,candidates,image_wh,probabilities,deployment['features'])
    if names!=deployment['schema']:raise ValueError('Deployment observation schema changed')
    x=values[deployment['feature_indices']]
    if deployment['shared_projection'] is not None:
        projection=deployment['shared_projection']
        if shared_context is None:raise ValueError('Selected gate requires shared U-Net pooling')
        projected=(np.asarray(shared_context)-projection['mean'])@np.asarray(projection['components']).T
        x=np.r_[x,projected]
    normalized=float(predict_small_head(deployment['head'],x[None])[0])
    score=normalized*math.sqrt(len(rows)+1)
    eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
    return dict(score=score if eligible else None,eligible=eligible,
                use_unet={str(q):eligible and p['threshold'] is not None and score>=p['threshold'] for q,p in deployment['policies'].items()})
