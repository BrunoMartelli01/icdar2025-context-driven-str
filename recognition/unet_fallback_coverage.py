"""GT-free missing-text and native OCR confidence evidence for image fallback."""
from __future__ import annotations
import math
import cv2
import numpy as np


def coverage_features(probabilities,canvas,pool,predictions,ocr,settings):
    if probabilities.ndim!=3 or probabilities.shape[0]!=2 or probabilities.shape[1:]!=canvas.shape:
        raise ValueError('Unaligned text/kernel maps')
    if not np.isfinite(probabilities).all() or not canvas.any():raise ValueError('Invalid predicted map/canvas')
    by={p['id']:p for p in pool};union=np.zeros(canvas.shape,bool);values={}
    def add(name,value):
        if not math.isfinite(float(value)):raise ValueError('Nonfinite coverage feature')
        values[name]=float(value)
    def stats(name,array):
        array=np.asarray(array,float);add(name+'_missing',not len(array));add(name+'_mean',array.mean() if len(array) else 0.)
        for q in settings['quantiles']:add(name+'_q'+str(q),np.quantile(array,q) if len(array) else 0.)
    source=[]
    for p in predictions:
        sid=p.get('source_proposal_id',p['id'])
        if sid not in by:raise ValueError('Unknown native source')
        q=by[sid];a,b,d,e=map(int,q['bbox_native']);h,w=canvas.shape
        if not 0<=a<d<=w or not 0<=b<e<=h:raise ValueError('Native proposal outside its qualified map')
        union[b:e,a:d]=True;source.append(q)
    add('rectangle_canvas_fraction',np.sum(union&canvas)/canvas.sum())
    for head,index in [('text',0),('kernel',1)]:
        probability=probabilities[index]
        total=float(probability[canvas].sum())
        add(head+'_mean_probability',total/canvas.sum())
        add(head+'_uncovered_probability_fraction',probability[canvas&~union].sum()/max(total,1e-12))
        for threshold in settings['map_thresholds']:
            mask=(probability>=threshold)&canvas;number,labels,components,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
            name=head+'_'+str(threshold);areas=components[1:,cv2.CC_STAT_AREA]
            add(name+'_foreground_fraction',mask.sum()/canvas.sum())
            add(name+'_uncovered_fraction',np.sum(mask&~union)/max(1,int(mask.sum())))
            stats(name+'_component_log_area',np.log1p(areas))
            uncovered=np.bincount(labels[mask&~union],minlength=number)[1:]
            missing_fraction=uncovered/np.maximum(areas,1)
            for area in settings['component_minimum_areas']:
                add(name+'_components_ge'+str(area),np.sum(areas>=area))
                add(name+'_uncovered_components_ge'+str(area),np.sum((areas>=area)&(missing_fraction>=settings['uncovered_component_fraction'])))
    for name,proposals in [('pool',pool),('selected',source)]:
        confidences=[];minimum=[];entropies=[]
        for p in proposals:
            native=ocr[p['crop_array_sha256']]
            if native['raw_text']!=p['text']:raise ValueError('Native OCR/RGB binding changed')
            token=np.asarray(native['token_probabilities'],float)
            if not np.isfinite(token).all() or np.any(token<0) or np.any(token>1):raise ValueError('Invalid native token confidence')
            confidences.append(float(np.exp(np.log(np.maximum(token,1e-12)).mean())) if len(token) else 0.)
            minimum.append(float(token.min()) if len(token) else 0.)
            entropies.append(float(-np.log(np.maximum(token,1e-12)).sum()))
        stats(name+'_native_confidence',confidences);stats(name+'_native_minimum',minimum)
        stats(name+'_native_surprisal',entropies)
    return np.asarray(list(values.values()),float),list(values)
