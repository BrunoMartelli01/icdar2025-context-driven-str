"""Supervised TRAIN-only retrieval of frozen branch errors; no neural updates."""
from __future__ import annotations

import numpy as np


def normalize(x):
    x=np.asarray(x,float);norm=np.linalg.norm(x,axis=-1,keepdims=True)
    return x/np.maximum(norm,1e-12)


def fit_memory(features,branches,ids,kind,k,names):
    if set(branches)!=set(ids): raise ValueError('Memory labels must be exactly the fitting population')
    visual=normalize([features[s]['visual_embedding'] for s in ids])
    x=np.array([[features[s]['values'][n] for n in names] for s in ids]);mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale==0]=1.
    vectors=visual if kind=='visual' else normalize(np.concatenate([visual,normalize((x-mean)/scale)],axis=1))
    target=[[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
        branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in ids]
    return {'kind':'memory','variant':kind,'k':k,'training_ids':ids,'training_rgb_sha256':[features[s]['image_sha256'] for s in ids],
        'vectors':vectors.tolist(),'targets':target,'mean':mean.tolist(),'scale':scale.tolist(),'feature_names':names}


def predict_memory(model,features,ids):
    if model['variant']=='visual': query=normalize([features[s]['visual_embedding'] for s in ids])
    else:
        x=np.array([[features[s]['values'][n] for n in model['feature_names']] for s in ids])
        query=normalize(np.concatenate([normalize([features[s]['visual_embedding'] for s in ids]),
            normalize((x-np.array(model['mean']))/np.array(model['scale']))],axis=1))
    cosine=query@np.array(model['vectors']).T;targets=np.asarray(model['targets'],float);out=[]
    for s,similarities in zip(ids,cosine):
        allowed=[i for i,t in enumerate(model['training_ids']) if t!=s and model['training_rgb_sha256'][i]!=features[s]['image_sha256']]
        neighbors=sorted(allowed,key=lambda i:(-similarities[i],model['training_ids'][i]))[:model['k']]
        if not neighbors: raise ValueError('No nonduplicate TRAIN reference for context memory')
        # Fixed inverse cosine-distance weighting; no TEST-fitted temperature.
        weights=1/np.maximum(1-similarities[neighbors],1e-6);weights=weights/weights.sum()
        out.append((weights[:,None]*targets[neighbors]).sum(axis=0))
    return np.asarray(out)
