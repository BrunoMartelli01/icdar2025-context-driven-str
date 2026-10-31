"""Approximate single-logit confidence seeds; preserve 8-connected word gaps."""

import numpy as np

from localize.word_multitask import word_instances


def separated_mask(probability,canvas,*,support_threshold,seed_threshold):
    p=np.asarray(probability)
    if not 0<support_threshold<=seed_threshold<1:
        raise ValueError('Explicit ordered support/seed probabilities required')
    config=dict(method='watershed',text_threshold=support_threshold,
        kernel_threshold=seed_threshold,center_threshold=.5,watershed_weights=[1.,0.])
    # The second map is explicitly the SAME foreground probability, not a
    # separately trained kernel head. No GT or ignore-validity map enters.
    labels,audit,support=word_instances(np.stack([p,p]),canvas,config)
    protected=(p>=seed_threshold)&canvas
    erase=np.zeros(p.shape,dtype=bool)
    h,w=p.shape
    for dy,dx in [(0,1),(1,-1),(1,0),(1,1)]:
        a=(slice(0,h-dy),slice(max(0,-dx),min(w,w-dx)))
        b=(slice(dy,h),slice(max(0,dx),min(w,w+dx)))
        left,right=labels[a],labels[b]
        contact=(left>0)&(right>0)&(left!=right)
        if np.any(contact&protected[a]&protected[b]):
            raise RuntimeError('Distinct protected seeds touch under8-connectivity')
        erase[a]|=contact&~protected[a];erase[b]|=contact&~protected[b]
    prediction=support&~erase
    if np.any(protected&~prediction) or np.any(prediction&~canvas):
        raise RuntimeError('Confidence seeds or RGB canvas changed')
    return prediction,dict(audit,boundary_pixels_removed=int(erase.sum()),
        protected_seed_pixels=int(protected.sum()))
