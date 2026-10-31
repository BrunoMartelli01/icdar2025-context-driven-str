"""Cheap single-image ICDAR gate; inputs exclude GT and fallback predictions."""
from __future__ import annotations
import numpy as np
from recognition.icdar_budget_router import observation,accept
from recognition.lightweight_unet_router import predict_head


def predict(bundle,feature_settings,rows,image_wh,*,tie_key,policies,probabilities=None,frame=None):
    if 'maps' in bundle['groups'] and probabilities is None:
        raise ValueError('Selected U-Net gate requires its genuine predicted maps')
    groups=observation(rows,image_wh,probabilities if 'maps' in bundle['groups'] else None,frame,settings=feature_settings)
    x=np.concatenate([groups[g] for g in bundle['groups']])[None]
    value=float(predict_head(bundle,x)[0])*(len(rows)+1)
    if not rows or any(not row['raw_text'].strip() for row in rows):value=None
    decisions={str(p['target_acceptance']):bool(accept({'image':value},p,{'image':tie_key})) for p in policies}
    return dict(score=value,accept_light=decisions,requires_deepsolo={q:not flag for q,flag in decisions.items()})
