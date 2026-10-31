"""Portable NumPy deployment of the qualified corrected-U-Net budget gate.

The original trained Ridge and validation cuts are preserved; no new fitting,
recognition, fallback inference or caption encoder is performed here.
"""
import numpy as np
from recognition.lightweight_unet_router import ocr_summary,mask_summary


def predict_portable_essential(deployment,sample_id,rows,image_wh,probabilities):
    if not sample_id or not isinstance(sample_id,str):raise ValueError('Stable sample ID required')
    x=np.r_[ocr_summary(rows,image_wh),mask_summary(probabilities,image_wh,rows,deployment['map_features'])]
    if x.shape!=(30,) or not np.isfinite(x).all():raise ValueError('Expected30 finite OCR/map descriptors')
    score=float(((x-deployment['mean'])/deployment['scale'])@np.asarray(deployment['coef'])+deployment['intercept'])*(len(rows)+1)
    eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
    score=score if eligible else None
    decisions={str(p['target_calibration_acceptance']):bool(score is not None and
        (score>p['threshold'] or score==p['threshold'] and sample_id<=p['tie_sample_id'])) for p in deployment['policies']}
    return dict(score=score,use_unet=decisions)
