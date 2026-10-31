"""Prediction-only single-image boundary for a frozen encoder-family checkpoint."""
import numpy as np
import torch

from recognition.encoder_family_router import collate_observations
from recognition.lightweight_unet_router import ocr_summary, mask_summary
from recognition.token_context_router import word_geometry


@torch.inference_mode()
def predict_encoder_gate(model, checkpoint, policies, sample_id, rows, image_wh, probabilities,
                         map_settings, *, context_arrays=None, device='cpu'):
    if not sample_id: raise ValueError('Stable sample ID required')
    scalar=np.concatenate([ocr_summary(rows,image_wh),mask_summary(probabilities,image_wh,rows,map_settings)]).astype(np.float32)
    scalar=((scalar-np.asarray(checkpoint['scalar_mean']))/np.asarray(checkpoint['scalar_scale'])).astype(np.float32)
    observation=dict(texts=[r['raw_text'] for r in rows],geometry=word_geometry(rows,image_wh),scalars=scalar)
    if model.arm in {'clip','both'}:
        if context_arrays is None: raise ValueError('CLIP evidence required by selected checkpoint')
        for key in ['word_text','word_image']:observation[key]=context_arrays[key]
        for key in ['patches','patch_geometry','patch_valid']:observation[key]=context_arrays[key][:196]
    if model.arm in {'blip','both'}:
        if context_arrays is None: raise ValueError('BLIP QFormer evidence required by selected checkpoint')
        observation['queries']=context_arrays['queries']
    model.eval();batch=collate_observations([observation],model.arm,device)
    settings=checkpoint['inference_settings'];counts=model(batch)[0].cpu().numpy()*settings['target_scale']+np.asarray(settings['target_mean'])
    eligible=bool(rows) and all(r['raw_text'].strip() for r in rows)
    score=float(2*counts[0]-settings['reference_f1']*counts[1]) if eligible else None
    return dict(predicted_utility=score,ground_truth_used=False,use_unet={str(p['target_calibration_acceptance']):bool(
        score is not None and (score>p['threshold'] or score==p['threshold'] and sample_id<=p['tie_sample_id'])) for p in policies})
