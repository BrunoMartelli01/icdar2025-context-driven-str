import numpy as np
import pytest
import torch

from recognition.encoder_family_router import EncoderFamilyRouter
from recognition.encoder_family_inference import predict_encoder_gate
from tests.recognition.test_encoder_family_router import SETTINGS


@pytest.mark.parametrize('arm',['none','clip','blip','both'])
def test_single_image_boundary_only_requires_selected_encoder(arm):
    torch.manual_seed(12);model=EncoderFamilyRouter(arm,SETTINGS,12).eval()
    checkpoint=dict(scalar_mean=[0.]*30,scalar_scale=[1.]*30,
        inference_settings=dict(target_scale=1.,target_mean=[0.,0.],reference_f1=.8))
    rows=[dict(raw_text='abc!',bbox=[1,2,8,9],confidence=.9,detector_score=.8,token_probabilities=[.9,.9,.9])]
    arrays={}
    if arm in ['clip','both']:
        arrays.update(word_text=np.zeros((1,512),np.float32),word_image=np.zeros((1,512),np.float32),
            patches=np.zeros((196,512),np.float32),patch_geometry=np.zeros((196,4),np.float32),patch_valid=np.ones(196,bool))
    if arm in ['blip','both']:arrays['queries']=np.zeros((32,768),np.float32)
    policies=[dict(target_calibration_acceptance=.5,threshold=-100.,tie_sample_id='z')]
    out=predict_encoder_gate(model,checkpoint,policies,'x',rows,[10,10],np.zeros((2,16,16),np.float32),
        dict(probability_threshold=.5,minimum_component_area=4),context_arrays=arrays or None)
    assert np.isfinite(out['predicted_utility']) and out['use_unet']['0.5'] and not out['ground_truth_used']
    if arm!='none':
        with pytest.raises(ValueError,match='evidence required'):
            predict_encoder_gate(model,checkpoint,policies,'x',rows,[10,10],np.zeros((2,16,16),np.float32),
                dict(probability_threshold=.5,minimum_component_area=4))
