"""Deploy calibrated word-slot selectors without ground truth or sklearn objects."""
from __future__ import annotations

import numpy as np
from recognition.word_slot_evaluation import image_features,word_features,select,merge_words


class FrozenWordSlotRouter:
    def __init__(self,fit_audit,router_config):
        self.audit=fit_audit;self.config=router_config
        self.normalization=router_config.get('text_normalization','casefold')

    def image_score(self,predictions,image_wh):
        audit=self.audit['image'];presence,confidence=audit['full_student_thresholds']
        selected=select(predictions,presence,confidence)
        features=np.asarray(image_features(selected,image_wh))
        normalized=(features-np.asarray(audit['scaler_mean']))/np.asarray(audit['scaler_scale'])
        score=float(normalized@np.asarray(audit['ridge_coefficients'])+audit['ridge_intercept'])
        return selected,score

    def quality(self,student,deep,image_wh):
        parameters=self.audit['word']['parameters']
        features=np.asarray([word_features(p,deep,image_wh,0,self.normalization) for p in student]+[word_features(p,student,image_wh,1,self.normalization) for p in deep])
        if not len(features):return np.empty(0)
        if parameters['kind']=='empirical_constant':return np.full(len(features),parameters['probability'])
        normalized=(features-np.asarray(parameters['scaler_mean']))/np.asarray(parameters['scaler_scale'])
        logit=normalized@np.asarray(parameters['logistic_coefficients'])[0]+parameters['logistic_intercept'][0]
        return np.exp(-np.logaddexp(0,-logit))

    def route(self,student,image_wh,*,kind,release_unet,run_deepsolo):
        """Callbacks make real model lifecycle and errors the caller's explicit boundary.

        release_unet ALWAYS precedes actual fallback. A failure propagates;
        no cached, fabricated, or alternate recognizer result is substituted.
        """
        if kind=='image':
            selected,score=self.image_score(student,image_wh)
            if score>=self.audit['image']['chosen']['threshold']:
                return {'predictions':selected,'expert':'unet','deepsolo_calls':0,'score':score}
            release_unet();deep=run_deepsolo()
            return {'predictions':deep,'expert':'deepsolo','deepsolo_calls':1,'score':score}
        if kind=='word':
            release_unet();deep=run_deepsolo();probability=self.quality(student,deep,image_wh)
            chosen=self.audit['word']['chosen']
            merged,decisions=merge_words(student,deep,probability[:len(student)],probability[len(student):],
                     append_threshold=chosen['append_threshold'],replace_threshold=chosen['replace_threshold'],
                     margin=chosen['margin'],conflict_iou=self.config['conflict_iou'],normalization=self.normalization)
            return {'predictions':merged,'expert':'word_merge','deepsolo_calls':1,'decisions':decisions}
        raise ValueError('Unknown router kind')
