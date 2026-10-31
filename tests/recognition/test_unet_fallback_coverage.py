"""Missing-text coverage must depend on native predicted geometry, never labels."""
import unittest
import numpy as np
from recognition.unet_fallback_coverage import coverage_features

class CoverageTests(unittest.TestCase):
    def settings(self):return dict(quantiles=[0.,1.],map_thresholds=[.5],component_minimum_areas=[1],uncovered_component_fraction=.5)
    def inputs(self):
        prob=np.zeros((2,6,6),np.float32);prob[:,0:2,0:2]=.9;prob[:,4:6,4:6]=.9
        pool=[dict(id='a',text='A',bbox_native=[0,0,2,2],crop_array_sha256='hash')]
        ocr={'hash':dict(raw_text='A',token_probabilities=[.9,.8])}
        return prob,np.ones((6,6),bool),pool,ocr
    def test_uncovered_predicted_component_is_counted(self):
        prob,canvas,pool,ocr=self.inputs();values,names=coverage_features(prob,canvas,pool,pool,ocr,self.settings());v=dict(zip(names,values))
        self.assertEqual(v['text_0.5_components_ge1'],2)
        self.assertEqual(v['text_0.5_uncovered_components_ge1'],1)
        self.assertAlmostEqual(v['text_0.5_uncovered_fraction'],.5)
    def test_empty_outputs_leave_all_components_uncovered(self):
        prob,canvas,pool,ocr=self.inputs();values,names=coverage_features(prob,canvas,pool,[],ocr,self.settings());v=dict(zip(names,values))
        self.assertEqual(v['text_0.5_uncovered_components_ge1'],2);self.assertTrue(np.isfinite(values).all())
    def test_edited_reading_uses_bound_native_ocr_without_reinterpreting_confidence(self):
        prob,canvas,pool,ocr=self.inputs();pred=[dict(id='edit:a',source_proposal_id='a',text='B')]
        values,names=coverage_features(prob,canvas,pool,pred,ocr,self.settings())
        self.assertAlmostEqual(dict(zip(names,values))['selected_native_confidence_mean'],np.sqrt(.72))
    def test_misaligned_native_reading_and_outside_box_fail(self):
        prob,canvas,pool,ocr=self.inputs();ocr['hash']['raw_text']='different'
        with self.assertRaises(ValueError):coverage_features(prob,canvas,pool,pool,ocr,self.settings())
        prob,canvas,pool,ocr=self.inputs();pool[0]['bbox_native']=[0,0,10,2]
        with self.assertRaises(ValueError):coverage_features(prob,canvas,pool,pool,ocr,self.settings())

if __name__=='__main__':unittest.main()
