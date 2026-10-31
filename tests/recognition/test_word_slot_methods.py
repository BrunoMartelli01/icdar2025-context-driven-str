"""Denominators, split identity, deployable gate lifecycle and coordinate tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

from segmentation.word_slot_data import original_to_canvas,canvas_to_original,write,read,sha
from recognition.word_slot_evaluation import measure,aggregate,merge_words,nms
from recognition.word_slot_routing import FrozenWordSlotRouter
from scripts.run_word_slot_methods import prepare_deep,normalized_edit,main as training_main,teacher_preprocessing,validate_teacher_entry,setup as slot_setup
from scripts.analyze_word_slot_routers import analyze
from scripts.report_word_slot_methods import hungarian,image_oracle,validate_mae_qualification
from scripts.qualify_word_slot_mae_teacher import qualify
from pipeline.word_slot_specialist import WordSlotSpecialist


def prediction(text='WORD',bbox=None):
    return {'text':text,'bbox':bbox or [1,1,9,9],'confidence':.9,'presence':.8}


class WordSlotProtocolTests(unittest.TestCase):
    def test_odd_letterbox_roundtrip_and_clipping(self):
        boxes=torch.tensor([[0.,0.,93.,52.],[2.,3.,80.,51.]])
        actual=canvas_to_original(original_to_canvas(boxes,(93,52)),(93,52))
        torch.testing.assert_close(actual,boxes,rtol=0,atol=1e-5)

    def test_text_and_geometry_and_one_to_one_denominators(self):
        truth=[{'text':'WORD','bbox':[1,1,9,9]}]
        m=measure([prediction('word'),prediction('word'),prediction('word',[30,30,40,40])],truth,.5)
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,2,0))
        empty=measure([],truth,.5)
        self.assertEqual((empty['tp'],empty['fn']),(0,1))
        self.assertEqual(aggregate([empty])['recall'],0)
        self.assertEqual(measure([prediction('W&ORD')],truth,.5)['tp'],0)

    def test_nms_uses_float_iou_for_normalized_reference_boxes(self):
        boxes=[[.10,.10,.30,.30],[.11,.11,.31,.31],[.70,.70,.90,.90]]
        self.assertEqual(nms(boxes,[.9,.8,.7],.5,20),[0,2])

    def test_independent_hungarian_matches_integer_pixel_metric_at_fractional_edges(self):
        predicted=[prediction('A',[.9,0.,2.9,2.]),prediction('A',[.8,0.,2.8,2.])]
        truth=[{'text':'a','bbox':[0,0,2,2]}]
        count,_=hungarian(predicted,truth,.5)
        self.assertEqual(count,1);self.assertEqual(measure(predicted,truth,.5)['tp'],count)

    def test_teacher_targets_cannot_change_after_mae_branch_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'branches/mae_character').mkdir(parents=True)
            write(root/'protocol.json',{'config':{}});write(root/'config.json',{})
            with patch('scripts.qualify_word_slot_mae_teacher.setup',side_effect=AssertionError('must not load GPU teacher')):
                with self.assertRaisesRegex(ValueError,'already initialized'):qualify(root,{},root/'config.json')

    def test_live_specialist_rejects_strict_before_reading_assets(self):
        with self.assertRaisesRegex(ValueError,'APPROXIMATE'):
            WordSlotSpecialist('/definitely/missing','slot_control',mode='strict')

    def test_live_release_moves_both_models_to_cpu_before_dropping_references(self):
        events=[]
        class Model:
            def __init__(self,name):self.name=name
            def cpu(self):events.append(self.name+'_cpu')
        class Reader:
            def __init__(self):self.model=Model('parseq')
            def release(self):events.append('reader_release');self.model=None
        instance=WordSlotSpecialist.__new__(WordSlotSpecialist);instance.model=Model('unet');instance.recognizer=Reader()
        with patch('torch.cuda.empty_cache'),patch('torch.cuda.memory_allocated',return_value=0):result=instance.release()
        self.assertEqual(events,['parseq_cpu','unet_cpu','reader_release']);self.assertTrue(result['student_weights_moved_to_cpu_before_release'])

    def test_pretrained_mae_recipe_rejects_legacy_finetuning_cache(self):
        expected=teacher_preprocessing('mae');self.assertEqual(expected['normalization_mean'],[.5]*3)
        self.assertEqual(expected['pil_interpolation'],'bicubic')
        entry={'teacher_sha256':'weights','image_sha256':'image'}
        with self.assertRaisesRegex(ValueError,'Unqualified MAE'):validate_teacher_entry(entry,'mae','weights','image')
        entry['input_preprocessing']={**expected,'additional_qualification_evidence':True}
        validate_teacher_entry(entry,'mae','weights','image')

    def test_report_accepts_direct_fresh_mae_qualification_and_rejects_stale_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'Missing MAE'):validate_mae_qualification(root)
            audit={'mae':{'status':'PASS','input_preprocessing':teacher_preprocessing('mae')}}
            write(root/'teacher_cache_audit.json',audit);validate_mae_qualification(root)
            audit['mae']['input_preprocessing']['normalization_mean']=[.485,.456,.406]
            write(root/'teacher_cache_audit.json',audit)
            with self.assertRaisesRegex(ValueError,'Unqualified MAE'):validate_mae_qualification(root)

    def test_report_historical_mae_failure_cannot_be_hidden_by_fresh_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'teacher_cache_audit.json',{'mae':{'status':'PASS','input_preprocessing':teacher_preprocessing('mae')}})
            receipt={'status':'PASS','config':teacher_preprocessing('mae')}
            write(root/'mae_preprocessing_v2_complete.json',receipt);validate_mae_qualification(root)
            receipt['status']='FAILED';write(root/'mae_preprocessing_v2_complete.json',receipt)
            with self.assertRaisesRegex(ValueError,'Unqualified MAE'):validate_mae_qualification(root)

    def test_runtime_cannot_claim_unexecuted_precision_or_grid_placement(self):
        runtime={'device':'cuda:0','precision':'float32','feature_cache_dtype':'float32','rgb_grid_device':'cpu',
                 'deterministic_algorithms':True,'allow_tf32':False}
        for key,value in [('precision','float16'),('feature_cache_dtype','float16'),('rgb_grid_device','cuda'),('allow_tf32',True)]:
            with self.assertRaisesRegex(ValueError,'Unqualified word-slot runtime'):
                slot_setup({'runtime':{**runtime,key:value}})

    def test_f1_oracle_matches_bruteforce_and_handles_fixed_fallback_cost(self):
        import itertools
        def row(tp,n,g):return {'tp':tp,'fp':n-tp,'fn':g-tp,'predictions':n,'gt_words':g}
        student=[row(2,2,2),row(0,0,1),row(0,2,1)]
        deep=[row(1,1,2),row(1,10,1),row(1,1,1)]
        for count in (None,1):
            scores=[]
            for choices in itertools.product((False,True),repeat=3):
                if count is not None and sum(choices)!=count:continue
                scores.append(aggregate([s if c else d for s,d,c in zip(student,deep,choices)])['f1'])
            self.assertAlmostEqual(image_oracle(student,deep,count)['f1'],max(scores))

    def test_word_router_replaces_one_conflict_and_keeps_other_deep_words(self):
        u=[prediction('B&B'),prediction('NEW',[20,20,30,30])]
        d=[prediction('bb'),prediction('GOOD',[40,40,50,50])]
        result,decisions=merge_words(u,d,[.99,.95],[.3,.9],append_threshold=.9,replace_threshold=.9,margin=.2,conflict_iou=.3)
        self.assertEqual({p['text'] for p in result},{'B&B','NEW','GOOD'})
        self.assertEqual({d['action'] for d in decisions},{'replace','append'})

    def router(self,threshold):
        audit={'image':{'full_student_thresholds':[0,0], 'scaler_mean':[0]*12,'scaler_scale':[1]*12,
                        'ridge_coefficients':[0]*12,'ridge_intercept':0,'chosen':{'threshold':threshold}},
               'word':{'parameters':{'kind':'empirical_constant','probability':.9},
                       'chosen':{'append_threshold':.95,'replace_threshold':.95,'margin':.1}}}
        return FrozenWordSlotRouter(audit,{'conflict_iou':.3})

    def test_real_routing_boundary_calls_release_before_deep_and_propagates_errors(self):
        events=[]
        def release():events.append('release')
        def deep():events.append('deep');return [prediction('DEEP')]
        result=self.router(-1).route([prediction()],(100,100),kind='image',release_unet=release,run_deepsolo=deep)
        self.assertEqual(events,[]);self.assertEqual(result['deepsolo_calls'],0)
        result=self.router(1).route([prediction()],(100,100),kind='image',release_unet=release,run_deepsolo=deep)
        self.assertEqual(events,['release','deep']);self.assertEqual(result['predictions'][0]['text'],'DEEP')
        def failed():raise RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError,'inference failed'):
            self.router(1).route([prediction()],(100,100),kind='word',release_unet=release,run_deepsolo=failed)

    def test_common_edit_risk_retains_punctuation(self):
        self.assertEqual(normalized_edit('ABC','abc'),0)
        self.assertAlmostEqual(normalized_edit('bb','B&B'),1/3)

    def test_gpu_out_of_memory_is_reported_and_propagated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=root/'configs/approximate/c.json';output=root/'results'
            write(config,{'reproduction_mode':'approximate','experiment_kind':'word_slot_methods_v1','input_sha256':{}})
            argv=['runner','--mode','approximate','--config',str(config),'--output',str(output),'--stage','warmup']
            with patch('scripts.run_word_slot_methods.REPO',root),patch.object(sys,'argv',argv),patch('scripts.run_word_slot_methods.setup',side_effect=torch.cuda.OutOfMemoryError('injected OOM qualification')):
                with self.assertRaises(torch.cuda.OutOfMemoryError):training_main()
            files=list(output.glob('failure_warmup_*.json'));self.assertEqual(len(files),1)
            result=read(files[0]);self.assertTrue(result['oom']);self.assertEqual(result['status'],'FAILED')
            self.assertFalse((output/'warmup/complete.json').exists())

    def test_deepsolo_join_uses_split_and_rgb_hash_not_only_sample_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);index={};source={'splits':{}}
            for phase,source_phase,text,char in [('evaluation','train','TRAIN','a'),('test','test','TEST','b')]:
                raw=root/(phase+'_raw.json');hash_=char*64
                response={'detections':[{'source_index':0,'score':.9,'text':text,
                          'geometry':{'boundary_points':[[1,1],[8,1],[8,8],[1,8]]}}]}
                write(raw,{'contract':{'image_sha256':hash_},'response':response})
                source['splits'][source_phase]={'per_image':[{'sample_id':'total_text:1','raw_response_path':str(raw),'raw_response_sha256':sha(raw)}]}
                index[phase]=[{'sample_id':'total_text:1','image_sha256':hash_,'image_wh':[100,100], 'orientation':1,
                              'ground_truth':[{'text':text,'bbox':[1,1,9,9]}]}]
            write(root/'source.json',source);write(root/'plan.json',{'config':{'score_threshold':.3}});write(root/'protocol.json',{'index':index})
            config={'paths':{'deepsolo_results':str(root/'source.json'),'deepsolo_plan':str(root/'plan.json')},'evaluation':{'iou_threshold':.5}}
            prepare_deep(root,config)
            for phase in index:
                self.assertEqual(read(root/'deepsolo'/phase/'total_text__1.json')['metric']['tp'],1)
            index['test'][0]['image_sha256']='c'*64;write(root/'protocol.json',{'index':index})
            with self.assertRaisesRegex(ValueError,'split/RGB'):prepare_deep(root,config)

    def test_router_fits_before_loading_same_named_test_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);index={}
            for phase,text in [('calibration','TRAIN'),('test','TEST')]:
                record={'sample_id':'total_text:1','image_wh':[100,100],'ground_truth':[{'text':text,'bbox':[1,1,9,9]}]}
                index[phase]=[record]
                write(root/'predictions'/'arm'/phase/'total_text__1.json',{'predictions':[prediction(text)]})
                write(root/'deepsolo'/phase/'total_text__1.json',{'predictions':[prediction(text)]})
            config={'evaluation':{'bootstrap_draws':2},'seed':3};write(root/'protocol.json',{'index':index,'config':config})
            def image_fit(records,student,deep,*args):
                self.assertEqual(student['total_text:1'][0]['text'],'TRAIN')
                class Model:
                    def predict(self,x):return np.zeros(len(x))
                return Model(),{'chosen':{'threshold':1},'full_student_thresholds':[0,0]}
            def word_fit(records,student,deep,*args):
                self.assertEqual(student['total_text:1'][0]['text'],'TRAIN')
                return .9,{'chosen':{'append_threshold':1.01,'replace_threshold':1.01,'margin':0}}
            c={'iou_threshold':.5,'conflict_iou':.3,'fixed_image_unet_fraction':.5}
            with patch('scripts.analyze_word_slot_routers.image_router',side_effect=image_fit),patch('scripts.analyze_word_slot_routers.word_router',side_effect=word_fit):
                analyze(root,c,['arm'],['test'])
            self.assertEqual(read(root/'router_results.json')['test']['arm']['deepsolo']['tp'],1)


if __name__=='__main__':unittest.main()
