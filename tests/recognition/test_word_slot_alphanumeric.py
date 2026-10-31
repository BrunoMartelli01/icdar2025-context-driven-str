"""Symbol-free matching, empty denominators and normalized router consistency."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from recognition.word_slot_text import normalize_word_text,ALPHANUMERIC
from recognition.word_slot_evaluation import measure,word_features,merge_words
from scripts.report_word_slot_methods import hungarian,router_result_label
from scripts.analyze_word_slot_routers import quality_training,analyze
from recognition.word_slot_routing import FrozenWordSlotRouter
from segmentation.word_slot_data import write
from scripts.reassess_word_slot_alphanumeric import compare_routed_predictions


def prediction(text,bbox=None):
    return {'text':text,'bbox':bbox or [1,1,9,9],'confidence':.9,'presence':.8}


class AlphanumericComparisonTests(unittest.TestCase):
    def test_router_reporting_rejects_missing_or_impossible_image_counts(self):
        with self.assertRaises(KeyError):router_result_label({'f1':.8,'images':300})
        for images,calls in [(300,301),(0,0),(300,-1),(300,True)]:
            with self.assertRaisesRegex(ValueError,'counts'):router_result_label({'f1':.8,'images':images,'logical_deepsolo_calls':calls})

    def test_router_reporting_uses_actual_denominator_for_partial_fallback_and_word_merging(self):
        self.assertIn('277/300; 92.33%',router_result_label({'f1':.8,'images':300,'logical_deepsolo_calls':277}))
        self.assertIn('125/125; 100.00%',router_result_label({'f1':.8,'images':125,'logical_deepsolo_calls':125}))

    def test_symbols_and_space_are_removed_without_accent_or_compatibility_folding(self):
        for raw,expected in [("FISHERMAN’S",'fishermans'),('B&B','bb'),('AB-12 / C','ab12c'),('café','café'),('Ａ1','ａ1'),('Straße','strasse')]:
            self.assertEqual(normalize_word_text(raw,ALPHANUMERIC),expected)
        self.assertNotEqual(normalize_word_text('café',ALPHANUMERIC),normalize_word_text('cafe',ALPHANUMERIC))

    def test_legacy_policy_retains_punctuation_and_unknown_policy_fails(self):
        self.assertEqual(normalize_word_text('B&B'),'b&b')
        self.assertEqual(measure([prediction('bb')],[prediction('B&B')],.5)['tp'],0)
        with self.assertRaisesRegex(ValueError,'Unknown'):measure([],[],.5,'typo')

    def test_normalized_exact_text_still_requires_geometry_and_one_to_one_assignment(self):
        truth=[prediction('B&B'),prediction('CO-OP',[20,20,30,30])]
        values=[prediction('bb'),prediction('B-B'),prediction('coop',[50,50,60,60])]
        result=measure(values,truth,.5,ALPHANUMERIC)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,1))
        count,matched=hungarian(values,truth,.5,normalization=ALPHANUMERIC)
        self.assertEqual(count,result['tp']);self.assertEqual(matched,set(result['matched_gt_indices']))
        self.assertEqual(values[0]['text'],'bb');self.assertEqual(truth[0]['text'],'B&B')

    def test_symbol_only_and_empty_strings_never_match_or_disappear(self):
        truth=[prediction('&'),prediction('')];values=[prediction('!'),prediction('')]
        m=measure(values,truth,.5,ALPHANUMERIC)
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['gt_words'],m['predictions']),(0,2,2,2,2))
        self.assertEqual(hungarian(values,truth,.5,normalization=ALPHANUMERIC)[0],0)

    def test_quality_labels_and_agreement_use_same_policy_but_keep_raw_features(self):
        record={'sample_id':'one','image_wh':[40,40],'ground_truth':[prediction('B&B')]}
        x,y,_=quality_training([record],{'one':[prediction('B&B')]},{'one':[prediction('bb')]},.5,ALPHANUMERIC)
        np.testing.assert_array_equal(y,[1,1]);self.assertEqual(x[0,4],1);self.assertEqual(x[0,10],1)
        self.assertEqual(word_features(prediction('B&B'),[prediction('bb')],[40,40],0)[10],0)

    def test_symbol_only_quality_label_and_agreement_are_not_positive(self):
        record={'sample_id':'one','image_wh':[40,40],'ground_truth':[prediction('&')]}
        x,y,_=quality_training([record],{'one':[prediction('&')]},{'one':[prediction('!')]},.5,ALPHANUMERIC)
        np.testing.assert_array_equal(y,[0,0]);self.assertEqual(x[0,10],0)

    def test_merge_does_not_replace_normalized_equal_words(self):
        options={'append_threshold':.9,'replace_threshold':.9,'margin':.2,'conflict_iou':.3}
        values,decisions=merge_words([prediction('B&B')],[prediction('bb')],[.99],[.1],**options,normalization=ALPHANUMERIC)
        self.assertEqual(values[0]['text'],'bb');self.assertEqual(decisions,[])
        _,legacy=merge_words([prediction('B&B')],[prediction('bb')],[.99],[.1],**options)
        self.assertEqual(legacy[0]['action'],'replace')

    def test_serialized_router_uses_normalization_on_actual_word_route(self):
        fit={'word':{'parameters':{'kind':'empirical_constant','probability':.99},
                     'chosen':{'append_threshold':.9,'replace_threshold':.9,'margin':0}}}
        router=FrozenWordSlotRouter(fit,{'text_normalization':ALPHANUMERIC,'conflict_iou':.3});events=[]
        result=router.route([prediction('B&B')],[40,40],kind='word',release_unet=lambda:events.append('release'),run_deepsolo=lambda:[prediction('bb')])
        self.assertEqual(events,['release']);self.assertEqual(result['predictions'][0]['text'],'bb');self.assertEqual(result['decisions'],[])

    def test_router_protocol_mismatch_fails_before_loading_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'protocol.json',{'config':{'evaluation':{'text_normalization':ALPHANUMERIC}},'index':{'calibration':[]}})
            with self.assertRaisesRegex(ValueError,'normalization disagree'):analyze(root,{'text_normalization':'casefold'},[],[])

    def test_serialized_audit_only_tolerates_float64_probability_roundoff(self):
        a=prediction('B&B');a['router_probability']=.4
        self.assertLess(compare_routed_predictions([a],[{**a,'router_probability':.4000000000000001}]),1e-12)
        with self.assertRaisesRegex(ValueError,'roundoff'):compare_routed_predictions([a],[{**a,'router_probability':.40000001}])

    def test_serialized_audit_never_tolerates_reading_or_expert_changes(self):
        a={**prediction('B&B'),'route_expert':'unet','router_probability':.4}
        with self.assertRaisesRegex(ValueError,'reading, box'):compare_routed_predictions([a],[{**a,'text':'bb'}])
        with self.assertRaisesRegex(ValueError,'reading, box'):compare_routed_predictions([a],[{**a,'route_expert':'deepsolo'}])

    def test_serialized_audit_rejects_nan_and_missing_or_extra_predictions(self):
        a={**prediction('B&B'),'router_probability':.4}
        with self.assertRaisesRegex(ValueError,'roundoff'):compare_routed_predictions([a],[{**a,'router_probability':float('nan')}])
        with self.assertRaisesRegex(ValueError,'count'):compare_routed_predictions([a],[])
