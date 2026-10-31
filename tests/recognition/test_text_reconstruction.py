"""Scientific boundary checks for text-only reconstruction, not model efficacy."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from recognition.text_reconstruction import (TextPrior,apply_operations,generate_operations,
    insertion_candidate,lexical_changes,lexical_metrics,operation_features,split_options,
    structural_oracle,supervised_target,synthetic_pairs,validate_repair,preserves_letters)


POLICY=dict(maximum_word_characters=32,maximum_split_words=3,minimum_split_characters=2,
            maximum_split_candidates=6,minimum_vertical_overlap=.5,maximum_gap_heights=1.5,
            maximum_height_ratio=2.5,maximum_neighbors=2,maximum_merge_blocks=3,maximum_operations_per_image=300)


def block(identity,text,box):
    return dict(id=identity,text=text,bbox=box,confidence=.8,minimum_token_probability=.5)


class TextReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.prior=TextPrior(['restaurant','new','york','hello','world','new'],['res','taurant'])

    def test_missing_characters_preserve_observed_order(self):
        self.assertTrue(insertion_candidate('TUDIO','STUDIO',maximum_insertions=2))
        self.assertTrue(insertion_candidate('OPE','OPERATING',maximum_insertions=8))
        self.assertFalse(insertion_candidate('OPE','OPERATING',maximum_insertions=2))
        self.assertFalse(insertion_candidate('H0TEL','HOTEL'))
        self.assertFalse(insertion_candidate('WORLD','WORD'))
        self.assertFalse(insertion_candidate('abc','acbd'))

    def test_numbers_punctuation_unicode_not_silently_rewritten(self):
        for a,b in [('M.O.T.','MOT'),('123','1234'),('café','cafés'),('','word')]:
            self.assertFalse(insertion_candidate(a,b))

    def test_original_already_complete_is_not_a_completion(self):
        self.assertFalse(insertion_candidate('HOTEL','hotel'))

    def test_letter_preservation_allows_boundary_repair(self):
        self.assertTrue(preserves_letters(['NEWYORK'],'NEW YORK'))
        self.assertTrue(preserves_letters(['RES','TAURANT'],'RESTAURANT'))
        self.assertFalse(preserves_letters(['WORLD'],'WORD'))
        self.assertFalse(preserves_letters(['REST','TAURANT'],'RESTAURANT'))
        self.assertFalse(preserves_letters(['M.O.T.'],'MOT'))

    def test_casefold_and_punctuation_metric(self):
        result=lexical_metrics(['HOTEL','MOT'],[{'text':'hotel'},{'text':'M.O.T.'}])
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['gt_words'],2)

    def test_duplicate_predictions_capped_by_annotation_multiplicity(self):
        self.assertEqual(lexical_metrics(['new','NEW'],[{'text':'new'}])['correct'],1)
        self.assertEqual(lexical_metrics(['new','NEW'],[{'text':'new'},{'text':'NEW'}])['correct'],2)

    def test_empty_observation_counts_as_prediction(self):
        result=lexical_metrics([''],[{'text':'new'}])
        self.assertEqual((result['predictions'],result['correct'],result['gt_words']),(1,0,1))
        self.assertEqual(lexical_metrics([],[])['f1'],0.)

    def test_recoveries_do_not_cancel_harms(self):
        changes=lexical_changes(['hello','bad'],['wrong','world'],[{'text':'hello'},{'text':'world'}])
        self.assertEqual((changes['recovered'],changes['damaged']),(1,1))

    def test_dictionary_boundaries_preserve_literal_characters(self):
        options=split_options('NEWYORK',self.prior,POLICY)
        self.assertIn(['NEW','YORK'],options)
        self.assertTrue(all(''.join(parts)=='NEWYORK' for parts in options))

    def test_split_uses_no_annotation_text(self):
        observations=[block('a','NEWYORK',[0,0,80,20])]
        with_labels=copy.deepcopy(observations);with_labels[0]['gt_text']='unrelated'
        self.assertEqual(generate_operations(observations,self.prior,POLICY),generate_operations(with_labels,self.prior,POLICY))

    def test_fragments_merge_in_spatial_order(self):
        observations=[block('right','TAURANT',[35,0,100,20]),block('left','RES',[0,0,30,20])]
        operations=generate_operations(observations,self.prior,POLICY)
        self.assertTrue(any(op['texts']==['RESTAURANT'] and op['parents']==['left','right'] for op in operations))

    def test_unrelated_rows_do_not_merge(self):
        observations=[block('a','RES',[0,0,30,20]),block('b','TAURANT',[35,100,100,120])]
        self.assertFalse(any(len(op['parents'])>1 for op in generate_operations(observations,self.prior,POLICY)))

    def test_exact_overlap_is_an_alternative(self):
        observations=[block('a','REST',[0,0,40,20]),block('b','TAURANT',[35,0,100,20])]
        operations=generate_operations(observations,self.prior,POLICY)
        self.assertTrue(any(op['op']=='overlap' and op['texts']==['RESTAURANT'] for op in operations))
        self.assertTrue(any(op['op']=='merge' and op['texts']==['RESTTAURANT'] for op in operations))

    def test_conflicting_operations_never_reuse_fragments(self):
        observations=[block('a','RES',[0,0,30,20]),block('b','TAURANT',[35,0,100,20])]
        operations=generate_operations(observations,self.prior,POLICY)
        selected=apply_operations(observations,operations,[.9]*len(operations),.5)
        self.assertEqual(sorted(p for op in selected for p in op['parents']),['a','b'])

    def test_keep_policy_preserves_empty_and_literal_observations(self):
        observations=[block('a','',[0,0,30,20]),block('b','M.O.T.',[35,0,100,20])]
        selected=apply_operations(observations,[],[],1.01)
        self.assertEqual([op['texts'][0] for op in selected],['','M.O.T.'])

    def test_nonfinite_scores_fail(self):
        observations=[block('a','NEWYORK',[0,0,80,20])]
        operations=generate_operations(observations,self.prior,POLICY)
        with self.assertRaises(ValueError):apply_operations(observations,operations,[float('nan')]*len(operations),.5)

    def test_duplicate_ids_fail(self):
        observations=[block('a','new',[0,0,30,20]),block('a','york',[35,0,100,20])]
        with self.assertRaises(ValueError):generate_operations(observations,self.prior,POLICY)

    def test_feature_shape_and_values(self):
        observations=[block('a','NEWYORK',[0,0,80,20])]
        operation=generate_operations(observations,self.prior,POLICY)[0]
        self.assertEqual(len(operation_features(operation,observations,self.prior)),24)

    def test_oracle_respects_exclusivity_and_multiplicity(self):
        observations=[block('a','NEWYORK',[0,0,80,20]),block('b','new',[0,40,30,60])]
        operations=generate_operations(observations,self.prior,POLICY)
        result=structural_oracle(observations,operations,[{'text':'new'},{'text':'york'}])
        self.assertEqual(result['correct'],2)
        self.assertTrue(result['diagnostic_only'])

    def test_oracle_does_not_invent_unavailable_spelling(self):
        observations=[block('a','GARBAGE',[0,0,80,20])]
        result=structural_oracle(observations,[],[{'text':'new'}])
        self.assertEqual(result['correct'],0)

    def test_training_target_geometric_fusion(self):
        words=[dict(text='NEW',bbox=[0,0,30,20]),dict(text='YORK',bbox=[35,0,80,20])]
        self.assertEqual(supervised_target([0,0,80,20],words,minimum_coverage=.5),'NEW YORK')
        self.assertIsNone(supervised_target([100,100,120,120],words,minimum_coverage=.5))

    def test_synthetic_corruption_is_seeded_and_preserves_labels(self):
        a=synthetic_pairs('Restaurant',random.Random(7));b=synthetic_pairs('Restaurant',random.Random(7))
        self.assertEqual(a,b)
        self.assertTrue(all(target=='Restaurant' for _,target,_ in a))
        self.assertIn(('Restaurant','Restaurant','identity'),a)

    def test_neural_output_limits_keep_original(self):
        policy=dict(maximum_output_words=3,maximum_output_bytes=95,maximum_edits=8,maximum_relative_edit=.5,protect_non_alphabetic=True)
        self.assertEqual(validate_repair(['TUDIO'],'STUDIO',policy),('STUDIO','accepted'))
        self.assertEqual(validate_repair(['M.O.T.'],'MOT',policy)[0],'M.O.T.')
        self.assertEqual(validate_repair(['NEWYORK'],'NEW YORK',policy),('NEW YORK','accepted'))
        self.assertEqual(validate_repair(['word'],'a narrative about a word',policy)[0],'word')

    def test_config_namespace_rejects_strict(self):
        from scripts.run_text_reconstruction_tests import checked_config
        with self.assertRaises(ValueError):checked_config(Path('configs/strict/nonexistent.json'),'approximate')
        with self.assertRaises(ValueError):checked_config(Path('configs/approximate/text_reconstruction_tests_v1.json'),'strict')

    def test_denoiser_pairs_access_fit_labels_only(self):
        from scripts.run_text_reconstruction_tests import training_pairs
        sample=dict(sample_id='fit',blocks=[block('a','REST',[0,0,40,20])],operations=[])
        data=dict(fit=[sample],calibration=[dict(sample_id='forbidden')],evaluation=[dict(sample_id='forbidden')],test=[dict(sample_id='forbidden')])
        def labels(item):
            self.assertEqual(item['sample_id'],'fit')
            return [dict(text='Restaurant',bbox=[0,0,100,20])]
        cfg=dict(seed=3,denoiser=dict(training_gt_coverage=.5,max_input_bytes=128,max_target_bytes=96,maximum_training_pairs=100))
        with patch('scripts.run_text_reconstruction_tests.annotations',side_effect=labels):pairs,counts=training_pairs(data,cfg)
        self.assertTrue(pairs)
        self.assertTrue(all(p['target']=='Restaurant' for p in pairs))

    def test_oom_receipt_and_exception_propagation(self):
        import torch
        from scripts.run_text_reconstruction_tests import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run'
            argv=['runner','--mode','approximate','--config','configs/approximate/text_reconstruction_tests_v1.json','--output',str(root),'--action','prepare']
            with patch('sys.argv',argv),patch('scripts.run_text_reconstruction_tests.checked_config',return_value={}),patch('scripts.run_text_reconstruction_tests.setup',side_effect=torch.cuda.OutOfMemoryError('injected OOM')):
                with self.assertRaises(torch.cuda.OutOfMemoryError):main()
            self.assertTrue(json.loads((root/'failure.json').read_text())['oom'])
            self.assertFalse((root/'results.json').exists())


if __name__=='__main__':unittest.main()
