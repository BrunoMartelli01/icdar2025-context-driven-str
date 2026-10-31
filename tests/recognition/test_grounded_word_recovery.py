"""Scientific boundary tests for RGB-grounded word recovery."""
import copy
import unittest
import numpy as np
import torch
from types import SimpleNamespace

from recognition.grounded_word_recovery import (KEEP,DELETE,STOP,BOS,PAD,edit_script,apply_edits,
    pointer_sequence,expanded_box,tile_boxes,map_tile_box,geometry_features,operation,decide,choose_target)
from models.grounded_word_recovery import GroundedVerifier,VisualEditTransducer
from scripts.grounded_word_search import query_features,query_adjacency
from localize.word_graph_recovery import HypothesisScorer
from scripts.run_grounded_word_recovery import VisualBank,run


def proposal(name,text='word',box=(0,0,10,10)):
    return dict(id=name,text=text,bbox=list(box),padded_bbox_original=list(box),parents=[],method='baseline')


class EditTests(unittest.TestCase):
    def test_exact_edit_reconstruction(self):
        for source,target in [('',''),('','A'),('abc',''),('IRAVELER','TRAVELER'),('JAY','JAYS'),
                              ('HAIGHT-','HAIGHT'),('Romance','Romance'),('é?','è!'),('Straße','STRASSE')]:
            with self.subTest(source=source,target=target):
                value=apply_edits(source,edit_script(source,target),64)
                self.assertTrue(value['valid']);self.assertEqual(value['text'],target)

    def test_invalid_actions_are_explicit(self):
        for source,actions,reason in [('a',[STOP],'unconsumed_source'),('',[KEEP,STOP],'source_overrun'),
            ('',[255,STOP],'invalid_utf8'),('a',[KEEP],'missing_stop')]:
            self.assertEqual(apply_edits(source,actions,64),dict(valid=False,reason=reason))

    def test_source_pointer(self):
        self.assertEqual(pointer_sequence([KEEP,65,DELETE,STOP]),[0,1,1,2])

    def test_maximum_output_is_enforced(self):
        self.assertEqual(apply_edits('',[65,66,STOP],1)['reason'],'output_limit')

    def test_generated_scripts_consume_source_and_terminate(self):
        torch.manual_seed(4)
        model=VisualEditTransducer(8,16,1,4,visual=True).eval()
        source=torch.tensor([[BOS,65,66],[BOS,67,PAD]])
        scripts=model.generate(source,torch.randn(2,3,8),[2,1])
        for original,script in zip(['AB','C'],scripts):
            self.assertEqual(script[-1],STOP)
            value=apply_edits(original,script,4)
            self.assertNotIn(value.get('reason'),['source_overrun','unconsumed_source','missing_stop','output_limit'])

    def test_text_only_has_identical_capacity_and_no_visual_dependence(self):
        torch.manual_seed(4);a=VisualEditTransducer(8,16,1,4,visual=False).eval()
        torch.manual_seed(4);b=VisualEditTransducer(8,16,1,4,visual=True).eval()
        self.assertEqual({k:v.shape for k,v in a.state_dict().items()},{k:v.shape for k,v in b.state_dict().items()})
        source=torch.tensor([[BOS,65]]);actions=torch.tensor([[BOS,KEEP]]);cursor=torch.tensor([[0,1]])
        with torch.no_grad():
            left=a(source,torch.zeros(1,3,8),actions,cursor)
            right=a(source,torch.ones(1,3,8),actions,cursor)
        torch.testing.assert_close(left,right,rtol=0,atol=0)


class PolicyTests(unittest.TestCase):
    def test_replacement_and_immutable_input(self):
        original=[proposal('a','OUL')];candidate=proposal('b','Out');before=copy.deepcopy(original)
        result=decide(original,[candidate],{'a':.5,'b':.9},threshold=.7,margin=.1,overlap=.6,replacements=True)
        self.assertEqual(result[0]['text'],'Out');self.assertEqual(original,before)

    def test_addonly_protects_original(self):
        p=proposal('a');q=proposal('b','wrong')
        self.assertEqual(decide([p],[q],{'a':.1,'b':.99},threshold=.5,margin=0,overlap=.6,replacements=False),[p])

    def test_multiple_conflicts_abstain(self):
        a=proposal('a',box=(0,0,10,10));b=proposal('b',box=(10,0,20,10))
        q=proposal('c',box=(0,0,20,10));q['parents']=['a','b']
        self.assertIsNone(operation([a,b],q,overlap=.6))

    def test_marginal_score_boundary_is_conservative(self):
        a=proposal('a');b=proposal('b','new')
        result=decide([a],[b],{'a':.5,'b':.75},threshold=.5,margin=.25,overlap=.6,replacements=True)
        self.assertEqual(result,[a])

    def test_keep_and_empty_denominators_remain_outputs(self):
        p=proposal('a','');self.assertEqual(decide([p],[],{},threshold='KEEP',margin=0,overlap=.6,replacements=True),[p])

    def test_nonfinite_validity_is_an_error(self):
        with self.assertRaises(ValueError):
            decide([],[proposal('a')],{'a':float('nan')},threshold=.5,margin=0,overlap=.6,replacements=True)

    def test_geometry_target_requires_iou_not_spelling(self):
        self.assertIsNone(choose_target(proposal('a'),[dict(text='word',bbox=[100,100,110,110])],.5))


class GeometryTests(unittest.TestCase):
    def test_local_to_original_coordinates(self):
        self.assertEqual(map_tile_box([2,3,12,14],[100,200,140,250]),[102,203,112,214])
        with self.assertRaises(ValueError):map_tile_box([0,0,50,50],[100,100,120,120])

    def test_context_and_tiles_stay_in_original_canvas(self):
        self.assertEqual(expanded_box([0,0,5,5],[10,20],.5),[0,0,8,8])
        for a,b,d,e in tile_boxes([11,21],.6):self.assertTrue(0<=a<d<=21 and 0<=b<e<=11)

    def test_query_features_never_read_gt_or_unacquired_OCR(self):
        actions=[dict(id='tile:0',box=[0,0,10,10],kind='tile',prior=0.)]
        sample=dict(image_hw=[20,20],words=[dict(text='SECRET')]);visual=np.ones((1,16,384),np.float32)
        c=dict(search=dict(dimensions=400));a=query_features(sample,actions,visual,[],{},c)
        sample['words']=[dict(text='CHANGED')];actions[0]['hypothetical_text']='LEAK'
        b=query_features(sample,actions,visual,[],{},c)
        np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(1,400))

    def test_query_adjacency_uses_geometry_without_text_fields(self):
        c={'search':{'neighbors':1,'maximum_distance':2.}}
        actions=[dict(box=[0,0,10,10]),dict(box=[11,0,21,10]),dict(box=[1000,0,1010,10])]
        adjacency=query_adjacency(actions,c)
        self.assertGreater(adjacency[0,1],0);self.assertEqual(adjacency[0,2],0)
        np.testing.assert_array_equal(adjacency,adjacency.T)

    def test_disjoint_graph_batch_preserves_independent_predictions(self):
        model=HypothesisScorer(4,8,2,message_passing=True).eval()
        a,b=torch.randn(2,4),torch.randn(3,4);ga,gb=torch.eye(2),torch.eye(3)
        with torch.no_grad():
            separate=torch.cat([model(a,ga),model(b,gb)])
            together=model(torch.cat([a,b]),torch.block_diag(ga,gb))
        torch.testing.assert_close(separate,together)

    def test_grounding_forward_and_gradient(self):
        model=GroundedVerifier(8,16,1,8);image=torch.randn(2,3,8,requires_grad=True)
        output=model(image,image,torch.tensor([[BOS,65],[BOS,66]]),torch.zeros(2,12))
        self.assertEqual(output.shape,(2,3));output.sum().backward()
        self.assertGreater(float(image.grad.abs().sum()),0.)

    def test_oom_is_propagated_without_cached_dummy_features(self):
        bank=VisualBank.__new__(VisualBank);bank.values={};bank.c={'runtime':{'encoding_batch_size':1}}
        def fail():raise torch.cuda.OutOfMemoryError('controlled qualification OOM')
        bank.recognizer=fail
        with self.assertRaises(torch.cuda.OutOfMemoryError):bank.encode([np.zeros((4,8,3),np.uint8)])
        self.assertEqual(bank.values,{})

    def test_strict_and_wrong_namespace_fail_before_runtime(self):
        for mode in ('strict','approximate'):
            with self.assertRaises(ValueError):run(SimpleNamespace(mode=mode,config='/tmp/not-an-approximate-config.json'))


if __name__=='__main__':unittest.main()
