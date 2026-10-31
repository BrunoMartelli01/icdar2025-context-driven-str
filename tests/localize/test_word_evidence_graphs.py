"""Scientific boundaries for visual ownership, alignment and active observation."""
import copy
import inspect
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from localize.word_evidence_graphs import (normalize_edges,relation_edges,visual_regions,
    EvidenceGraph,align_characters,consensus_reading,action_crop)
from localize.word_graph_recovery import preserve_and_add
from recognition.word_evidence_alignment import character_centers
from scripts.word_evidence_active import state_features
from scripts.word_evidence_active import ActionReader
from scripts.evaluate_cached_totaltext_padding import crop_hash


def proposal(name,box):
    return dict(id=name,bbox=box.copy(),bbox_native=box.copy(),padded_bbox_original=box.copy(),
        parents=[],method='baseline',text=name,crop_array_sha256=name)


class RelationTests(unittest.TestCase):
    def test_conflict_is_not_corroboration(self):
        pool=[proposal('a',[0,0,10,10]),proposal('b',[0,0,10,10]),proposal('c',[15,0,25,10])]
        support,conflict,collapsed=relation_edges(pool,np.ones((3,3),np.float32),overlap=.6)
        self.assertEqual(support[0,1],0.);self.assertGreater(conflict[0,1],0.)
        self.assertGreater(support[0,2],0.);self.assertEqual(conflict[0,2],0.)
        self.assertTrue(np.isfinite(collapsed).all())

    def test_no_neighbors_has_finite_identity(self):
        np.testing.assert_array_equal(normalize_edges(np.zeros((2,2)),self_edges=True),np.eye(2))
        self.assertEqual(normalize_edges(np.empty((0,0))).shape,(0,0))
        with self.assertRaises(ValueError):normalize_edges(np.array([[-1.]]))

    def test_equal_capacity_controls_and_permutation(self):
        torch.manual_seed(1);a=EvidenceGraph(5,8,2,typed=True,ownership=True)
        b=EvidenceGraph(5,8,2,typed=False,ownership=False);b.load_state_dict(a.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x=torch.randn(3,5);r=torch.randn(2,12);inc=torch.tensor([[1.,0.],[0.,1.],[.4,.5]])
        support=torch.eye(3);conflict=torch.tensor([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
        collapsed=normalize_edges(np.ones((3,3)),self_edges=True);collapsed=torch.tensor(collapsed)
        original=a(x,support,conflict,collapsed,r,inc)
        order=torch.tensor([2,0,1]);regions=torch.tensor([1,0])
        value=a(x[order],support[order][:,order],conflict[order][:,order],collapsed[order][:,order],r[regions],inc[order][:,regions])
        self.assertTrue(torch.allclose(original[order],value,atol=1e-6))
        self.assertFalse(torch.allclose(original,b(x,support,conflict,collapsed,r,inc)))
        original.sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in a.parameters()))

    def test_empty_regions_are_valid(self):
        model=EvidenceGraph(5,8,2,typed=True,ownership=True)
        value=model(torch.randn(2,5),torch.eye(2),torch.zeros(2,2),torch.eye(2),torch.empty(0,12),torch.empty(2,0))
        self.assertEqual(tuple(value.shape),(2,));self.assertTrue(torch.isfinite(value).all())


class RegionTests(unittest.TestCase):
    def test_incidence_counts_foreground_not_bbox_area(self):
        probabilities=np.zeros((2,8,8),np.float32);probabilities[0,0:4,0:2]=.9
        pool=[proposal('a',[0,0,2,8]),proposal('b',[2,0,8,8])]
        regions,incidence=visual_regions(probabilities,np.ones((8,8)),pool,[pool[0]],tile_size=8,
            text_threshold=.5,kernel_threshold=.5,minimum_pixels=1,maximum_regions=4)
        np.testing.assert_array_equal(incidence,[[1.],[0.]])
        self.assertEqual(regions[0,9],1.);self.assertEqual(regions[0,10],0.)

    def test_missing_canvas_has_no_evidence(self):
        r,i=visual_regions(np.ones((2,8,8)),np.zeros((8,8)),[],[],tile_size=4,text_threshold=.5,
            kernel_threshold=.5,minimum_pixels=1,maximum_regions=4)
        self.assertEqual(r.shape,(0,12));self.assertEqual(i.shape,(0,0))


class CharacterTests(unittest.TestCase):
    def reading(self,name,text,confidence,centers=None):
        return dict(id=name,text=text,confidence=confidence,height=10,
            centers=centers if centers is not None else [[i*8,5] for i in range(len(text))])

    def test_two_wrong_readings_can_reconstruct_without_a_dictionary(self):
        a=self.reading('a','ADMI5SION',[.99]*4+[.2]+[.99]*4)
        b=self.reading('b','AOMISSION',[.9,.2]+[.9]*7)
        text,confidence,donors=consensus_reading(a,[b],maximum_distance=.35,confidence_margin=.02)
        self.assertEqual(text,'ADMISSION');self.assertEqual(donors[4],'b')

    def test_unrelated_spatial_character_cannot_be_copied(self):
        a=self.reading('a','a',[.1]);b=self.reading('b','b',[.99],[[100,100]])
        self.assertEqual(consensus_reading(a,[b],maximum_distance=.35,confidence_margin=.02)[0],'a')

    def test_different_lengths_are_not_invented_or_concatenated(self):
        a=self.reading('a','abcd',[.9,.2,.9,.9]);b=self.reading('b','B',[.99],[[8,5]])
        self.assertEqual(align_characters(a,b,maximum_distance=.35).tolist(),[-1,0,-1,-1])
        self.assertEqual(consensus_reading(a,[b],maximum_distance=.35,confidence_margin=.02)[0],'aBcd')

    def test_unicode_and_punctuation_are_copied_exactly(self):
        a=self.reading('a','é?',[.9,.1]);b=self.reading('b','é!',[.8,.99])
        self.assertEqual(consensus_reading(a,[b],maximum_distance=.35,confidence_margin=.02)[0],'é!')

    def test_coordinate_projection_uses_pixel_centers_and_non_square_grid(self):
        center,entropy=character_centers(np.eye(4)[:2],(1,4),[10,20,50,40])
        np.testing.assert_allclose(center,[[15,30],[25,30]])
        np.testing.assert_allclose(entropy,0.)
        with self.assertRaises(ValueError):character_centers(np.zeros((1,4)),(1,4),[0,0,4,1])


class ActiveTests(unittest.TestCase):
    def test_action_changes_recognition_pixels_without_mutating_parent(self):
        rgb=np.arange(20*30*3,dtype=np.uint8).reshape(20,30,3);p=proposal('a',[5,5,15,15]);before=copy.deepcopy(p)
        crop,box,removed=action_crop(rgb,p,[p],'left',fraction=.25,overlap=.6,isolation_periphery=.12)
        self.assertEqual(box,[2,5,15,15]);self.assertEqual(crop.shape,(10,13,3));self.assertEqual(p,before)
        self.assertEqual(removed,0)

    def test_unknown_action_fails(self):
        with self.assertRaises(ValueError):action_crop(np.zeros((10,10,3),np.uint8),proposal('a',[0,0,3,3]),[],'invent',fraction=.25,overlap=.6,isolation_periphery=.12)

    def test_unknown_action_results_cannot_enter_features(self):
        p=proposal('a',[5,5,15,15]);a=dict(proposal=dict(p,id='action:a:left',action='left'),parent=p,index=0,removed=0)
        sample=dict(arrays=dict(features=np.ones((1,224),np.float32)),words=[dict(text='SECRET',bbox=[0,0,1,1])])
        c=dict(active=dict(actions=['left','right','vertical','isolate'],query_budget=2),compatibility_iou=.6)
        before=state_features(sample,[a],{},[],c)
        sample['words']=[dict(text='CHANGED',bbox=[9,9,99,99])]
        after=state_features(sample,[a],{},[],c)
        np.testing.assert_array_equal(before,after);self.assertEqual(before.shape,(1,236))
        observed={'action:a:left':dict(raw_text='NEW',token_probabilities=[.2,.3,.4])}
        self.assertFalse(np.array_equal(before,state_features(sample,[a],observed,[],c)))

    def test_appending_never_mutates_an_existing_observation(self):
        p=proposal('a',[0,0,5,5]);q=proposal('b',[10,0,15,5]);before=copy.deepcopy([p])
        result=preserve_and_add([p],[p,q],{'a':0.,'b':.9},threshold=.5,overlap=.6)
        self.assertEqual(result[:1],before)

    def test_inference_primitives_have_no_annotation_arguments(self):
        for function in (relation_edges,visual_regions,consensus_reading,action_crop):
            self.assertFalse({'words','annotations','ground_truth','owners'}&set(inspect.signature(function).parameters))

    def test_oom_propagates_without_cached_dummy_readings(self):
        crop=np.zeros((4,4,3),np.uint8);key=crop_hash(crop)
        def fail(images):raise torch.cuda.OutOfMemoryError('controlled interface OOM')
        with tempfile.TemporaryDirectory() as directory:
            fake=SimpleNamespace(model=torch.nn.Linear(1,1),recognize_candidate_batch=fail)
            reader=ActionReader.__new__(ActionReader);reader.c={'runtime':{'ocr_batch_size':1}}
            reader.cache=SimpleNamespace(rows={},recognizer=lambda:fake,output=Path(directory)/'ocr.jsonl')
            reader.physical_new=0;reader.current_digest=None;reader.state_audits=[]
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                reader.read([dict(crop=crop,proposal=dict(id='action',crop_array_sha256=key))])
            self.assertEqual(reader.cache.rows,{});self.assertEqual(reader.physical_new,0)


if __name__=='__main__':unittest.main()
