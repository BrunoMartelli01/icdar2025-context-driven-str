"""Scientific invariants for non-destructive graphs and word-level objectives."""
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from localize.word_graph_recovery import (HypothesisScorer,conflicts,preserve_and_add,
    marginal_training_targets,addition_features,hypothesis_adjacency,training_owner_targets,
    point_extents,complete_word_loss,geometry_targets,dense_geometry_loss,regional_distillation)
from localize.graph_occlusion_followups import rectangle_points


def proposal(name,box,method='baseline',parents=()):
    return dict(id=name,bbox=box,padded_bbox_original=box.copy(),method=method,parents=list(parents),crop_array_sha256=name)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.a=proposal('a',[0,0,10,10]);self.b=proposal('b',[20,0,30,10]);self.duplicate=proposal('c',[0,0,10,10])
        self.readings={k:{'raw_text':v,'token_probabilities':[.9]} for k,v in [('a','WORD'),('b','other'),('c','word')]}
        self.words=[dict(bbox=[0,0,10,10],text='word'),dict(bbox=[20,0,30,10],text='OTHER')]

    def test_high_score_duplicate_cannot_remove_or_replace_baseline(self):
        baseline=[self.a];original=copy.deepcopy(baseline)
        result=preserve_and_add(baseline,[self.a,self.b,self.duplicate],{'a':0.,'b':.8,'c':1.},threshold=.5,overlap=.6)
        self.assertEqual([p['id'] for p in result],['a','b']);self.assertEqual(baseline,original)
        result[0]['bbox'][0]=1;self.assertEqual(baseline,original)

    def test_every_baseline_observation_is_locked_even_if_mutually_conflicting(self):
        baseline=[self.a,self.duplicate]
        result=preserve_and_add(baseline,[*baseline,self.b],{'a':0.,'c':0.,'b':0.},threshold=.5,overlap=.6)
        self.assertEqual(result,baseline)

    def test_sibling_splits_coexist_but_parent_cannot_be_added(self):
        a=proposal('x',[0,0,4,4],'split',['parent']);b=proposal('y',[6,0,10,4],'split',['parent'])
        parent=proposal('parent',[0,0,10,4])
        self.assertFalse(conflicts(a,b,.6));self.assertTrue(conflicts(a,parent,.6))
        pp=preserve_and_add([a],[a,b,parent],{'x':0.,'y':.8,'parent':.7},threshold=.5,overlap=.6)
        self.assertEqual([p['id'] for p in pp],['x','y'])

    def test_targets_are_incremental_matching_not_ordinary_correctness(self):
        y,valid=marginal_training_targets([self.a],[self.a,self.b,self.duplicate],self.words,self.readings,overlap=.6)
        np.testing.assert_array_equal(y,[0,1,0]);np.testing.assert_array_equal(valid,[False,True,False])

    def test_punctuation_retained_and_unicode_casefold(self):
        b=proposal('b',[20,0,30,10]);readings={'b':dict(raw_text='STRASSE')}
        y,_=marginal_training_targets([], [b], [dict(bbox=b['bbox'],text='Straße')],readings,overlap=.6)
        self.assertEqual(float(y[0]),1.)
        readings['b']['raw_text']='STRASSE.'
        y,_=marginal_training_targets([], [b], [dict(bbox=b['bbox'],text='Straße')],readings,overlap=.6)
        self.assertEqual(float(y[0]),0.)

    def test_invalid_scores_and_baseline_subset_fail(self):
        for score in [float('nan'),float('inf'),-.1,1.1]:
            with self.assertRaises(ValueError):preserve_and_add([],[self.a],{'a':score},threshold=.5,overlap=.6)
        with self.assertRaises(ValueError):preserve_and_add([self.a],[self.b],{'b':.9},threshold=.5,overlap=.6)
        with self.assertRaises(ValueError):preserve_and_add([],[self.a,self.a],{'a':.9},threshold=.5,overlap=.6)

    def test_features_are_deterministic_and_inference_has_no_gt_argument(self):
        args=([self.a,self.b],[self.a],{'a':[1.,2.],'b':[2.,3.]},self.readings,[[1.,0.],[0.,1.]])
        a=addition_features(*args,text_bins=8,overlap=.6);b=addition_features(*args,text_bins=8,overlap=.6)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all())
        for function in [addition_features,hypothesis_adjacency,preserve_and_add]:
            self.assertFalse({'owners','words','annotations','validity'} & set(inspect.signature(function).parameters))

    def test_graph_control_capacity_and_permutation_equivariance(self):
        torch.manual_seed(2);a=HypothesisScorer(4,8,2,message_passing=True);b=HypothesisScorer(4,8,2,message_passing=False)
        b.load_state_dict(a.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x=torch.randn(3,4);adj=torch.tensor([[.5,.5,0.],[.5,.5,0.],[0.,0.,1.]])
        order=torch.tensor([2,0,1]);self.assertTrue(torch.allclose(a(x,adj)[order],a(x[order],adj[order][:,order]),atol=1e-6))
        self.assertFalse(torch.allclose(a(x,adj),b(x,adj)))
        a(x,adj).sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in a.parameters()))


class CompletenessTests(unittest.TestCase):
    def test_owner_target_uses_full_word_not_cropped_visible_support(self):
        owner=np.zeros((16,16),np.int32);owner[3:10,2:13]=1
        target,ids,valid=training_owner_targets([[4,3,12,10]],owner,np.ones_like(owner),minimum_iou=.3,minimum_valid_fraction=.95)
        np.testing.assert_array_equal(target,[[2,3,13,10]]);self.assertEqual(ids.tolist(),[1]);self.assertTrue(valid[0])

    def test_invalid_owner_window_has_no_supervision(self):
        owner=np.zeros((8,8),np.int32);owner[2:5,2:6]=1
        _,_,valid=training_owner_targets([[1,1,7,7]],owner,np.zeros_like(owner),minimum_iou=.3,minimum_valid_fraction=.95)
        self.assertFalse(valid[0])

    def loss(self,box,owner=None,**weights):
        source=torch.tensor([[2.,2.,12.,12.]])
        p=torch.tensor(rectangle_points([box],16),requires_grad=True)
        target=source.clone();owner=torch.zeros(16,16,dtype=torch.long) if owner is None else owner
        result=complete_word_loss(p,source,target,torch.tensor([1]),owner,torch.tensor([True]),
            inside_weight=weights.get('inside',2.),extent_weight=weights.get('extent',1.),contamination_weight=weights.get('other',.5))
        result.backward();self.assertTrue(torch.isfinite(p.grad).all());return float(result)

    def test_inward_clipping_costs_more_than_equal_outward_motion(self):
        self.assertGreater(self.loss([3,3,11,11]),self.loss([1,1,13,13]))

    def test_other_word_contamination_is_penalized(self):
        owner=torch.zeros(16,16,dtype=torch.long);owner[3:11,12:15]=2
        self.assertGreater(self.loss([2,2,15,12],owner,inside=0,extent=0,other=1),self.loss([2,2,10,12],owner,inside=0,extent=0,other=1))

    def test_pixel_center_exclusive_extents_match_identity(self):
        boxes=torch.tensor([[1.,2.,8.,9.]])
        self.assertTrue(torch.equal(point_extents(torch.tensor(rectangle_points(boxes.numpy(),16))),boxes))

    def test_empty_valid_target_keeps_zero_differentiable_loss(self):
        points=torch.randn(1,16,2,requires_grad=True);boxes=torch.tensor([[0.,0.,8.,8.]])
        loss=complete_word_loss(points,boxes,boxes,torch.tensor([0]),torch.zeros(8,8),torch.tensor([False]),inside_weight=2,extent_weight=1,contamination_weight=1)
        loss.backward();self.assertEqual(float(loss),0.);self.assertTrue(torch.equal(points.grad,torch.zeros_like(points)))


class GeometryAndRegionalTests(unittest.TestCase):
    def test_geometric_fields_are_finite_owner_specific_and_background_zero(self):
        owner=np.zeros((16,20),np.int32);owner[2:8,2:9]=1;owner[9:14,12:18]=2
        field=geometry_targets(owner);self.assertEqual(field.shape,(3,16,20));self.assertTrue(np.isfinite(field).all())
        self.assertTrue(np.all(field[:,owner==0]==0));self.assertAlmostEqual(float(field[0,owner==1].max()),1.)
        np.testing.assert_allclose(np.linalg.norm(field[1:,owner>0],axis=0),1.,atol=1e-6)

    def test_dense_geometry_zero_vector_has_finite_gradient(self):
        raw=torch.zeros(1,3,8,8,requires_grad=True);owners=np.zeros((8,8),np.int32);owners[2:6,2:6]=1
        target=torch.tensor(geometry_targets(owners))[None];fg=torch.tensor(owners>0)[None,None]
        loss=dense_geometry_loss(raw,target,fg,torch.ones_like(fg),magnitude_weight=.1);loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all());self.assertGreater(float(raw.grad.abs().sum()),0.)

    def test_regional_loss_detaches_teacher_and_ignores_unknown_regions(self):
        logits=torch.zeros(1,2,8,8,requires_grad=True);teacher=torch.ones_like(logits,requires_grad=True)*.9;teacher.retain_grad()
        loss=regional_distillation(logits,teacher,[[dict(bbox_native=[2,2,6,6])]],temperature=1.,extent_weight=1.)
        loss.backward();self.assertIsNone(teacher.grad);self.assertEqual(float(logits.grad[:,:,:2].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,:,2:6,2:6].abs().sum()),0.)

    def test_no_verified_region_produces_no_distillation_gradient(self):
        logits=torch.randn(1,2,8,8,requires_grad=True)
        loss=regional_distillation(logits,torch.rand_like(logits),[[]],temperature=1.,extent_weight=1.)
        loss.backward();self.assertEqual(float(loss),0.);self.assertEqual(float(logits.grad.abs().sum()),0.)

    def test_wrong_ocr_and_partial_teacher_words_are_rejected(self):
        from scripts.word_graph_regional_stages import verify_fit_regions
        owners=np.zeros((8,8),np.int64);owners[1:5,1:5]=1
        good=proposal('good',[1,1,5,5]);good['bbox_native']=good['bbox']
        bad=proposal('bad',[1,1,5,5]);bad['bbox_native']=bad['bbox']
        records={'s':dict(verified_admitted=[bad,good])};prob=np.ones((2,8,8),np.float32)*.95
        cfg=dict(minimum_confidence=.8,minimum_iou=.5,text_threshold=.65,minimum_purity=.9,minimum_coverage=.8)
        batches={'s':dict(owners_native=torch.tensor(owners))};words={'s':[dict(bbox=[1,1,5,5],text='YES')]}
        readings={'good':dict(raw_text='yes',token_probabilities=[.9]),'bad':dict(raw_text='NO',token_probabilities=[.99])}
        accepted,audit=verify_fit_regions(records,{'s':prob},batches,words,readings,cfg)
        self.assertEqual(len(accepted['s']),1);self.assertEqual(audit['s']['wrong_text_or_geometry'],1)
        prob[0,1:5,1:3]=0
        accepted,_=verify_fit_regions(records,{'s':prob},batches,words,readings,cfg)
        self.assertEqual(accepted['s'],[])


class BoundaryTests(unittest.TestCase):
    def test_full_supervised_batch_retains_separation_and_rejects_missing_fields(self):
        from scripts.run_word_graph_recovery import training_batch
        from scripts.run_teacher_student_tests import supervised_loss
        example=dict(image=torch.zeros(3,8,8),mask=torch.zeros(1,8,8),kernel=torch.zeros(1,8,8),
            validity=torch.ones(1,8,8),separation=torch.zeros(1,8,8),owners_native=torch.zeros(8,8,dtype=torch.long))
        example['separation'][:,-2:]=1
        batch=training_batch([example],device='cpu');logits=torch.zeros(1,2,8,8,requires_grad=True)
        with torch.no_grad():logits[0,0,-2:]=3
        settings=dict(bce_weight=1.,dice_weight=1.,smooth=1.,separation_weight=4.)
        weighted=supervised_loss(logits,batch,settings)
        ordinary=supervised_loss(logits,batch,{**settings,'separation_weight':1.})
        self.assertGreater(float(weighted),float(ordinary));weighted.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        del example['separation']
        with self.assertRaisesRegex(ValueError,'separation'):training_batch([example],device='cpu')

    def test_namespace_and_independence_fail_before_runtime(self):
        from scripts.run_word_graph_recovery import validate_namespace,REPO
        c=dict(reproduction_mode='approximate',experiment_kind='word_graph_recovery_v1',metric_iou_threshold=.5,budget=20,
            require_independent_validation=False,new_decisions=[dict(classification='ASSUMPTION')])
        for mode,namespace in [('strict','approximate'),('approximate','strict'),('approximate','demo')]:
            with self.assertRaises(ValueError):validate_namespace(mode,REPO/'configs'/namespace/'a.json',c)
        with self.assertRaisesRegex(ValueError,'NOT independent'):validate_namespace('approximate',REPO/'configs/approximate/a.json',{**c,'require_independent_validation':True})

    def test_contour_artifact_corruption_fails(self):
        from scripts.run_word_graph_recovery import followup_contour
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'contour_data/fit/s.json';p.parent.mkdir(parents=True);p.write_text('{}')
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(artifacts={'contour_data/fit/s.json':dict(sha256='0'*64)})))
            with self.assertRaisesRegex(ValueError,'changed'):followup_contour(str(root),str(manifest),'fit','s')

    def test_oom_failure_receipt_is_explicit(self):
        from scripts.run_graph_occlusion_research import record_failure
        with tempfile.TemporaryDirectory() as directory:
            record_failure(Path(directory),torch.cuda.OutOfMemoryError('injected qualification failure'))
            status=json.loads((Path(directory)/'failure.json').read_text());self.assertEqual(status['status'],'FAILED');self.assertTrue(status['oom'])


if __name__=='__main__':unittest.main()
