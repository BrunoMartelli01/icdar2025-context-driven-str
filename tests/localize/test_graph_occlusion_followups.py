"""Focused scientific tests for the four follow-up experiments."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from localize.graph_occlusion_followups import (predicted_strip_mask,apply_native_occlusion,occlude_original_rgb,
    positive_edges,use_enriched_graph,rectangle_points,resample_boundary,ring_adjacency,
    ContourPointRefiner,cyclic_contour_loss,contour_boxes)


class FollowupTests(unittest.TestCase):
    def test_cuts_gt_blind_reproducible_and_bounded(self):
        boxes=[[1,2,15,6],[17,1,20,17]];kw=dict(fraction=.15,maximum_words=2)
        a=predicted_strip_mask(boxes,(20,24),torch.Generator().manual_seed(4),**kw)
        b=predicted_strip_mask(boxes,(20,24),torch.Generator().manual_seed(4),**kw)
        np.testing.assert_array_equal(a,b);self.assertGreater(a.sum(),0)
        valid=np.zeros_like(a);valid[2:6,1:15]=True;valid[1:17,17:20]=True
        self.assertFalse((a&~valid).any())

    def test_native_rgb_alignment_and_input_preservation(self):
        image=torch.ones(1,3,8,8);mask=np.zeros((4,4),bool);mask[1,2]=True
        output=apply_native_occlusion(image,mask)
        self.assertEqual(int((output==0).sum()),12);self.assertTrue((image==1).all())

    def test_original_rgb_letterbox_cut_and_fill(self):
        rgb=np.ones((4,8,3),np.uint8)*255;mask=np.zeros((8,8),bool);mask[2:6,3:5]=True
        value=occlude_original_rgb(rgb,mask,[128,128,128])
        self.assertTrue(np.all(value[:,3:5]==128));self.assertTrue(np.all(value[:,:3]==255));self.assertTrue(np.all(rgb==255))

    def test_sampling_uses_train_links_and_equal_image_updates(self):
        graph=dict(assignments=np.array([1,2]),eligible=np.array([True,True]),edges=np.array([[0,1]]))
        augmented={**graph,'assignments':np.array([1,1])}
        self.assertEqual(positive_edges(graph),0);self.assertEqual(positive_edges(augmented),1)
        self.assertTrue(use_enriched_graph(graph,augmented,torch.Generator().manual_seed(1),1.))
        self.assertFalse(use_enriched_graph(augmented,graph,torch.Generator().manual_seed(1),1.))

    def test_rectangle_extrema_and_exact_identity(self):
        boxes=[[2,3,12,8],[0,0,1,1]];points=rectangle_points(boxes,32)
        self.assertEqual(contour_boxes(points,points,boxes,16),boxes)
        np.testing.assert_array_equal(points[0].min(0),[2.5,3.5]);np.testing.assert_array_equal(points[0].max(0),[11.5,7.5])

    def test_resampling_boundary_clockwise_and_empty(self):
        mask=np.zeros((12,16),bool);mask[2:9,3:13]=True
        points,count=resample_boundary(mask,32);self.assertEqual(points.shape,(32,2));self.assertEqual(count,1)
        self.assertIsNone(resample_boundary(np.zeros_like(mask),32)[0])
        area=np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1))
        self.assertGreater(area,0)

    def test_ring_symmetry_neighbors_and_normalization(self):
        a=ring_adjacency(16);self.assertTrue(torch.equal(a,a.T))
        self.assertTrue(torch.allclose(a.sum(1),torch.ones(16)))
        self.assertTrue(torch.all((a>0).sum(1)==5))

    def test_refiner_identity_equal_parameters_and_finite_gradients(self):
        torch.manual_seed(1);gnn=ContourPointRefiner(3,8,2,message_passing=True,iterations=2,maximum_step=.1)
        control=ContourPointRefiner(3,8,2,message_passing=False,iterations=2,maximum_step=.1)
        control.load_state_dict(gnn.state_dict(),strict=True)
        boxes=torch.tensor([[2.,2.,12.,10.]]);initial=torch.tensor(rectangle_points(boxes.numpy(),16))
        canvas=torch.randn(1,3,16,16)
        prediction=gnn(canvas,initial,boxes,16);self.assertTrue(torch.equal(prediction,initial))
        loss=cyclic_contour_loss(prediction,initial+1,torch.tensor([True]),boxes,beta=.1)
        loss.backward();self.assertIsNotNone(gnn.offset.weight.grad)
        self.assertTrue(torch.isfinite(gnn.offset.weight.grad).all());self.assertGreater(float(gnn.offset.weight.grad.abs().sum()),0)
        self.assertEqual(sum(p.numel() for p in gnn.parameters()),sum(p.numel() for p in control.parameters()))

    def test_sampler_retains_coordinate_gradient_after_first_update(self):
        model=ContourPointRefiner(2,8,1,message_passing=True,iterations=2,maximum_step=.05)
        torch.nn.init.normal_(model.offset.weight,std=.01)
        boxes=torch.tensor([[2.,2.,12.,10.]]);initial=torch.tensor(rectangle_points(boxes.numpy(),16),requires_grad=True)
        prediction=model(torch.randn(1,2,16,16),initial,boxes,16)
        prediction.square().mean().backward();self.assertTrue(torch.isfinite(initial.grad).all())

    def test_cyclic_loss_rotation_and_invalid_mask(self):
        boxes=torch.tensor([[0.,0.,10.,10.]]);points=torch.tensor(rectangle_points(boxes.numpy(),16),requires_grad=True)
        self.assertLess(float(cyclic_contour_loss(points,points.roll(3,1),torch.tensor([True]),boxes,beta=.1)),1e-7)
        zero=cyclic_contour_loss(points,points,torch.tensor([False]),boxes,beta=.1);zero.backward();self.assertEqual(float(zero),0)

    def test_namespace_isolation_before_any_model_or_gpu_load(self):
        from scripts.run_graph_occlusion_followups import prepare,REPO
        config=dict(reproduction_mode='approximate',experiment_kind='graph_occlusion_followups_v2')
        with patch('scripts.run_graph_occlusion_followups.read',return_value=config):
            for mode,namespace in [('strict','approximate'),('approximate','strict'),('approximate','demo')]:
                with self.assertRaisesRegex(ValueError,'APPROXIMATE'):
                    prepare(SimpleNamespace(mode=mode,config=REPO/'configs'/namespace/'fixture.json'))

    def test_prior_artifact_drift_propagates(self):
        from scripts.run_graph_occlusion_followups import previous
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data.json').write_text('{}')
            (root/'manifest.json').write_text(json.dumps({'artifacts':{'data.json':{'sha256':'0'*64}}}))
            config=dict(prior_run=str(root),paths={'prior_artifact_manifest':str(root/'manifest.json')})
            with self.assertRaisesRegex(ValueError,'changed'):previous(config,'data.json')

    def test_contour_trace_hash_tracks_deformed_rgb_before_filtering(self):
        from scripts.run_graph_occlusion_followups import contour_records
        from scripts.evaluate_cached_totaltext_padding import crop_hash
        from localize.occlusion_four_tests import row
        rgb=np.arange(16*16*3,dtype=np.uint16).reshape(16,16,3).astype(np.uint8)
        proposal=row([4,4,10,10],(16,16),'p','baseline');proposal['bbox_native']=[4,4,10,10]
        proposal['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[4:10,4:10]))
        record=dict(sample_id='fixture',image_hw=[16,16],arms={'mk':{'pool':[proposal]}},features={'mk':{}})
        arrays=dict(initial=rectangle_points([[4,4,10,10]],16),boxes=np.array([[4,4,10,10]],np.float32),
            feature_map=np.zeros((2,8,8),np.float32),probabilities=np.ones((2,16,16),np.float32)*.9)
        class Shift(torch.nn.Module):
            def forward(self,feature_map,initial,boxes,native_size):return initial+torch.tensor([1.,0.])
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.run_graph_occlusion_followups.load_contour',return_value=({'record':record},arrays)),patch('scripts.run_graph_occlusion_followups.load_rgb',return_value=rgb),patch.object(torch.Tensor,'cuda',lambda self:self):
                contour_records(Shift(),{},Path(directory),[{'sample_id':'fixture'}],'test',
                    {'resolved_config':{'postprocessing':{'text_threshold':.65}}},Path(directory)/'trace')
            trace=json.loads((Path(directory)/'trace/fixture.json').read_text())['record']['arms']['mk']['pool'][0]
            self.assertEqual(trace['bbox'],[5,4,11,10])
            self.assertEqual(trace['crop_array_sha256'],crop_hash(np.ascontiguousarray(rgb[4:10,5:11])))


if __name__=='__main__':unittest.main()
