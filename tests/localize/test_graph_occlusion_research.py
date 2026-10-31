"""Scientific contracts for fragment graphs and occlusion distillation."""
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from localize.graph_occlusion_research import (FragmentGraph, neighbor_graph, graph_groups,
    fragment_labels, graph_supervision, occlude_instances, teacher_reliability,
    pixel_distillation, instance_structure_distillation, balanced_binary_loss)


class GraphOcclusionTests(unittest.TestCase):
    def test_fragments_partition_support_and_keep_unseeded_remainder(self):
        labels = np.zeros((8, 12), np.int32); labels[1:5, 1:9] = 1; labels[6, 10] = 2
        prob = np.full((2, 8, 12), .1, np.float32); prob[0, labels > 0] = .9
        prob[1, 2:4, 2:4] = .95; prob[1, 2:4, 6:8] = .95
        result, audit = fragment_labels(prob, labels, threshold=.8, minimum_seed_area=2,
                                       maximum_pieces=4, maximum_nodes=20)
        self.assertEqual(audit['atoms'], 3); np.testing.assert_array_equal(result > 0, labels > 0)
        self.assertNotEqual(result[2, 2], result[2, 6])

    def test_cap_reports_dropped_pixels(self):
        labels = np.array([[1, 1, 0, 2]], np.int32); p = np.full((2, 1, 4), .2)
        _, audit = fragment_labels(p, labels, threshold=.8, minimum_seed_area=1,
                                   maximum_pieces=4, maximum_nodes=1)
        self.assertEqual(audit['dropped_pixels'], 1)

    def test_graph_empty_symmetry_and_distance_gate(self):
        a, e = neighbor_graph([], neighbors=3, maximum_distance=2)
        self.assertEqual(e.shape, (0, 2))
        a, e = neighbor_graph([[0,0,4,4], [5,0,9,4], [100,0,104,4]], neighbors=2, maximum_distance=2)
        np.testing.assert_array_equal(e, [[0,1]]); np.testing.assert_allclose(a, a.T)

    def test_parameter_control_and_neighbor_effect(self):
        torch.manual_seed(4); graph = FragmentGraph(3, 8, 2, True); control = FragmentGraph(3, 8, 2, False)
        control.load_state_dict(graph.state_dict(), strict=True)
        x = torch.randn(3, 3); a = torch.eye(3); e = torch.tensor([[0,1]])
        self.assertTrue(torch.equal(graph(x,a,e)[0], control(x,a,e)[0]))
        a = torch.full((3,3), 1/3)
        self.assertFalse(torch.equal(graph(x,a,e)[0], control(x,a,e)[0]))
        self.assertEqual(sum(p.numel() for p in graph.parameters()), sum(p.numel() for p in control.parameters()))
        self.assertTrue(torch.allclose(graph(x,a,e)[1], graph(x,a,e.flip(-1))[1]))

    def test_transitive_groups_threshold_and_inactive_node(self):
        boxes = [[0,0,1,1]] * 3; edges = np.array([[0,1],[1,2]])
        self.assertEqual(graph_groups(boxes,[.9]*3,edges,[.5,.6],node_threshold=.5,edge_threshold=.5), [[0,1,2]])
        self.assertEqual(graph_groups(boxes,[.9,.1,.9],edges,[.9,.9],node_threshold=.5,edge_threshold=.5), [[0],[2]])

    def test_training_assignments_ignore_ambiguous_and_invalid(self):
        labels = np.array([[1,1,2,2,3,3]]); owners = np.array([[7,7,8,9,0,0]])
        ids, valid = graph_supervision(labels, owners, np.ones_like(labels), purity=.8)
        np.testing.assert_array_equal(ids, [7,8,0]); np.testing.assert_array_equal(valid, [True,False,True])
        _, valid = graph_supervision(labels, owners, np.zeros_like(labels), purity=.8)
        self.assertFalse(valid.any())

    def test_occlusion_mapping_reproducibility_and_unchanged_input(self):
        owners = torch.zeros((1,4,8), dtype=torch.int64); owners[:,1:3,2:6] = 1
        x = torch.ones((1,3,8,16)); kwargs = dict(probability=1.,width_fraction=.25,maximum_words=4)
        a, mask = occlude_instances(x,owners,torch.Generator().manual_seed(3),**kwargs)
        b, _ = occlude_instances(x,owners,torch.Generator().manual_seed(3),**kwargs)
        self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.all(x == 1))
        self.assertEqual(int(mask.sum()),2); self.assertEqual(int((a == 0).sum()),24)

    def test_wrong_uncertain_and_ignore_teacher_pixels_rejected(self):
        teacher = torch.tensor([[[[.95,.95,.5,.05]]]])
        target = torch.tensor([[[[1.,0.,1.,0.]]]])
        valid = torch.tensor([[[[1.,1.,1.,0.]]]])
        mask = teacher_reliability(teacher,target,valid,.8)
        self.assertEqual(mask.flatten().tolist(),[True,False,False,False])

    def test_losses_detach_teacher_and_ignore_invalid_positions(self):
        logits = torch.zeros((1,2,2,4),requires_grad=True)
        teacher = torch.full_like(logits,.95,requires_grad=True)
        valid = torch.zeros_like(logits,dtype=torch.bool); valid[:,:,:,:2] = True
        owners = torch.ones((1,2,4),dtype=torch.int64)
        loss = pixel_distillation(logits,teacher,valid) + instance_structure_distillation(
            logits,teacher,valid,owners,torch.zeros((1,1,2,4)))
        loss.backward(); self.assertIsNone(teacher.grad)
        self.assertEqual(float(logits.grad[:,:,:,2:].abs().sum()),0.)
        self.assertGreater(float(logits.grad[:,:,:,:2].abs().sum()),0.)

    def test_no_supervision_is_finite_differentiable_zero(self):
        logits = torch.zeros(2,requires_grad=True)
        loss = balanced_binary_loss(logits,torch.ones(2),torch.zeros(2,dtype=torch.bool))
        loss.backward(); self.assertEqual(float(loss),0.)

    def test_strict_and_other_namespaces_fail_before_runtime(self):
        from scripts.run_graph_occlusion_research import prepare, REPO
        c = dict(reproduction_mode='approximate',experiment_kind='graph_occlusion_research_v1')
        with patch('scripts.run_graph_occlusion_research.read',return_value=c):
            for mode, namespace in [('strict','approximate'),('approximate','strict'),('approximate','demo')]:
                args = SimpleNamespace(mode=mode,config=REPO/'configs'/namespace/'fixture.json')
                with self.assertRaisesRegex(ValueError,'APPROXIMATE'): prepare(args)

    def test_oom_is_reported_as_failure(self):
        from scripts.run_graph_occlusion_research import record_failure
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); record_failure(path,torch.cuda.OutOfMemoryError('test allocation failure'))
            self.assertTrue(json.loads((path/'failure.json').read_text())['oom'])
            self.assertEqual(json.loads((path/'status.json').read_text())['status'],'FAILED')

    def test_graph_has_real_finite_training_gradients(self):
        torch.manual_seed(2); model=FragmentGraph(3,8,2,True)
        x=torch.randn(3,3); adjacency=torch.full((3,3),1/3); edges=torch.tensor([[0,1],[1,2]])
        node,edge=model(x,adjacency,edges)
        loss=balanced_binary_loss(node,torch.tensor([1.,1.,0.]),torch.ones(3,dtype=torch.bool))
        loss=loss+balanced_binary_loss(edge,torch.tensor([1.,0.]),torch.ones(2,dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__': unittest.main()
