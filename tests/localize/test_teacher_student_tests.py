"""Scientific boundary tests for the isolated teacher/student pilots."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from localize.teacher_student_tests import (ema_update,confidence_consistency,token_distillation,
    occlude_words,grouped_set_loss,compatible_selection,teacher_matches)


def proposal(identity,box,method='baseline',parents=()):
    return dict(id=identity,bbox=box,method=method,parents=list(parents),crop_array_sha256=identity)


class TeacherStudentTests(unittest.TestCase):
    def test_ema_values_buffers_and_no_teacher_gradient(self):
        student=nn.BatchNorm1d(2);teacher=copy.deepcopy(student).requires_grad_(False)
        with torch.no_grad():student.weight.fill_(3);student.num_batches_tracked.fill_(9)
        ema_update(teacher,student,.5)
        self.assertTrue(torch.equal(teacher.weight,torch.full((2,),2.)))
        self.assertEqual(int(teacher.num_batches_tracked),9)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_ema_fails_on_mismatched_states(self):
        with self.assertRaises(ValueError):ema_update(nn.Linear(2,1),nn.Linear(3,1),.9)
        with self.assertRaises(ValueError):ema_update(nn.Linear(2,1),nn.Linear(2,1),1.)

    def test_consistency_ignores_uncertain_and_invalid(self):
        logits=torch.zeros((1,2,1,3),requires_grad=True)
        teacher=torch.tensor([[[[.95,.5,.95]],[[.05,.5,.05]]]],requires_grad=True)
        valid=torch.tensor([[[[1.,1.,0.]]]])
        loss=confidence_consistency(logits,teacher,valid,.9);loss.backward()
        self.assertGreater(float(logits.grad[:,:,:,0].abs().sum()),0.)
        self.assertEqual(float(logits.grad[:,:,:,1:].abs().sum()),0.)
        self.assertIsNone(teacher.grad)

    def test_consistency_empty_is_differentiable_zero(self):
        logits=torch.zeros((1,2,2,2),requires_grad=True)
        loss=confidence_consistency(logits,torch.full_like(logits,.5),torch.ones((1,1,2,2)),.9)
        loss.backward();self.assertEqual(float(loss),0.)

    def test_token_kl_padding_detach_and_identity(self):
        student=torch.randn(2,4,5,requires_grad=True);teacher=student.detach().clone().requires_grad_(True)
        target=torch.tensor([[0,1,2,9],[2,0,9,9]])
        loss=token_distillation(student,teacher,target,9,2.)
        self.assertLess(abs(float(loss)),1e-5)
        loss.backward();self.assertIsNone(teacher.grad)
        self.assertEqual(float(student.grad[0,3].abs().sum()),0.)

    def test_occlusion_determinism_and_limited_width(self):
        x=torch.ones((4,3,32,128));g=torch.Generator().manual_seed(17)
        a=occlude_words(x,g,1.,.08);b=occlude_words(x,torch.Generator().manual_seed(17),1.,.08)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(x,torch.ones_like(x)))
        self.assertTrue(torch.all((a==0).sum((1,2,3))==3*32*10))

    def test_grouped_loss_penalizes_duplicate_occupancy(self):
        logits=torch.tensor([4.,4.,-4.],requires_grad=True);targets=torch.tensor([1.,1.,0.]);groups=torch.tensor([0,0,-1])
        a=grouped_set_loss(logits,targets,groups,1,0.,0.)
        b=grouped_set_loss(logits,targets,groups,1,1.,.25)
        self.assertGreater(float(b),float(a));b.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_compatible_split_siblings_and_parent(self):
        a=proposal('a',[0,0,20,10]);left=proposal('l',[0,0,9,10],'split',['a']);right=proposal('r',[11,0,20,10],'split',['a'])
        kept=compatible_selection([a,left,right],{'a':.6,'l':.9,'r':.8},.1,.6)
        self.assertEqual([r['id'] for r in kept],['l','r'])

    def test_compatible_duplicate_boxes(self):
        a=proposal('a',[0,0,10,10]);b=proposal('b',[1,0,11,10])
        self.assertEqual(len(compatible_selection([a,b],{'a':.8,'b':.7},.5,.6)),1)

    def test_pseudo_requires_position_text_and_confidence(self):
        a=proposal('a',[0,0,10,10]);b=proposal('b',[1,0,11,10])
        readings={'a':dict(raw_text='WORD',token_probabilities=[.95]*4),'b':dict(raw_text='word',token_probabilities=[.95]*4)}
        self.assertEqual(len(teacher_matches([a],[b],readings,.7,.85)),1)
        readings['b']['raw_text']='wrong'
        self.assertEqual(teacher_matches([a],[b],readings,.7,.85),[])
        readings['b']['raw_text']='word';readings['b']['token_probabilities']=[.1]
        self.assertEqual(teacher_matches([a],[b],readings,.7,.85),[])

    def test_pseudo_one_to_one(self):
        a=proposal('a',[0,0,10,10]);b=proposal('b',[0,0,10,10]);q=proposal('q',[0,0,10,10])
        readings={k:dict(raw_text='x',token_probabilities=[.99]) for k in ('a','b','q')}
        self.assertEqual(len(teacher_matches([a,b],[q],readings,.7,.85)),1)

    def test_strict_and_cross_namespace_fail_before_execution(self):
        from scripts.run_teacher_student_tests import context,REPO
        with patch('scripts.run_teacher_student_tests.read',return_value={'reproduction_mode':'approximate','experiment_kind':'teacher_student_tests_v1'}):
            for mode,namespace in (('strict','approximate'),('approximate','strict'),('approximate','demo')):
                args=SimpleNamespace(config=REPO/'configs'/namespace/'fixture.json',mode=mode)
                with self.assertRaisesRegex(ValueError,'APPROXIMATE'):context(args)

    def test_supervised_maps_ignore_invalid_pixels(self):
        from scripts.run_teacher_student_tests import supervised_loss
        logits=torch.zeros((1,2,2,2),requires_grad=True)
        batch=dict(mask=torch.zeros((1,1,2,2)),kernel=torch.zeros((1,1,2,2)),validity=torch.tensor([[[[1.,0.],[1.,0.]]]]),separation=torch.zeros((1,1,2,2)))
        loss=supervised_loss(logits,batch,dict(bce_weight=1.,dice_weight=1.,smooth=1.,separation_weight=3.))
        loss.backward();self.assertEqual(float(logits.grad[:,:,:,1].abs().sum()),0.)


if __name__=='__main__':unittest.main()
