"""Slot/pixel leakage, exact prior reuse, sampling and conflicting-gradient tests."""
import unittest
import torch
from torch.nn import functional as F

from models.word_slot_unet import WordSlotUNet, sampling_grid, deterministic_sample
from segmentation.word_slot_objectives import (instance_loss, make_denoising,
                                               denoising_loss, pcgrad_two,
                                               character_contrastive)


class WordSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        cls.model = WordSlotUNet(slots=4, width=8, heads=2, key_size=8,
                                  skip_floor=.25, strip_points=4)
        cls.pyramid = tuple(torch.randn(1, c, s, s) for c, s in zip((512, 256, 128, 64), (4, 8, 16, 32)))

    def target(self):
        masks = torch.zeros(2, 16, 16); masks[0, 2:7, 1:8] = 1; masks[1, 10:14, 8:15] = 1
        return {"boxes": torch.tensor([[.05,.10,.50,.45],[.50,.60,.95,.90]]),
                "masks": masks, "validity": torch.ones(16,16)}

    def test_shapes_backprop_and_all_original_tensors_are_frozen(self):
        model = self.model.train()
        out = model.decode_pyramid(self.pyramid)
        self.assertEqual(out['boxes'].shape,(4,4)); self.assertEqual(out['masks'].shape,(4,32,32))
        loss, pair, _ = instance_loss(out['boxes'],out['presence'],F.interpolate(out['masks'][None],(16,16),mode='nearest')[0],
                                     self.target(),no_object_weight=.1,mask_weight=1)
        loss.backward()
        self.assertEqual(len(pair[0]),2)
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if n in model.base_names))
        for block in model.exchange:
            self.assertGreater(float(block.pixel_out.weight.grad.abs().sum()),0)
        self.assertFalse(model.up0.training)
        model.zero_grad(set_to_none=True)

    def test_denoising_cannot_leak_through_either_queries_or_pixels(self):
        model = self.model.train()
        plain = model.decode_pyramid(self.pyramid)
        target = self.target()
        dn = make_denoising(target,groups=2,noise=.4,background_queries=2,generator=torch.Generator().manual_seed(2))
        noisy = model.decode_pyramid(self.pyramid,denoising=dn)
        for name in ('boxes','presence','masks','features','queries'):
            torch.testing.assert_close(plain[name],noisy[name],rtol=1e-5,atol=2e-6)
        value = denoising_loss(noisy,target,dn,no_object_weight=.1,mask_weight=1)
        self.assertTrue(torch.isfinite(value))
        model.eval()
        with self.assertRaises(ValueError):model.decode_pyramid(self.pyramid,denoising=dn)

    def test_prior_loading_requires_every_key(self):
        state = {k:v for k,v in self.model.state_dict().items() if k in self.model.base_names}
        audit = self.model.initialize_prior(state)
        self.assertTrue(audit['strict_load'])
        broken = dict(state);del broken[next(iter(broken))]
        with self.assertRaises(ValueError):self.model.initialize_prior(broken)

    def test_dense_training_groups_are_diverse_and_do_not_change_normal_outputs(self):
        model=self.model.train();plain=model.decode_pyramid(self.pyramid)
        dense=model.decode_pyramid(self.pyramid,dense_groups=2,dense_offsets=torch.full((1,4,4),.1))
        torch.testing.assert_close(plain['boxes'],dense['boxes'],rtol=1e-5,atol=2e-6)
        self.assertFalse(torch.allclose(dense['all_boxes'][:4],dense['all_boxes'][4:8]))
        model.eval()
        with self.assertRaises(ValueError):model.decode_pyramid(self.pyramid,dense_groups=2)

    def test_identity_and_zero_offset_sampler_have_exact_pixel_centers(self):
        image = torch.arange(3*8*16,dtype=torch.float32).reshape(3,8,16)
        boxes = torch.tensor([[0.,0.,1.,1.]],requires_grad=True)
        grid = sampling_grid(boxes,None,height=8,width=16,strength=.2)
        actual = deterministic_sample(image,grid,'cpu')
        torch.testing.assert_close(actual[0],image,rtol=0,atol=0)
        offsets = torch.zeros(1,2,4,2,requires_grad=True)
        curved = sampling_grid(boxes,offsets,height=8,width=16,strength=.2)
        torch.testing.assert_close(grid,curved,rtol=0,atol=0)
        deterministic_sample(image,curved,'cpu').sum().backward()
        self.assertGreater(float(offsets.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_hand_calculated_pcgrad_and_private_parameters(self):
        p = torch.nn.Parameter(torch.zeros(2)); private = torch.nn.Parameter(torch.zeros(()))
        # g1=(1,0), g2=(-1,1); projected sum=(.5,1.5).
        evidence = pcgrad_two(p[0]+private,-p[0]+p[1],[p,private])
        torch.testing.assert_close(p.grad,torch.tensor([.5,1.5]))
        self.assertEqual(float(private.grad),1);self.assertTrue(evidence['conflict'])
        q=torch.nn.Parameter(torch.zeros(2))
        evidence=pcgrad_two(q[0],q.sum(),[q])
        torch.testing.assert_close(q.grad,torch.tensor([2.,1.]));self.assertFalse(evidence['conflict'])

    def test_empty_targets_do_not_disappear_from_background_loss(self):
        target=self.target(); target['boxes']=target['boxes'][:0];target['masks']=target['masks'][:0]
        logits=torch.zeros(4,requires_grad=True)
        loss,pair,_=instance_loss(torch.rand(4,4),logits,torch.randn(4,16,16),target,no_object_weight=.1,mask_weight=1)
        self.assertTrue(torch.isfinite(loss));self.assertEqual(len(pair[0]),0)
        loss.backward();self.assertTrue((logits.grad>0).all())

    def test_same_character_is_not_a_false_contrastive_negative(self):
        features=torch.eye(3,requires_grad=True)
        loss=character_contrastive(features,features.detach(),torch.tensor([1,1,1]),.1)
        self.assertEqual(float(loss),0)


if __name__=='__main__':unittest.main()
