"""Real encoder compatibility and trainable U-Net-free decoder qualification."""
import unittest
import torch
from torchvision.models import resnet50
from models.word_graph_decoder import GeometricProposalDecoder


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(3)
        cls.state=resnet50(weights=None).state_dict()

    def test_no_unet_modules_shapes_gradients_and_frozen_encoder(self):
        model=GeometricProposalDecoder(self.state,channels=4,native_size=32).train()
        self.assertFalse(model.requires_unet);self.assertFalse(model.encoder.training)
        self.assertFalse(any('unet' in type(m).__name__.lower() or 'attentiongate' in type(m).__name__.lower() for m in model.modules()))
        output=model.forward_evidence(torch.randn(1,3,64,64))
        self.assertEqual(output['words'].shape,(1,2,32,32));self.assertEqual(output['geometry'].shape,(1,3,32,32))
        (output['words'].square().mean()+output['geometry'].square().mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertGreater(float(model.word.weight.grad.abs().sum()),0.)
        self.assertTrue(all(torch.equal(v,self.state[k]) for k,v in model.encoder.state_dict().items()))

    def test_missing_encoder_tensor_is_not_silently_accepted(self):
        state=dict(self.state);del state['layer4.0.conv1.weight']
        with self.assertRaises(RuntimeError):GeometricProposalDecoder(state,channels=4,native_size=32)

    def test_wrong_coordinate_contract_fails(self):
        model=GeometricProposalDecoder(self.state,channels=4,native_size=32)
        with self.assertRaises(ValueError):model(torch.zeros(1,3,64,96))


if __name__=='__main__':unittest.main()
