import torch

from models.joint_word_instance_unet import JointWordInstanceUNet
from models.unet_dense_instances import DenseWordInstances
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.unet_dense_instances import center_targets, dense_instance_loss


def test_dense_instance_loss_updates_unet_decoder_but_keeps_encoder_frozen():
    torch.set_num_threads(2)
    torch.manual_seed(8)
    base = WordRefitUNet(partial_encoder=False, skip_floor=.25, initial_gap_bias=-2.)
    head = DenseWordInstances(input_channels=66, width=8, groups=2, dilations=[1],
        initial_probability=.01, initial_size=[.2, .1])
    net = JointWordInstanceUNet(base, head, train_decoder=True).train()
    output = net(torch.randn(1, 3, 64, 64))
    assert output['segmentation'].shape == (1, 2, 32, 32)
    assert output['center'].shape == (1, 1, 16, 16)
    target = center_targets([[.2, .3, .8, .6]], torch.ones(16, 16),
                            radius_fraction=1/6, minimum_radius=0)
    settings = dict(focal_alpha=2., focal_beta=4., log_size_bounds=[-9., .7],
                    center_weight=1., size_weight=1., offset_weight=1., giou_weight=1.)
    loss, _ = dense_instance_loss(output, [target], settings)
    loss.backward()
    assert base.up0.block.conv1.weight.grad is not None
    assert base.up0.block.conv1.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in base.encoder.parameters())
    assert not base.encoder.training and base.up0.training
