import torch

from models.unet_adv import AttentionGate, UpBlock


def test_attention_gate_preserves_skip_channels():
    gate = AttentionGate(
        g_ch=128,
        x_ch=64,
        out_ch=128,
    )
    gate.eval()

    g = torch.randn(2, 128, 16, 16)
    x = torch.randn(2, 64, 16, 16)

    with torch.inference_mode():
        y = gate(g, x)

    assert y.shape == x.shape
    assert gate.psi.out_channels == 1
    assert torch.isfinite(y).all()


def test_upblock_accepts_different_skip_channel_count():
    block = UpBlock(
        dec_in_ch=256,
        enc_skip_ch=64,
        out_ch=128,
    )
    block.eval()

    decoder = torch.randn(2, 256, 8, 8)
    skip = torch.randn(2, 64, 16, 16)

    with torch.inference_mode():
        y = block(decoder, skip)

    assert y.shape == (2, 128, 16, 16)
    assert block.block.conv1.in_channels == 192
    assert torch.isfinite(y).all()
