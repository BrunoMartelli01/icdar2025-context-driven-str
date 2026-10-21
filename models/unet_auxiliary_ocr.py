"""Training-only sequence reader on shared native U-Net features (APPROXIMATE)."""

import torch
from torch import nn

from models.adaspot_mask_align import sample_mask_aligned


class AuxiliaryCTCHead(nn.Module):
    """No BN/dropout: matched arms differ only in shared-feature backpropagation."""

    def __init__(self, classes, *, channels=64, hidden=64):
        super().__init__()
        if min(classes, channels, hidden) < 2:
            raise ValueError("Positive reader dimensions and blank+characters required")
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.sequence = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * hidden, classes)

    def forward(self, crops):
        if crops.ndim != 4 or not torch.isfinite(crops).all():
            raise ValueError("Finite NCHW feature crops required")
        features = self.convolution(crops).mean(dim=2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence).transpose(0, 1).log_softmax(-1)


def axis_feature_crops(features, boxes, *, output_hw, detach):
    """Half-open native pixel-edge boxes -> bin-center bilinear feature crops.

    Both arms explicitly sample on CPU, including the detached control. Thus a
    CPU/GPU sampler difference cannot masquerade as the gradient intervention.
    Transfers keep autograd intact; no gradient through discrete GT boxes needed.
    """
    if (features.ndim != 4 or len(features) != 1 or features.dtype != torch.float32
            or len(output_hw) != 2 or any(type(v) is not int or v < 1 for v in output_hw)):
        raise ValueError("Single-image FP32 features and positive output HW required")
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device="cpu").reshape(-1, 4)
    h, w = features.shape[-2:]
    if (not len(boxes) or not torch.isfinite(boxes).all()
            or torch.any(boxes[:, :2] < 0)
            or torch.any(boxes[:, 2:] > boxes.new_tensor([w, h]))
            or torch.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("Nonempty, bounded, positive native boxes required")
    oh, ow = output_hw
    xx = (torch.arange(ow, dtype=torch.float32) + .5) / ow
    yy = (torch.arange(oh, dtype=torch.float32) + .5) / oh
    x = boxes[:, 0, None] + xx * (boxes[:, 2]-boxes[:, 0])[:, None] - .5
    y = boxes[:, 1, None] + yy * (boxes[:, 3]-boxes[:, 1])[:, None] - .5
    grids = torch.stack((x[:, None, :].expand(-1, oh, -1),
                         y[:, :, None].expand(-1, -1, ow)), -1)
    source = features.detach() if detach else features
    return sample_mask_aligned(source.cpu(), grids,
                               deterministic_cpu_backward=False).to(features.device)
