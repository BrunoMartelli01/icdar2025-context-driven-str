from types import SimpleNamespace

import pytest
import torch
from torch import nn

from recognition.parseq_crop_objective import bounded_boxes, sample_boxes, frozen_parseq_loss


def test_exclusive_boxes_sample_exact_pixel_centers_and_propagate_box_gradients():
    rgb = torch.arange(20, dtype=torch.float32).view(1, 1, 4, 5).expand(1, 3, 4, 5)/20
    boxes = torch.tensor([[1., 1., 4., 3.]], requires_grad=True)
    out = sample_boxes(rgb, boxes, (2, 3))
    torch.testing.assert_close(out, rgb[:, :, 1:3, 1:4], rtol=0, atol=1e-6)
    out.sum().backward()
    assert torch.isfinite(boxes.grad).all() and boxes.grad.abs().sum() > 0


def test_bounded_boxes_cannot_invert_or_escape_context_and_zero_is_identity():
    box = torch.tensor([[1., 2., 5., 7.]])
    torch.testing.assert_close(bounded_boxes(box, torch.zeros_like(box), (8, 8), max_fraction=.25), box)
    result = bounded_boxes(box, torch.tensor([[100., 100., -100., -100.]]), (8, 8), max_fraction=.25)
    assert (result[:, 2:] > result[:, :2]).all()
    assert (result >= 0).all() and (result <= 8).all()


class Tokenizer:
    pad_id = 4

    def encode(self, texts, device):
        return torch.tensor([[3, 1, 0]]*len(texts), device=device)


class ToyPARSeq(nn.Module):
    max_label_length = 25

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(1, 3, bias=False)
        with torch.no_grad(): self.head.weight[:, 0] = torch.tensor([1., -1., .25])

    def encode(self, crops): return crops.mean((1, 2, 3)).reshape(-1, 1, 1)

    def decode(self, context, memory, **kw):
        assert kw["tgt_mask"].dtype == torch.bool
        assert kw["tgt_mask"].tolist() == [[False, True], [False, False]]
        return memory.expand(-1, context.shape[1], -1)


def recognizer():
    system = nn.Module(); system.model = ToyPARSeq(); system.tokenizer = Tokenizer()
    return SimpleNamespace(model=system.eval().requires_grad_(False))


def test_teacher_forcing_keeps_reader_frozen_and_backpropagates_to_crop_coordinates():
    reader = recognizer()
    initial = {k: v.clone() for k, v in reader.model.state_dict().items()}
    rgb = torch.linspace(0, 1, 100).view(1, 1, 10, 10).expand(1, 3, 10, 10)
    boxes = torch.tensor([[1., 1., 8., 8.]], requires_grad=True)
    loss = frozen_parseq_loss(reader, sample_boxes(rgb, boxes, (3, 5)), ["a"])
    loss.backward()
    assert boxes.grad.abs().sum() > 0 and torch.isfinite(boxes.grad).all()
    assert all(p.grad is None for p in reader.model.parameters())
    for k, v in reader.model.state_dict().items(): torch.testing.assert_close(v, initial[k], rtol=0, atol=0)


def test_unfrozen_or_training_reader_and_long_labels_fail_explicitly():
    reader = recognizer(); crops = torch.rand(1, 3, 3, 5)
    reader.model.train()
    with pytest.raises(ValueError, match="eval"): frozen_parseq_loss(reader, crops, ["a"])
    reader.model.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"): frozen_parseq_loss(reader, crops, ["a"])
    reader.model.requires_grad_(False)
    with pytest.raises(ValueError, match="length"): frozen_parseq_loss(reader, crops, ["a"*26])
