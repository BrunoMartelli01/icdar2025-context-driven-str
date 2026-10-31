"""Source-loss fidelity, invalid-pixel gradients, emissions and split isolation."""
import pytest
import torch

from segmentation.icdar13_matched_training import matched_word_loss, emit_words, choose_selection
from segmentation.adaspot_gap import gap_loss


def test_restored_loss_exactly_matches_qualified_totaltext_and_ignores_padding():
    torch.manual_seed(4)
    logits = torch.randn(2, 2, 8, 8, requires_grad=True)
    target = torch.zeros(2, 1, 8, 8); target[:, :, 2:6, 2:6] = 1
    kernel = torch.zeros_like(target); kernel[:, :, 3:5, 3:5] = 1
    valid = torch.ones_like(target); valid[:, :, 0] = 0
    separator = torch.zeros_like(target); separator[:, :, 6] = 1
    batch = dict(mask=target, kernel=kernel, validity=valid, separation=separator, gap_target=torch.zeros_like(target))
    config = dict(separation_weight=3., kernel_loss_weight=1., source_loss=dict(bce_weight=1., dice_weight=1., smooth=1.))
    restored = matched_word_loss(logits, batch, config, "totaltext_bce_dice")
    expected, _ = gap_loss(dict(words=logits, gap=None), batch,
        dict(**config["source_loss"], separation_weight=3., lambda_text=1., lambda_kernel=1., lambda_gap=1.))
    assert torch.equal(restored, expected)
    restored.backward()
    assert logits.grad[:, :, 0].count_nonzero() == 0
    assert logits.grad[:, :, 1:].abs().sum() > 0
    with pytest.raises(ValueError): matched_word_loss(logits, batch, config, "guessed")


def test_emission_rejects_empty_and_punctuation_before_scoring_at_inclusive_cut():
    rows = [dict(raw_text=s, sequence_confidence_native=p) for s, p in [("", 1.), ("...", 1.), ("Café!", .9), ("A", .89)]]
    assert [r["raw_text"] for r in emit_words(rows, .9)] == ["Café!"]
    with pytest.raises(ValueError): emit_words([dict(raw_text="A", sequence_confidence_native=float("nan"))], .9)


def test_only_selection_partition_can_choose_winner_and_earlier_tie_wins():
    rows = [dict(split="selection", counts=dict(f1=.8), name=n) for n in ("source", "adapted")]
    assert choose_selection(rows)["name"] == "source"
    with pytest.raises(ValueError): choose_selection(rows + [dict(split="test", counts=dict(f1=.99))])
