import pytest

from scripts.run_icdar13_transfer_duration import choose_epoch
from scripts.audit_icdar13_transfer_duration import f1, norm


def test_checkpoint_selection_uses_validation_and_earliest_tie():
    rows = [dict(epoch=n, split="validation", counts={"f1": f}) for n, f in [(0, .7), (10, .8), (30, .8), (40, .9)]]
    assert choose_epoch(rows, 30)["epoch"] == 10
    assert choose_epoch(rows, 60)["epoch"] == 40
    with pytest.raises(ValueError):
        choose_epoch([dict(epoch=1, split="test", counts={"f1": 1.})], 60)
    with pytest.raises(ValueError):
        choose_epoch([], 60)


def test_independent_metric_normalization_and_micro_counts():
    assert norm("Café! 1") == "café1"
    assert f1([{"tp": 2, "fp": 1, "fn": 0}, {"tp": 0, "fp": 0, "fn": 1}]) == pytest.approx(2 / 3)
