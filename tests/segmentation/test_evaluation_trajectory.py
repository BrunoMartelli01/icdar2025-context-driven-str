from __future__ import annotations

from segmentation.evaluate_trajectory import trajectory_summary


def _record(epoch: int, iou: float, f1: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "macro_foreground_iou": iou,
        "macro_foreground_f1": f1,
    }


def test_test_trajectory_is_leaky_and_cannot_select_a_checkpoint() -> None:
    summary = trajectory_summary(
        [_record(1, 0.4, 0.5), _record(2, 0.7, 0.6), _record(3, 0.6, 0.8)],
        source_epoch_count=100,
    )
    assert summary["result_class"] == "leaky test-as-validation diagnostic"
    assert summary["test_set_reused_across_epochs"] is True
    assert summary["checkpoint_selection_permitted"] is False
    assert summary["table3_claim_permitted"] is False
    assert summary["complete"] is False
    assert summary["highest_observed_macro_iou"]["epoch"] == 2
    assert summary["highest_observed_macro_f1"]["epoch"] == 3


def test_complete_trajectory_requires_all_source_epochs() -> None:
    records = [_record(epoch, epoch / 100.0, epoch / 101.0) for epoch in range(1, 101)]
    summary = trajectory_summary(records, source_epoch_count=100)
    assert summary["complete"] is True
    assert summary["completed_epoch_count"] == 100
