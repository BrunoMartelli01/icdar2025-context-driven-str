import hashlib
import json

import numpy as np
import pytest

from segmentation.word_checkpoint_metrics import native_metrics, selection_records
from scripts.train_totaltext_compact_v2_word_selection import prune_to_winners


def row(epoch, native, bbox, iou):
    return {"epoch": epoch, "validation": {"native_word_f1": native, "bbox_word_f1": bbox, "foreground_iou": iou},
            "checkpoint": f"epoch-{epoch:04d}.pt"}


def test_native_disappeared_owner_and_duplicate_fragment_do_not_inflate_f1():
    owners = np.zeros((8, 8), dtype=np.int32)
    owners[1:3, 1:3] = 1
    pred = owners > 0
    pred[6, 6] = True
    m = native_metrics(pred, owners > 0, np.ones((8, 8), bool), owners, [1, 2], connectivity=8, iou_threshold=.5)
    assert m["native_instance"] == {"tp": 1, "predicted": 2, "gt": 2, "precision": .5, "recall": .5, "f1": .5}
    assert m["missing_gt_at_native"] == 1


def test_fused_words_have_at_most_one_match():
    owners = np.zeros((8, 8), dtype=np.int32)
    owners[1:3, 1:3] = 1
    owners[1:3, 3:5] = 2
    m = native_metrics(owners > 0, owners > 0, np.ones((8, 8), bool), owners, [1, 2], connectivity=8, iou_threshold=.5)
    assert m["native_instance"]["tp"] == 1
    assert m["fused_components_touching_multiple_words"] == 1
    assert m["words_touched_by_fused_components"] == 2


def test_ignore_foreground_does_not_become_a_false_native_component():
    owners = np.zeros((8, 8), dtype=np.int32)
    pred = np.zeros((8, 8), bool)
    pred[2:4, 2:4] = True
    valid = ~pred
    m = native_metrics(pred, np.zeros_like(pred), valid, owners, [], connectivity=8, iou_threshold=.5)
    assert m["native_instance"]["predicted"] == 0


def test_each_signal_can_select_a_different_checkpoint_and_ties_keep_earliest():
    records = [row(1, .9, .1, .1), row(2, .5, .9, .1), row(3, .9, .5, .9), row(4, .1, .1, .1)]
    winners = selection_records(records)
    assert {k: v["epoch"] for k, v in winners.items()} == {
        "best_native_word_f1": 1, "best_bbox_word_f1": 2, "best_foreground_iou": 3, "last": 4}


@pytest.mark.parametrize("records", [[], [row(2, .1, .1, .1)], [row(1, float("nan"), .1, .1)]])
def test_missing_epochs_and_invalid_metrics_fail(records):
    with pytest.raises(ValueError):
        selection_records(records)


def test_retention_bounds_storage_and_preserves_all_independent_winners(tmp_path):
    records = [row(1, .9, .1, .1), row(2, .5, .9, .1), row(3, .5, .5, .9), row(4, .1, .1, .1), row(5, .1, .1, .1)]
    for r in records:
        path = tmp_path / r["checkpoint"]
        path.write_bytes(bytes([r["epoch"]]))
        r["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _, removed = prune_to_winners(tmp_path, records)
    assert removed == ["epoch-0004.pt"]
    assert len(list(tmp_path.glob("*.pt"))) == 4


def test_missing_selected_checkpoint_stops_before_any_deletion(tmp_path):
    records = [row(1, .9, .9, .9), row(2, .1, .1, .1), row(3, .1, .1, .1)]
    for r in records:
        path = tmp_path / r["checkpoint"]
        path.write_bytes(bytes([r["epoch"]]))
        r["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / records[0]["checkpoint"]).unlink()
    with pytest.raises(ValueError):
        prune_to_winners(tmp_path, records)
    assert (tmp_path / records[1]["checkpoint"]).exists()


def test_changed_obsolete_file_stops_before_removing_any_other_obsolete_file(tmp_path):
    records = [row(1, .9, .9, .9), row(2, .1, .1, .1), row(3, .1, .1, .1), row(4, .1, .1, .1)]
    for r in records:
        path = tmp_path / r["checkpoint"]
        path.write_bytes(bytes([r["epoch"]]))
        r["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / records[2]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="obsolete checkpoint changed"):
        prune_to_winners(tmp_path, records)
    assert (tmp_path / records[1]["checkpoint"]).exists()
