from __future__ import annotations

import numpy as np

from localize.line_word_blocks import prepare_word_blocks_v2
from scripts.evaluate_supervised_word_boundary_train import (
    SupervisedTrainRecord,
    _adjacent_pairs,
    _map_native_box,
    _native_word_box,
    _preparation_params,
    _raw_boxes_from_labels,
    _supervised_groups,
)


def test_native_coordinate_helpers_and_component_boxes() -> None:
    labels = np.array([[0, 1, 1], [2, 0, 1]], dtype=np.int32)
    assert _raw_boxes_from_labels(labels) == {1: (1, 0, 3, 2), 2: (0, 1, 1, 2)}
    assert _map_native_box((0, 0, 128, 128), 640, 480) == (0, 0, 640, 480)
    assert _native_word_box((0, 0, 640, 480), 640, 480) == (0, 0, 128, 128)


def test_adjacent_classifier_cut_produces_consecutive_groups() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[20:30, 10:15] = 1
    mask[20:30, 20:25] = 1
    mask[20:30, 50:55] = 1
    prepared = prepare_word_blocks_v2(mask, _preparation_params())
    assert len(prepared.line_candidates) == 1
    assignments = {atom.atom_id: 0 for atom in prepared.atoms}
    pairs = _adjacent_pairs(prepared, assignments)
    assert len(pairs) == 2
    assert all(len(pair["features"]) == 10 for pair in pairs)
    assert np.isfinite(np.asarray([pair["features"] for pair in pairs])).all()

    record = SupervisedTrainRecord(
        sample_id="fixture",
        width=128,
        height=128,
        word_boxes=(),
        prepared=prepared,
        pairs=pairs,
        supervised_word_evidence=(),
        v1_groups_before_cap=(),
        v1_retained_raw_ids=frozenset(),
        v1_word_evidence=(),
    )
    groups, retained = _supervised_groups(record, (0.1, 0.9), threshold=0.5)
    assert len(groups) == 2
    assert groups[0]["raw_component_ids"] == (1, 2)
    assert groups[1]["raw_component_ids"] == (3,)
    assert retained == frozenset({(1, 2), (3,)})


def test_ambiguous_adjacent_pair_is_ignored() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:50, 10:15] = 1
    mask[40:50, 20:25] = 1
    prepared = prepare_word_blocks_v2(mask, _preparation_params())
    atom_ids = tuple(atom.atom_id for atom in prepared.atoms)
    pairs = _adjacent_pairs(prepared, {atom_ids[0]: 0, atom_ids[1]: None})
    assert len(pairs) == 1
    assert pairs[0]["label"] is None
