import numpy as np
import pytest
import torch

from localize.word_detector_training import normalized_box, mask_training_ignore, strict_initialization, training_roles


def test_exclusive_word_box_normalization_roundtrips():
    x, y, w, h = normalized_box([2, 4, 18, 14], [20, 20])
    assert [20*(x-w/2), 20*(y-h/2), 20*(x+w/2), 20*(y+h/2)] == pytest.approx([2, 4, 18, 14])
    with pytest.raises(ValueError, match='exclusive'):
        normalized_box([2, 4, 2, 14], [20, 20])


def test_training_ignore_preserves_care_pixels_and_original_image():
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    validity = np.ones((3, 4), bool); validity[0, 0] = False
    result = mask_training_ignore(rgb, validity, 128)
    assert np.all(result[0, 0] == 128)
    np.testing.assert_array_equal(result[validity], rgb[validity])
    assert not np.array_equal(result, rgb)


def test_detector_strict_initialization_rejects_missing_or_wrong_tensors():
    model = torch.nn.Linear(3, 2); state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert strict_initialization(model, state)['strict']
    with pytest.raises(ValueError, match='missing'):
        strict_initialization(model, {'weight': state['weight']})
    with pytest.raises(ValueError, match='shapes'):
        strict_initialization(model, dict(state, weight=torch.ones(2, 4)))


def test_final_fixed_epoch_refit_includes_validation_and_reports_overlap():
    parts = dict(fit=['a', 'b'], selection=['c'], calibration=['d'])
    held = training_roles(parts, 'heldout_selection')
    final = training_roles(parts, 'all_train_fixed_epochs')
    assert held == dict(train=['a', 'b'], validation=['c', 'd'])
    assert final == dict(train=['a', 'b', 'c', 'd'], validation=['c', 'd'])
    with pytest.raises(ValueError, match='disjoint'):
        training_roles(dict(fit=['a'], selection=['a'], calibration=[]), 'all_train_fixed_epochs')
    with pytest.raises(ValueError, match='protocol'):
        training_roles(parts, 'unknown')
