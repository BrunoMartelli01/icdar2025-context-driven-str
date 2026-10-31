import numpy as np
import pytest

from recognition.uniform_utility_fusion import paired_f1_lower


def test_paired_bound_identical_outputs_is_zero_and_constant_improvement_is_positive():
    base = np.array([[1, 1, 1], [1, 1, 1]])
    indices = np.array([[0, 0], [0, 1], [1, 1]])
    assert paired_f1_lower(base, base, indices, .05) == 0
    better = np.array([[2, 0, 0], [2, 0, 0]])
    assert paired_f1_lower(base, better, indices, .05) == pytest.approx(.5)


def test_paired_bound_retains_images_with_empty_denominators():
    base = np.array([[0, 0, 0], [1, 1, 1]])
    candidate = np.array([[0, 1, 0], [1, 0, 1]])
    indices = np.array([[0, 0], [1, 1]])
    assert np.isfinite(paired_f1_lower(base, candidate, indices, .05))


def test_paired_bound_rejects_unaligned_image_counts():
    with pytest.raises(ValueError, match='Aligned'):
        paired_f1_lower(np.zeros((2,3)), np.zeros((3,3)), np.array([[0,1]]), .05)
import pytest


@pytest.mark.parametrize('recipes', [[], [dict(kind='ridge', alpha=1.), dict(kind='hgb')]])
def test_explicit_count_family_cannot_silently_default_or_mix_algorithms(tmp_path, recipes):
    from scripts.run_uniform_unet_training import write
    from scripts.train_uniform_utility_fusion import train
    config = tmp_path/'approximate/config.json'
    write(config, dict(mode='approximate', pinned_inputs={}, count_recipes=recipes))
    with pytest.raises(ValueError, match='single shared count-model family'):
        train(config, 'icdar13')
