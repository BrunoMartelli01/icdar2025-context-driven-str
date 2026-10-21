from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.export_train_native_probabilities import (
    _contained,
    _load_reference_mask,
    _sha_array,
    _sha_file,
    _validate_probability_map,
)


def test_probability_contract_is_native_finite_float32() -> None:
    source = np.linspace(0.0, 1.0, 128 * 128, dtype=np.float64).reshape(128, 128)
    value = _validate_probability_map(source)
    assert value.shape == (128, 128)
    assert value.dtype.str == "<f4"
    assert value.flags.c_contiguous


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((127, 128), np.float32),
        np.full((128, 128), np.nan, np.float32),
        np.full((128, 128), -0.01, np.float32),
        np.full((128, 128), 1.01, np.float32),
    ],
)
def test_probability_contract_rejects_invalid_maps(value: np.ndarray) -> None:
    with pytest.raises(RuntimeError):
        _validate_probability_map(value)


def test_reference_mask_hashes_and_containment(tmp_path: Path) -> None:
    mask = np.zeros((128, 128), np.uint8)
    mask[10:20, 30:40] = 1
    path = tmp_path / "mask.png"
    Image.fromarray(mask * 255, mode="L").save(path)
    record = {
        "sample_id": "1",
        "native_mask_path": "mask.png",
        "native_mask_file_sha256": _sha_file(path),
        "native_mask_array_sha256": _sha_array(mask),
        "foreground_pixels": int(mask.sum()),
    }
    assert np.array_equal(_load_reference_mask(tmp_path, record), mask)
    with pytest.raises(ValueError):
        _contained(tmp_path, "../mask.png")


def test_reference_mask_fails_closed_on_array_drift(tmp_path: Path) -> None:
    mask = np.zeros((128, 128), np.uint8)
    path = tmp_path / "mask.png"
    Image.fromarray(mask, mode="L").save(path)
    record = {
        "sample_id": "1",
        "native_mask_path": "mask.png",
        "native_mask_file_sha256": _sha_file(path),
        "native_mask_array_sha256": "0" * 64,
        "foreground_pixels": 0,
    }
    with pytest.raises(RuntimeError, match="array mismatch"):
        _load_reference_mask(tmp_path, record)
