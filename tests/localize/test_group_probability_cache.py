import hashlib
import json

import numpy as np
import pytest

from diagnostics.group_coherence_probability_train import read_probability, project_support
from diagnostics.gt_mask_localization_oracle import sha
from pipeline.frozen_segmentation import backproject_binary_mask


def fixture(tmp_path):
    p = np.full((128, 128), .3, dtype=np.float32); p[10:20, 10:20] = .8
    np.save(tmp_path / '100.npy', p)
    array_sha = hashlib.sha256(p.tobytes()).hexdigest()
    meta = {"image_sha256": "image", "segmentation_metadata": {
        "checkpoint": {"sha256": "checkpoint"}, "native_probability_sha256": array_sha}}
    entry = {"sample_id": "100", "checkpoint_sha256": "checkpoint", "image_sha256": "image",
             "file_sha256": sha(tmp_path/'100.npy'), "array_sha256": array_sha}
    (tmp_path/'100.json').write_text(json.dumps(entry))
    return p, (p >= .5).astype(np.uint8), meta, entry


def test_verified_probability_projection_reproduces_mask(tmp_path):
    p, mask, meta, _ = fixture(tmp_path)
    np.testing.assert_array_equal(read_probability(tmp_path, '100', mask, meta), p)
    np.testing.assert_array_equal(project_support(p, (333, 517)) >= .5,
                                  backproject_binary_mask(mask, (333, 517)) > 0)


@pytest.mark.parametrize('field', ['sample_id', 'checkpoint_sha256', 'image_sha256', 'file_sha256', 'array_sha256'])
def test_corrupt_cache_metadata_fails_closed(tmp_path, field):
    p, mask, meta, entry = fixture(tmp_path)
    entry[field] = 'wrong'; (tmp_path/'100.json').write_text(json.dumps(entry))
    with pytest.raises(RuntimeError): read_probability(tmp_path, '100', mask, meta)


def test_historical_probability_hash_and_mask_both_required(tmp_path):
    _, mask, meta, _ = fixture(tmp_path)
    wrong_mask = mask.copy(); wrong_mask[0, 0] = 1
    with pytest.raises(RuntimeError): read_probability(tmp_path, '100', wrong_mask, meta)
    meta['segmentation_metadata']['native_probability_sha256'] = 'wrong'
    with pytest.raises(RuntimeError): read_probability(tmp_path, '100', mask, meta)


def test_missing_cache_cannot_be_fabricated_from_binary(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_probability(tmp_path, '100', np.zeros((128,128), dtype=np.uint8), {})
