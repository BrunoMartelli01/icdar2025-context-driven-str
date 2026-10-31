import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recognition.uniform_encoder_context import project, raw_context
from scripts.run_uniform_unet_training import write
from segmentation.uniform_unet_fusion import file_hash


def test_context_bank_binds_actual_features_to_rgb_identity(tmp_path):
    matrix=np.ones((1,4096),np.float32);np.save(tmp_path/'features.npy',matrix)
    write(tmp_path/'index.json',dict(s=dict(row=0,rgb_sha256='expected')))
    c=dict(encoder_context=str(tmp_path),pinned_inputs={str(tmp_path/n):file_hash(tmp_path/n) for n in ['features.npy','index.json']})
    np.testing.assert_array_equal(raw_context(c,dict(s=dict(rgb_sha256='expected')))['s'],matrix[0])
    with pytest.raises(ValueError,match='another RGB'):
        raw_context(c,dict(s=dict(rgb_sha256='other')))
    assert raw_context({}, {}) is None


def test_projection_requires_real_finite_encoder_width():
    x=np.random.default_rng(1).normal(size=(8,4096))
    model=make_pipeline(StandardScaler(),PCA(n_components=2,svd_solver='full')).fit(x)
    assert project(model,x[0]).shape==(2,)
    with pytest.raises(ValueError,match='4096'):
        project(model,np.zeros(64))
    x[0,1]=np.nan
    with pytest.raises(ValueError,match='Finite'):
        project(model,x[0])
